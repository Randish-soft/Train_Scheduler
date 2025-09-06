# pipeline/steps/ingest.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import log  # shared logger
from ..flows import import_optional

pd = import_optional("pandas")
gpd = import_optional("geopandas")
np = import_optional("numpy")
ox = import_optional("osmnx")         # online OSM fetch (Overpass)
requests = import_optional("requests")
rasterio = import_optional("rasterio")  # for quick DEM sanity checks
elevation = import_optional("elevation")  # downloads SRTM as GeoTIFF (optional)
shapely = import_optional("shapely")

# ----------------- Constants -----------------
INGEST_DIRNAME = "ingest"
EDGES_NAME = "edges.parquet"
NODES_NAME = "nodes.parquet"
INGEST_SUMMARY = "ingest_summary.json"
DEM_NAME = "dem.tif"


# ----------------- Helpers -----------------
def _paths_from_config(cfg: Dict[str, Any]) -> Dict[str, Path]:
    try:
        artifacts_root = Path(cfg.get("artifacts_dir") or cfg.get("artifacts") or "artifacts")
        ingest_dir = artifacts_root / INGEST_DIRNAME
        runs_dir = artifacts_root / "runs"
        ingest_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        return {
            "artifacts": artifacts_root,
            "ingest": ingest_dir,
            "runs": runs_dir,
            "edges_out": ingest_dir / EDGES_NAME,
            "nodes_out": ingest_dir / NODES_NAME,
            "dem_out": ingest_dir / DEM_NAME,
        }
    except Exception as e:
        log.error("Failed to initialize artifact directories: %s", e, exc_info=True)
        raise


def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        log.debug("Wrote JSON: %s", path)
    except Exception as e:
        log.error("Failed to write JSON %s: %s", path, e, exc_info=True)
        raise


def _detect_input_files(cfg: Dict[str, Any]) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Detect user-provided sources. If absent, we will fetch online.
    Returns: (edges_vector, nodes_vector, osm_pbf_placeholder_unused)
    """
    dp = cfg.get("data_paths", {}) or {}
    edges_vec = Path(dp.get("edges", "")) if dp.get("edges") else None
    nodes_vec = Path(dp.get("nodes", "")) if dp.get("nodes") else None

    if edges_vec and not edges_vec.exists():
        log.error("Configured edges file does not exist: %s", edges_vec)
        raise FileNotFoundError(edges_vec)
    if nodes_vec and not nodes_vec.exists():
        log.error("Configured nodes file does not exist: %s", nodes_vec)
        raise FileNotFoundError(nodes_vec)

    # Default lookups in data/input
    if not edges_vec:
        guess = Path("data/input/edges.parquet")
        if guess.exists():
            edges_vec = guess
    if not nodes_vec:
        for guess_name in ("nodes.parquet", "nodes.csv", "nodes.geojson"):
            guess = Path("data/input") / guess_name
            if guess.exists():
                nodes_vec = guess
                break

    return edges_vec, nodes_vec, None


def _write_parquet_or_csv(df, out_path: Path) -> Path:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(df, "to_parquet"):
            df.to_parquet(out_path, index=False)
            log.info("Wrote %s", out_path)
            return out_path
        # fallback
        csv_path = out_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        log.warning("Parquet engine missing; wrote CSV: %s", csv_path)
        return csv_path
    except Exception as e:
        log.error("Failed to write %s: %s", out_path, e, exc_info=True)
        raise


# ----------------- Online fetchers -----------------
def _geocode_place_gdf(place: str):
    """
    Get boundary polygon for a place/country using osmnx (Nominatim). Returns GeoDataFrame or None.
    """
    if not (ox and gpd):
        log.warning("osmnx/geopandas not available; cannot geocode '%s'.", place)
        return None
    try:
        gdf = ox.geocode_to_gdf(place, which_result=None, by_osmid=False)
        if gdf is None or gdf.empty:
            log.error("No geocode result for place: %s", place)
            return None
        # Ensure polygon geometry
        gdf = gdf.to_crs("EPSG:4326")
        return gdf
    except Exception as e:
        log.error("Geocoding failed for '%s': %s", place, e, exc_info=True)
        return None


def _fetch_osm_railways_with_osmnx(polygon_gdf):
    """
    Fetch railway features with OSMnx using Overpass. Supports both:
      - ox.features_from_polygon(...)  (newer OSMnx)
      - ox.geometries_from_polygon(...) (older OSMnx)
    Returns (edges_gdf, nodes_gdf) or (None, None) on failure.
    """
    if not (ox and gpd and shapely):
        log.warning("osmnx/geopandas/shapely not available; cannot fetch OSM railways.")
        return None, None

    from shapely.geometry import LineString, MultiLineString

    try:
        poly = polygon_gdf.geometry.unary_union
        if poly is None:
            log.error("Polygon union failed.")
            return None, None

        tags = {"railway": True}  # grab all railway=* features

        # OSMnx API compatibility
        fetch_fn = getattr(ox, "features_from_polygon", None)
        if fetch_fn is None:
            fetch_fn = getattr(ox, "geometries_from_polygon", None)
        if fetch_fn is None:
            log.error("OSMnx lacks both features_from_polygon and geometries_from_polygon.")
            return None, None

        geoms = fetch_fn(poly, tags)
        if geoms is None or geoms.empty:
            log.error("No railway features returned from Overpass for the polygon.")
            return None, None

        # Keep only linework
        geoms = geoms[geoms.geometry.apply(lambda g: isinstance(g, (LineString, MultiLineString)))].copy()
        if geoms.empty:
            log.error("No line geometries among returned railway features.")
            return None, None

        # Normalize schema → edges
        edges = geoms.reset_index(drop=True).copy()
        # Work in meters for lengths
        edges = edges.to_crs("EPSG:3857")
        edges["edge_id"] = edges.index.astype("int64")
        edges["length_m"] = edges.geometry.length.astype(float)

        # Attribute coercions
        def _coerce_speed(col):
            if col not in edges.columns:
                return None
            s = edges[col].astype(str)
            # pick first numeric in strings like '120;100' or '120 mph'
            return (
                s.str.extract(r"(\d+)", expand=False)
                 .astype(float, errors="ignore")
                 .where(lambda x: x.notna(), None)
            )

        edges["max_speed_kph"] = _coerce_speed("maxspeed")
        edges["track_count"] = edges["tracks"] if "tracks" in edges.columns else None
        edges["structure"] = None
        edges["electrified"] = edges["electrified"] if "electrified" in edges.columns else None
        edges["env"] = None
        edges["u"] = None
        edges["v"] = None

        # Minimal node set (representative points)
        nodes = gpd.GeoDataFrame(
            {"node_id": edges["edge_id"].values, "edge_id": edges["edge_id"].values},
            geometry=edges.geometry.representative_point(),
            crs=edges.crs,
        )

        log.info("Fetched %d railway edges via Overpass/OSMnx.", len(edges))
        return edges, nodes

    except Exception as e:
        log.error("OSM railway fetch failed: %s", e, exc_info=True)
        return None, None



def _download_dem_for_polygon(polygon_gdf, out_tif: Path) -> Optional[Path]:
    """
    Download a DEM covering the polygon:
      1) Prefer 'elevation' package (SRTM) — no API key, downloads a clipped GeoTIFF.
      2) If missing, try a tiny fallback using Open-Elevation sampling grid and rasterize (very coarse).
    Returns path to GeoTIFF or None.
    """
    try:
        out_tif.parent.mkdir(parents=True, exist_ok=True)
        if elevation:
            # Use bounds to clip SRTM (requires 'elevation' binary or python package)
            b = polygon_gdf.to_crs(4326).total_bounds  # (minx, miny, maxx, maxy)
            bounds = (b[0], b[1], b[2], b[3])
            log.info("Downloading SRTM DEM via 'elevation' for bounds: %s", bounds)
            # Configure cache dir inside artifacts
            os.environ.setdefault("ELEVATION_CACHE_DIR", str(out_tif.parent / "elev_cache"))
            # Clip & merge SRTM tiles to out_tif
            elevation.clip(bounds=bounds, output=str(out_tif))
            # Quick validity check
            if rasterio:
                try:
                    with rasterio.open(out_tif) as src:
                        _ = src.read(1, masked=True)
                except Exception as e:
                    log.warning("DEM validity check failed: %s", e)
            return out_tif
        else:
            log.warning("'elevation' package not available; skipping DEM download.")
            return None
    except Exception as e:
        log.error("DEM download failed: %s", e, exc_info=True)
        return None


# ----------------- Public API -----------------
def run_ingest(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize inputs into standardized `edges` and `nodes` artifacts for the pipeline.
    If local files are absent, this function will fetch:
      - Country/region polygon via Nominatim (osmnx)
      - Railway lines via Overpass (osmnx)
      - DEM (SRTM) via `elevation` package
    Config keys of interest:
      country: "Belgium" (preferred) or a place string supported by OSM Nominatim
      data_paths: optional overrides for edges, nodes, dem
      crs: target CRS for saved geometries (default EPSG:3857)
    """
    if not pd:
        raise RuntimeError("pandas is required for ingest step but is not available.")

    paths = _paths_from_config(config)
    crs_target = config.get("crs") or "EPSG:3857"

    edges_vec, nodes_vec, _ = _detect_input_files(config)

    # Decide if we need online fetching
    need_online_edges = edges_vec is None
    need_online_nodes = nodes_vec is None

    edges_df = None
    nodes_df = None
    dem_path = None
    source = None

    # If local provided, read them
    if not need_online_edges:
        try:
            if gpd:
                edges_df = gpd.read_file(edges_vec)
            else:
                edges_df = pd.read_parquet(edges_vec) if str(edges_vec).endswith(".parquet") else pd.read_csv(edges_vec)
            source = str(edges_vec)
            log.info("Loaded local edges: %s (%d rows)", edges_vec, len(edges_df))
        except Exception as e:
            log.warning("Failed to read local edges (%s); will fall back to online. %s", edges_vec, e)
            edges_df = None
            need_online_edges = True

    if not need_online_nodes and nodes_vec:
        try:
            if gpd and nodes_vec.suffix.lower() in (".gpkg", ".geojson"):
                nodes_df = gpd.read_file(nodes_vec)
            else:
                nodes_df = pd.read_parquet(nodes_vec) if str(nodes_vec).endswith(".parquet") else pd.read_csv(nodes_vec)
            log.info("Loaded local nodes: %s (%d rows)", nodes_vec, len(nodes_df))
        except Exception as e:
            log.warning("Failed to read local nodes (%s); will synthesize/fetch. %s", nodes_vec, e)
            nodes_df = None
            need_online_nodes = True

    # ONLINE: geocode + railway fetch
    polygon_gdf = None
    if need_online_edges:
        place = config.get("country") or config.get("place") or config.get("region") or "Belgium"
        polygon_gdf = _geocode_place_gdf(place)
        if polygon_gdf is None:
            raise RuntimeError(f"Could not geocode '{place}' and no local edges were provided.")
        edges_df, nodes_df_auto = _fetch_osm_railways_with_osmnx(polygon_gdf)
        if edges_df is None:
            raise RuntimeError("Online railway fetch failed and no local edges available.")
        if nodes_df is None:
            nodes_df = nodes_df_auto
        source = f"Overpass/OSM for {place}"

    # Reproject (if geometries present)
    geom_based = bool(gpd and hasattr(edges_df, "geometry"))
    if geom_based and gpd:
        try:
            if getattr(edges_df, "crs", None) is None:
                edges_df.set_crs("EPSG:4326", inplace=True)
            if str(edges_df.crs) != str(crs_target):
                edges_df = edges_df.to_crs(crs_target)
                log.info("Reprojected edges to %s", crs_target)

            if nodes_df is not None and hasattr(nodes_df, "geometry"):
                if getattr(nodes_df, "crs", None) is None:
                    nodes_df.set_crs(edges_df.crs, inplace=True)
                if str(nodes_df.crs) != str(edges_df.crs):
                    nodes_df = nodes_df.to_crs(edges_df.crs)
                    log.info("Reprojected nodes to %s", edges_df.crs)
        except Exception as e:
            log.error("CRS handling failed: %s", e, exc_info=True)
            raise

    # Normalize schema (ensure required columns)
    def _ensure_edge_columns(df) -> Any:
        try:
            if "edge_id" not in df.columns:
                df["edge_id"] = df.index.astype("int64")
            if "length_m" not in df.columns:
                if gpd and hasattr(df, "geometry"):
                    df["length_m"] = df.geometry.length.astype(float)
                else:
                    raise ValueError("length_m missing and geometry unavailable to compute it.")
            for c in ("u", "v", "max_speed_kph", "track_count", "structure", "electrified", "env"):
                if c not in df.columns:
                    df[c] = None
            return df
        except Exception as e:
            log.error("Edge schema normalization failed: %s", e, exc_info=True)
            raise

    edges_df = _ensure_edge_columns(edges_df)

    # Nodes: ensure node_id
    if nodes_df is not None:
        if "node_id" not in nodes_df.columns:
            nodes_df = nodes_df.rename(columns={"id": "node_id"}) if "id" in nodes_df.columns else nodes_df
            if "node_id" not in nodes_df.columns:
                nodes_df["node_id"] = nodes_df.index
    else:
        # synthesize minimal nodes if none
        if gpd and hasattr(edges_df, "geometry"):
            try:
                from shapely.geometry import LineString, Point
                starts = edges_df.geometry.apply(lambda ls: ls.coords[0] if hasattr(ls, "coords") else None)
                ends = edges_df.geometry.apply(lambda ls: ls.coords[-1] if hasattr(ls, "coords") else None)
                starts = [s for s in starts if s is not None]
                ends = [e for e in ends if e is not None]
                pts = starts + ends
                nodes_df = gpd.GeoDataFrame({"node_id": range(len(pts))}, geometry=gpd.points_from_xy([p[0] for p in pts], [p[1] for p in pts]), crs=edges_df.crs)
                log.info("Synthesized %d nodes from edge endpoints.", len(nodes_df))
            except Exception as e:
                log.warning("Failed to synthesize nodes; creating empty nodes table. %s", e)
                nodes_df = pd.DataFrame({"node_id": []})

    # Persist edges/nodes
    edges_out = _write_parquet_or_csv(edges_df, paths["edges_out"])
    nodes_out = _write_parquet_or_csv(nodes_df, paths["nodes_out"])

    # DEM: prefer configured file; else fetch online
    dem_cfg = None
    try:
        dem_cfg = config.get("data_paths", {}).get("dem")
    except Exception:
        pass

    if dem_cfg:
        p = Path(dem_cfg)
        if p.exists():
            dem_path = str(p)
            log.info("Using configured DEM: %s", dem_path)
        else:
            log.warning("DEM configured but not found: %s", p)

    if dem_path is None:
        # try to download covering polygon bounds
        if polygon_gdf is None and gpd and hasattr(edges_df, "geometry"):
            try:
                polygon_gdf = gpd.GeoDataFrame(geometry=[edges_df.unary_union.convex_hull], crs=edges_df.crs).to_crs(4326)
            except Exception:
                polygon_gdf = None
        if polygon_gdf is not None:
            tif = _download_dem_for_polygon(polygon_gdf, paths["dem_out"])
            if tif and Path(tif).exists():
                dem_path = str(tif)

    # Summary
    summary = {
        "edges_path": str(edges_out),
        "nodes_path": str(nodes_out),
        "dem_path": dem_path,
        "crs": str(crs_target),
        "geom_based": bool(gpd and hasattr(edges_df, "geometry")),
        "n_edges": int(len(edges_df)) if edges_df is not None else 0,
        "n_nodes": int(len(nodes_df)) if nodes_df is not None else 0,
        "source": source or "unknown",
        "online_fetch": need_online_edges or need_online_nodes or bool(dem_path),
    }
    try:
        _safe_write_json(paths["runs"] / INGEST_SUMMARY, summary)
    except Exception as e:
        log.warning("Failed to write ingest summary JSON: %s", e, exc_info=True)

    log.info("Ingest complete (online-enabled): edges=%s nodes=%s dem=%s",
             summary.get("edges_path"), summary.get("nodes_path"), summary.get("dem_path"))
    return summary
