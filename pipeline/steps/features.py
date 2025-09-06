# pipeline/steps/features.py
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import log  # shared logger from steps.__init__
from ..flows import import_optional

pd = import_optional("pandas")
gpd = import_optional("geopandas")
np = import_optional("numpy")
rasterio = import_optional("rasterio")
rasterio_sample = None
if rasterio:
    try:
        from rasterio.sample import sample_gen as _sg  # type: ignore[attr-defined]
    except Exception:
        _sg = None
    rasterio_sample = _sg

# ---------- Data contracts ----------
FEATURES_DIRNAME = "features"
FEATURES_FILE = "edge_features.parquet"
FEATURES_SUMMARY = "features_summary.json"


@dataclass
class Paths:
    artifacts_dir: Path
    models_dir: Path
    geo_dir: Path
    reports_dir: Path
    runs_dir: Path
    features_dir: Path

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> "Paths":
        try:
            artifacts_root = Path(
                cfg.get("artifacts_dir") or cfg.get("artifacts") or "artifacts"
            )
            p = Paths(
                artifacts_dir=artifacts_root,
                models_dir=artifacts_root / "models",
                geo_dir=artifacts_root / "geo",
                reports_dir=artifacts_root / "reports",
                runs_dir=artifacts_root / "runs",
                features_dir=artifacts_root / FEATURES_DIRNAME,
            )
            for d in [p.artifacts_dir, p.features_dir, p.runs_dir]:
                d.mkdir(parents=True, exist_ok=True)
            return p
        except Exception as e:
            log.error("Failed to construct artifact paths: %s", e, exc_info=True)
            raise

def _read_edges_any(path: Path):
    """
    Robust loader for edges:
      - .parquet  -> geopandas.read_parquet (if available) else pandas.read_parquet
      - .geojson/.gpkg/... -> geopandas.read_file
      - .csv -> pandas.read_csv
    """
    try:
        s = path.suffix.lower()
        if s in (".parquet", ".pq"):
            if gpd:
                return gpd.read_parquet(path)
            if pd:
                return pd.read_parquet(path)
            raise RuntimeError("pandas is required to read parquet.")
        if s in (".csv", ".tsv"):
            if not pd:
                raise RuntimeError("pandas is required to read CSV.")
            return pd.read_csv(path)
        if gpd:
            return gpd.read_file(path)
        raise RuntimeError(f"Unsupported format without GeoPandas: {path}")
    except Exception as e:
        log.error("Failed to load edges from %s: %s", path, e, exc_info=True)
        raise


# ---------- Helpers ----------
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


def _preview(df_like: Any, n: int = 3) -> str:
    try:
        if df_like is None:
            return "<None>"
        if hasattr(df_like, "head"):
            return str(df_like.head(n))
        return str(df_like)[:500]
    except Exception:
        return "<preview unavailable>"


def _ensure_columns(df, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _to_parquet(df, path: Path) -> None:
    try:
        if hasattr(df, "to_parquet"):
            df.to_parquet(path, index=False)
        else:
            # Fallback to CSV if parquet engine missing
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            log.warning("Parquet not available; wrote CSV instead: %s", csv_path)
            return
        log.info("Wrote features parquet: %s", path)
    except Exception as e:
        log.error("Failed to persist features to %s: %s", path, e, exc_info=True)
        raise


def _curvature_from_polyline(geom) -> Optional[float]:
    """
    Very simple curvature proxy: mean of angle change per unit length along the line.
    Works with Shapely LineString/pygeos geometry if GeoPandas is present.
    Returns None if unavailable.
    """
    if not gpd:
        return None
    try:
        from shapely.geometry import LineString
        from shapely.ops import substring  # noqa: F401  (import ensures shapely present)

        if not isinstance(geom, LineString):
            return None
        coords = list(geom.coords)
        if len(coords) < 3:
            return 0.0
        total = 0.0
        moved = 0.0
        for i in range(1, len(coords) - 1):
            x0, y0 = coords[i - 1]
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            v1 = (x1 - x0, y1 - y0)
            v2 = (x2 - x1, y2 - y1)
            # angle between vectors
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            n1 = math.hypot(*v1)
            n2 = math.hypot(*v2)
            if n1 == 0 or n2 == 0:
                continue
            cosang = max(min(dot / (n1 * n2), 1.0), -1.0)
            ang = math.acos(cosang)
            total += ang
            moved += n1
        if moved == 0:
            return 0.0
        return float(total / moved)  # radians per meter (in projected CRS)
    except Exception:
        return None


def _sample_dem_along_edges(edges_gdf, dem_path: Optional[Path]) -> List[Optional[float]]:
    """
    Returns list of mean slope (%) for each edge using a DEM raster if possible.
    If DEM or raster tooling missing, returns None for each edge.
    """
    if not (gpd and rasterio and dem_path and Path(dem_path).exists()):
        log.warning("DEM sampling unavailable (missing GeoPandas/Rasterio or DEM file). Slopes set to None.")
        return [None] * len(edges_gdf)

    try:
        import numpy as _np
        from rasterio.windows import from_bounds
        from shapely.geometry import LineString

        slopes: List[Optional[float]] = []
        with rasterio.open(dem_path) as src:
            tr = src.transform
            arr = src.read(1, masked=True)
            resx = tr.a
            resy = -tr.e if tr.e < 0 else tr.e
            res = float((abs(resx) + abs(resy)) / 2.0)

            for geom in edges_gdf.geometry:
                if not isinstance(geom, LineString):
                    slopes.append(None)
                    continue
                # sample along the line every ~res meters
                length = max(geom.length, res)
                steps = max(int(length / res), 2)
                zs = []
                for t in _np.linspace(0.0, 1.0, steps):
                    x, y = geom.interpolate(t, normalized=True).coords[0]
                    row, col = ~tr * (x, y)
                    row, col = int(row), int(col)
                    if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
                        z = float(arr[row, col]) if arr[row, col] is not _np.ma.masked else _np.nan
                        zs.append(z)
                zs = [z for z in zs if not (_np.isnan(z) or _np.isinf(z))]
                if len(zs) < 2:
                    slopes.append(None)
                    continue
                dz = max(zs) - min(zs)
                slope_pct = 100.0 * dz / length if length > 0 else None
                slopes.append(float(slope_pct) if slope_pct is not None else None)
        return slopes
    except Exception as e:
        log.warning("DEM sampling failed; setting slopes to None. Reason: %s", e)
        return [None] * len(edges_gdf)


# ---------- Main API ----------
def build_features(config: Dict[str, Any], ingest_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build per-edge feature table for downstream training/inference.

    Inputs:
      - config: scenario YAML as dict (expects artifacts_dir, optional DEM path via config['data_paths'])
      - ingest_artifacts: dict produced by ingest.run_ingest with at least:
          {
            "edges_path": "<path to edges file>",   # GeoPackage/GeoJSON/Parquet
            "nodes_path": "<path to nodes file>",   # optional
            "crs": "EPSG:xxxx",                     # optional
            "dem_path": "<path to DEM>",            # optional (can also come from config)
            ...
          }

    Outputs:
      - {
          "features_path": "<artifacts/features/edge_features.parquet>",
          "n_edges": <int>,
          "columns": [...],
          "preview": {...},
        }
    """
    if not pd:
        raise RuntimeError("pandas is required for features step but is not available.")
    paths = Paths.from_config(config)

    # Resolve sources
    edges_path = Path(ingest_artifacts.get("edges_path", ""))
    nodes_path = Path(ingest_artifacts.get("nodes_path", "")) if ingest_artifacts.get("nodes_path") else None
    dem_path_cfg = None
    try:
        dem_path_cfg = Path(config.get("data_paths", {}).get("dem", "")) if config.get("data_paths") else None
    except Exception:
        dem_path_cfg = None
    dem_path_ing = Path(ingest_artifacts.get("dem_path", "")) if ingest_artifacts.get("dem_path") else None
    dem_path = dem_path_ing or dem_path_cfg

    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_path}")

    # Load edges (GeoPandas if available for geometry-based features)
    use_gpd = bool(gpd)
    # Load edges (GeoParquet/GeoJSON/GPKG/CSV)
    try:
        edges = _read_edges_any(edges_path)
        use_gpd = bool(gpd and hasattr(edges, "geometry"))
        log.info("Loaded edges: %s | %d rows | use_gpd=%s", edges_path, len(edges), use_gpd)
    except Exception as e:
        log.error("Failed to load edges from %s: %s", edges_path, e, exc_info=True)
        raise


    # Basic sanity checks
    required_edge_cols = ["edge_id", "u", "v", "length_m"]
    try:
        _ensure_columns(edges, required_edge_cols, "edges")
    except Exception as e:
        # Try to coerce likely names
        rename_map = {}
        if "length" in edges.columns and "length_m" not in edges.columns:
            rename_map["length"] = "length_m"
        if rename_map:
            edges = edges.rename(columns=rename_map)
            _ensure_columns(edges, required_edge_cols, "edges (after rename)")
        else:
            raise

    # Feature engineering
    feature_rows = []
    geom_available = use_gpd and "geometry" in edges.columns

    # Curvature proxy (if geometry)
    curvature_vals: List[Optional[float]] = [None] * len(edges)
    if geom_available:
        try:
            curvature_vals = [_curvature_from_polyline(geom) for geom in edges.geometry]
            log.info("Computed curvature proxy for %d edges.", len(edges))
        except Exception as e:
            log.warning("Curvature computation failed; setting None. %s", e)

    # Slope proxy from DEM (optional)
    slope_vals: List[Optional[float]] = [None] * len(edges)
    if geom_available and dem_path:
        slope_vals = _sample_dem_along_edges(edges, dem_path)
    else:
        if not dem_path:
            log.warning("DEM path not provided; slope features will be None.")
        elif not geom_available:
            log.warning("Geometry missing; cannot compute slope from DEM.")

    # Environment / placeholders
    env_cols = []
    if "env" in edges.columns and edges["env"].notna().any():
        env_cols = ["env"]
    else:
        # optional: infer urban vs non-urban by attribute density if available
        pass

    # Speed and track attributes if present
    extra_cols = [c for c in ["max_speed_kph", "track_count", "structure", "electrified"] if c in edges.columns]

    # Assemble feature frame
    try:
        base_cols = required_edge_cols + extra_cols + env_cols
        # Ensure presence
        for c in base_cols:
            if c not in edges.columns:
                edges[c] = None

        feats = pd.DataFrame({
            "edge_id": edges["edge_id"],
            "u": edges["u"],
            "v": edges["v"],
            "length_m": edges["length_m"],
            "curvature_rad_per_m": curvature_vals,
            "slope_pct": slope_vals,
            "max_speed_kph": edges.get("max_speed_kph", None),
            "track_count": edges.get("track_count", None),
            "structure": edges.get("structure", None),
            "electrified": edges.get("electrified", None),
            "env": edges.get("env", None),
        })
        # Derived helpers
        feats["is_tunnel"] = feats["structure"].astype(str).str.lower().eq("tunnel")
        feats["is_bridge"] = feats["structure"].astype(str).str.lower().eq("bridge")
        feats["is_elevated"] = feats["structure"].astype(str).str.lower().eq("elevated")
        feats["is_urban"] = feats["env"].astype(str).str.lower().eq("urban")
        feats["len_km"] = feats["length_m"].astype(float) / 1000.0
        log.info("Assembled feature table with %d rows and %d columns.", len(feats), feats.shape[1])
    except Exception as e:
        log.error("Feature assembly failed: %s", e, exc_info=True)
        raise

    # Persist outputs
    features_path = paths.features_dir / FEATURES_FILE
    try:
        _to_parquet(feats, features_path)
    except Exception:
        # already logged in _to_parquet; re-raise to fail-fast
        raise

    # Write summary metadata
    try:
        summary = {
            "n_edges": int(len(feats)),
            "columns": list(map(str, feats.columns)),
            "features_path": str(features_path),
            "used_dem": str(dem_path) if dem_path else None,
            "geom_based": bool(geom_available),
            "preview": feats.head(5).to_dict(orient="records"),
        }
        _safe_write_json(paths.runs_dir / FEATURES_SUMMARY, summary)
    except Exception as e:
        log.warning("Failed to write features summary JSON: %s", e, exc_info=True)

    return {
        "features_path": str(features_path),
        "n_edges": int(len(feats)),
        "columns": list(map(str, feats.columns)),
        "preview": feats.head(3).to_dict(orient="records"),
    }
