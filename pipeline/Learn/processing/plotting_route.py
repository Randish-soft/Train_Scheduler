"""
Learn/processing/plotting_route.py
----------------------------------

Builds a *section-annotated* GeoPackage for every passenger rail line
in the reference country.

Output files
~~~~~~~~~~~~
pipeline/data/<country-slug>/routes/
    ├── segments.gpkg        (layer=segments)   – every ~250 m segment with attrs
    ├── lines_meta.parquet                     – one row per Line Name
    └── meta.json                             – snapshot

Dependencies
~~~~~~~~~~~~
geopandas, pandas, shapely, osmnx, rasterio, numpy, pyproj, tqdm
(rasterio >= 1.3 for vector sampling)
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio as rio
from rasterio.features import rasterize
from rasterio.sample import sample_gen
from shapely.geometry import LineString, Point
from shapely.ops import split
from tqdm import tqdm

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"

# ---------------------------------------------------------------------- #
# Tunables – adjust later if you get better data
# ---------------------------------------------------------------------- #
# Segment length used to densify & break long lines (metres)
SEG_LEN_M = 250
# Speed bins  (upper bound km/h, label)
SPEED_BINS = [(120, "low"), (200, "medium"), (999, "high")]
# Cost coefficients €/track-m  (placeholder!)
COST_COEF = {"ground": 500, "embankment": 800, "bridge": 3000, "tunnel": 15000}

# ---------------------------------------------------------------------- #
# Dataclass wrapper
# ---------------------------------------------------------------------- #
@dataclass(slots=True)
class RouteArtifacts:
    segments: Path
    lines_meta: Path
    meta: Path

    def as_dict(self) -> Dict[str, Path]:
        return self.__dict__


# ---------------------------------------------------------------------- #
# Public entry
# ---------------------------------------------------------------------- #
def plot(
    inputs: Dict[str, Path | None],
    terrain: Dict[str, Path],
    *,
    country: str,
    refresh: bool = False,
    data_root: Path | str | None = None,
) -> Dict[str, Path]:
    """
    Build route-segments file (or read from cache) and return file paths.

    Parameters
    ----------
    inputs   dict   – output from processing_input.py
    terrain  dict   – output from processing_terrain.py
    country  str
    refresh  bool   – force rebuild
    """
    data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    slug = country.lower().replace(" ", "_")
    out_dir = data_root / slug / "routes"
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_fp = out_dir / "segments.gpkg"
    meta_fp = out_dir / "lines_meta.parquet"
    snap_fp = out_dir / "meta.json"

    if not refresh and seg_fp.exists() and meta_fp.exists():
        LOG.info("🛤   Using cached route file for %s", country)
        return RouteArtifacts(seg_fp, meta_fp, snap_fp).as_dict()

    LOG.info("🛤   Plotting & sectioning routes for %s …", country)

    # 1) Load rail geometry from OSM parquet
    rails = gpd.read_parquet(inputs["osm"]).to_crs(3857)  # type: ignore[arg-type]
    rails = rails[rails["railway"].isin(["rail", "light_rail", "subway"])]

    # 2) Load GTFS shapes (optional) → align OSM lines to service Line Name
    shapes = _load_gtfs_shapes(inputs.get("gtfs"))
    if shapes is not None:
        rails = _snap_osm_to_shapes(rails, shapes)

    # 3) Densify & split into ~250 m segments
    LOG.info("   ⤷ Splitting into %.0f m segments", SEG_LEN_M)
    segs = _explode_and_split(rails, SEG_LEN_M)

    # 4) Classify segment physical form from OSM tags
    segs["form"] = segs["attrs"].apply(_classify_form)

    # 5) Speed binning (from line tag if present, else heuristic by form)
    segs["speed_kph"] = segs["attrs"].apply(_infer_speed)
    segs["speed_class"] = segs["speed_kph"].apply(_classify_speed)

    # 6) Terrain overlay → mean elev & slope per segment
    elev_fp, slope_fp = terrain["elevation"], terrain["slope"]  # type: ignore[index]
    segs[["elev_m", "slope_deg"]] = list(
        tqdm(_sample_rasters(segs.geometry, elev_fp, slope_fp), total=len(segs), desc="Terrain")
    )

    # 7) Cost logging (placeholder linear model)
    segs["cost_est"] = segs.apply(lambda r: r["length_m"] * COST_COEF[r["form"]], axis=1)

    # 8) Platform logging stub (real impl will join station DB)
    segs["platforms"] = 0  # TODO: replace with real counts

    # 9) Persist
    segs.to_file(seg_fp, layer="segments", driver="GPKG")
    _write_lines_meta(segs, meta_fp)

    # 10) Snapshot metadata
    snap = {
        "country": country,
        "segments": int(len(segs)),
        "total_km": float(segs.length.sum() / 1000),
        "speed_bins": dict(segs.groupby("speed_class").size()),
    }
    snap_fp.write_text(json.dumps(snap, indent=2))

    return RouteArtifacts(seg_fp, meta_fp, snap_fp).as_dict()


# ---------------------------------------------------------------------- #
# Implementation helpers
# ---------------------------------------------------------------------- #
def _load_gtfs_shapes(gtfs_path: Path | None) -> gpd.GeoDataFrame | None:
    if gtfs_path is None:
        return None
    shapes_fp = Path(gtfs_path) / "shapes.txt"
    if not shapes_fp.exists():
        return None
    shp = pd.read_csv(shapes_fp)
    shp["geometry"] = gpd.points_from_xy(shp.shape_pt_lon, shp.shape_pt_lat, crs=4326)
    shp = shp.sort_values(["shape_id", "shape_pt_sequence"])
    # dissolve each shape_id to a LineString
    lines = (
        shp.groupby("shape_id")
        .geometry.apply(lambda pts: LineString(list(pts.to_crs(3857).values)))
        .reset_index()
    )
    gdf = gpd.GeoDataFrame(lines, geometry=0, crs=3857).rename(columns={0: "geometry"})
    return gdf


def _snap_osm_to_shapes(rails: gpd.GeoDataFrame, shapes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Spatially join OSM rail edges to nearest GTFS shape (≤400 m) and
    propagate *Line Name* attribute.
    """
    shapes = shapes.rename(columns={"shape_id": "line_name"})
    joined = gpd.sjoin_nearest(rails, shapes[["line_name", "geometry"]], max_distance=400)
    rails["line_name"] = joined["line_name"]
    return rails


def _explode_and_split(rails: gpd.GeoDataFrame, seg_len_m: float) -> gpd.GeoDataFrame:
    rails = rails.reset_index(drop=True).explode(ignore_index=True)
    seg_rows: List[dict] = []
    for idx, row in tqdm(rails.iterrows(), total=len(rails), desc="Explode"):
        line: LineString = row.geometry
        attrs = row.drop("geometry").to_dict()
        # densify
        num_pts = max(2, int(math.ceil(line.length / seg_len_m)))
        dline = LineString([line.interpolate(i / (num_pts - 1), normalized=True) for i in range(num_pts)])
        # split into consecutive pairs
        coords = list(dline.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            seg_rows.append(
                {
                    **attrs,
                    "geometry": LineString([a, b]),
                    "length_m": LineString([a, b]).length,
                    "attrs": attrs,
                }
            )
    return gpd.GeoDataFrame(seg_rows, geometry="geometry", crs=rails.crs)


def _classify_form(attrs: dict) -> str:
    if attrs.get("tunnel") == "yes":
        return "tunnel"
    if attrs.get("bridge") == "yes":
        return "bridge"
    if attrs.get("embankment") == "yes":
        return "embankment"
    return "ground"


def _infer_speed(attrs: dict) -> int:
    tag = attrs.get("maxspeed")
    if tag:
        try:
            return int(tag.split()[0])
        except Exception:  # noqa: BLE001
            pass
    # fallback by form
    form = _classify_form(attrs)
    return {"ground": 120, "embankment": 140, "bridge": 180, "tunnel": 250}[form]


def _classify_speed(kph: int) -> str:
    for ub, label in SPEED_BINS:
        if kph <= ub:
            return label
    return "high"


def _sample_rasters(
    geoms: gpd.GeoSeries, elev_fp: Path, slope_fp: Path
) -> List[Tuple[float, float]]:
    with rio.open(elev_fp) as elev, rio.open(slope_fp) as slope:
        elev_samp = sample_gen(elev, geoms, indexes=1)
        slope_samp = sample_gen(slope, geoms, indexes=1)
        for e, s in zip(elev_samp, slope_samp):
            yield float(e[0]), float(s[0] / 100)  # slope stored ×100


def _write_lines_meta(segs: gpd.GeoDataFrame, out_fp: Path) -> None:
    meta = (
        segs.groupby("line_name")
        .agg(
            km=("length_m", lambda x: x.sum() / 1000),
            mean_speed=("speed_kph", "mean"),
            segments=("length_m", "size"),
        )
        .reset_index()
    )
    meta.to_parquet(out_fp, index=False)
