"""
Learn/optimization/constraint_analyzer.py
=========================================
Generate terrain-, environment- and population-based constraint rasters.

Public API
----------
analyze(terrain: dict, demand: dict, *, country: str, refresh=False) -> dict

Returns
-------
{
  "hard_mask"   : Path   # uint8 COG  (1 = forbid)
  "soft_cost"   : Path   # uint16 COG (add-on €/m)
  "meta"        : Path
}
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import geopandas as gpd
import numpy as np
import rasterio as rio
import requests
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.warp import reproject
from shapely.geometry import mapping
from tqdm import tqdm

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
# ---------------------------------------------------------------------------- #
# Thresholds / tunables
SLOPE_HARD_DEG = 12     # forbid above this
SLOPE_SOFT_DEG = 6      # surcharge between soft & hard
URBAN_DENS_THRESH = 7000  # pop/km² → noise & visual
WETLAND_LC_CODES = {90, 180, 200}  # Copernicus CLC
SOFT_COST_EUR = {
    "urban": 1200, "wetland": 800, "moderate_slope": 600
}
# ---------------------------------------------------------------------------- #


@dataclass(slots=True)
class ConstraintArtifacts:
    hard_mask: Path
    soft_cost: Path
    meta: Path

    def as_dict(self) -> Dict[str, Path]:
        return self.__dict__


# ---------------------------------------------------------------------------- #
def analyze(
    terrain: Dict[str, Path],
    demand_model: Dict[str, Path],
    *,
    country: str,
    refresh: bool = False,
) -> Dict[str, Path]:
    """
    Build two rasters aligned to the DEM:

    • *hard_mask*  (uint8)  – 1 where routing is forbidden  
    • *soft_cost*  (uint16) – €/m surcharge the optimiser can pay to cross
    """
    slug = country.lower().replace(" ", "_")
    out_dir = DATA_ROOT / slug / "constraints"
    out_dir.mkdir(parents=True, exist_ok=True)

    hard_fp = out_dir / "hard_mask.tif"
    soft_fp = out_dir / "soft_cost.tif"
    meta_fp = out_dir / "meta.json"

    if (not refresh) and hard_fp.exists() and soft_fp.exists():
        LOG.info("🚫  Using cached constraint rasters for %s", country)
        return ConstraintArtifacts(hard_fp, soft_fp, meta_fp).as_dict()

    # Open reference DEM (defines grid & CRS)
    with rio.open(terrain["elevation"]) as dem:  # type: ignore[index]
        prof = dem.profile.copy()
        prof.update(dtype="uint8", nodata=0, compress="DEFLATE", tiled=True)
        hard = np.zeros((dem.height, dem.width), dtype=np.uint8)

    # --------------------------------------------------------------------- #
    # 1) HARD – slope >
    # --------------------------------------------------------------------- #
    with rio.open(terrain["slope"]) as slope_ds:  # type: ignore[index]
        slope = slope_ds.read(1).astype(np.float32) / 100
    hard[slope >= SLOPE_HARD_DEG] = 1

    # --------------------------------------------------------------------- #
    # 2) HARD – wetland (Copernicus land-cover)
    # --------------------------------------------------------------------- #
    lc = _download_cop_landcover(terrain["elevation"], country)
    hard[np.isin(lc, list(WETLAND_LC_CODES))] = 1

    # --------------------------------------------------------------------- #
    # 3) SOFT – initialise cost surface
    # --------------------------------------------------------------------- #
    soft = np.zeros_like(hard, dtype=np.uint16)
    soft[(slope >= SLOPE_SOFT_DEG) & (slope < SLOPE_HARD_DEG)] = SOFT_COST_EUR["moderate_slope"]

    # Urban surcharge by population density
    pop_dens = _raster_pop_density(demand_model["zones"], terrain["elevation"])  # type: ignore[index]
    soft[pop_dens >= URBAN_DENS_THRESH] = SOFT_COST_EUR["urban"]

    # Wetland surcharge (if not already forbidden)
    soft[(lc == 180) & (hard == 0)] = SOFT_COST_EUR["wetland"]

    # --------------------------------------------------------------------- #
    # 4) Write rasters (align to DEM grid)
    # --------------------------------------------------------------------- #
    _write_mask_or_cost(hard, prof, hard_fp)
    prof["dtype"] = "uint16"
    _write_mask_or_cost(soft, prof, soft_fp)

    meta = {
        "country": country,
        "slope_hard_deg": SLOPE_HARD_DEG,
        "slope_soft_deg": SLOPE_SOFT_DEG,
        "urban_density_thresh": URBAN_DENS_THRESH,
        "wetland_codes": list(WETLAND_LC_CODES),
        "soft_cost_eur": SOFT_COST_EUR,
    }
    meta_fp.write_text(json.dumps(meta, indent=2))

    return ConstraintArtifacts(hard_fp, soft_fp, meta_fp).as_dict()


# ---------------------------------------------------------------------------- #
# Helper utilities
# ---------------------------------------------------------------------------- #
def _write_mask_or_cost(arr, profile, fp: Path):
    with rio.open(fp, "w", **profile) as dst:
        dst.write(arr, 1)


def _download_cop_landcover(dem_path: Path, country: str) -> np.ndarray:
    """
    Clip Copernicus CLC 2018 (100 m) to DEM grid.
    """
    LOG.info("   ⤷ Copernicus CLC clip …")
    url = (
        "https://download.wallonie.be/opendata/copernicus/landcover/clc2018_v2020_20u1"
        "/CLC2018_V2020_20u1.tif"
    )
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd) / "clc.tif"
        if not tmp.exists():
            r = requests.get(url, timeout=60)
            tmp.write_bytes(r.content)
        with rio.open(tmp) as src, rio.open(dem_path) as ref:
            arr = np.empty((ref.height, ref.width), dtype=np.uint16)
            reproject(
                source=rio.band(src, 1),
                destination=arr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                resampling=Resampling.nearest,
            )
    return arr


def _raster_pop_density(zones_fp: Path, dem_path: Path) -> np.ndarray:
    """
    Burn zone centroids (pop) into the DEM grid with kernel density ≈ 1 km.
    """
    zones = gpd.read_file(zones_fp, layer="zones")
    with rio.open(dem_path) as ref:
        shapes = ((mapping(geom.buffer(500)), pop) for geom, pop in zip(zones.geometry, zones.population))
        ras = rasterize(
            shapes,
            out_shape=(ref.height, ref.width),
            transform=ref.transform,
            fill=0,
            all_touched=True,
            dtype="float32",
        )
        km2_cell = abs(ref.transform.a) * abs(ref.transform.e) / 1_000_000
        return ras / km2_cell
