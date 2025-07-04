"""
terrain.py – DEM fetch & utilities (OpenTopography edition, v0·6)

* Downloads SRTMGL1_E (≈30 m) tiles from OpenTopography’s GlobalDEM API.
* Falls back to ASTER GDEM v3 (≈30 m) if SRTM is unavailable in the area.
* Caches every GeoTIFF under  data/_dem/  so subsequent runs are instant.
* Provides `load_dem(boundary_gdf)`   → rasterio dataset
         and `slope_percent(dem_ds)` → ndarray of % slope for the whole tile.
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
import rasterio.enums
import rasterio.windows
import requests
from rasterio.io import DatasetReader
from shapely.geometry import box

from . import DATA_DIR

logger = logging.getLogger("bcpc.terrain")

_DEM_CACHE = DATA_DIR / "_dem"
_DEM_CACHE.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------#
# Config                                                                     #
# ---------------------------------------------------------------------------#
OT_API = "https://portal.opentopography.org/API/globaldem"
OT_KEY = os.getenv("OT_API_KEY", "153e670200e6b3568ff813c994fda446")  # ← your key
OT_TIMEOUT = 60  # seconds


# ---------------------------------------------------------------------------#
# Helpers                                                                    #
# ---------------------------------------------------------------------------#
def _bbox_from_gdf(gdf) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    pad = 0.05  # deg padding so edges aren’t cut off
    return maxy + pad, miny - pad, minx - pad, maxx + pad  # north, south, west, east


def _hash_bbox(n: float, s: float, w: float, e: float, demtype: str) -> str:
    m = hashlib.md5()
    m.update(f"{demtype}:{n:.4f},{s:.4f},{w:.4f},{e:.4f}".encode())
    return m.hexdigest()[:16]


def _download_dem(n: float, s: float, w: float, e: float, demtype: str) -> Path:
    """
    Download a DEM for the given bbox and return the cached file path.
    """
    cache_name = f"{_hash_bbox(n, s, w, e, demtype)}_{demtype}.tif"
    cache_path = _DEM_CACHE / cache_name
    if cache_path.exists():
        logger.debug("DEM cache hit (%s)", cache_path.name)
        return cache_path

    params = {
        "demtype": demtype,
        "south":  s,
        "north":  n,
        "west":   w,
        "east":   e,
        "outputFormat": "GTiff",
        "API_Key": OT_KEY,
    }
    logger.info("Fetching %s DEM %.2f,%.2f,%.2f,%.2f", demtype, s, n, w, e)
    with requests.get(OT_API, params=params, stream=True, timeout=OT_TIMEOUT) as r:
        r.raise_for_status()
        # The API returns a small text message if the tile is absent – guard it
        if r.headers.get("Content-Type", "").startswith("text"):
            raise RuntimeError(r.text.strip())

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        for chunk in r.iter_content(chunk_size=1 << 16):
            tmp.write(chunk)
        tmp.close()
        Path(tmp.name).rename(cache_path)
    return cache_path


def load_dem(boundary_gdf) -> DatasetReader:
    """
    Return a rasterio dataset covering the boundary.

    Tries SRTMGL1_E first (≈30 m); if that fails, falls back to ASTER GDEM v3.
    """
    n, s, w, e = _bbox_from_gdf(boundary_gdf)
    for dem in ("SRTMGL1_E", "ASTER30m"):
        try:
            tif = _download_dem(n, s, w, e, dem)
            ds = rasterio.open(tif)
            logger.debug("Opened DEM %s", tif.name)
            return ds
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s fetch failed – %s", dem, exc)
    raise RuntimeError("DEM download failed for both SRTMGL1_E and ASTER30m")


def slope_percent(ds: DatasetReader) -> np.ndarray:
    """
    Return a %-slope raster (same shape as DEM) using Horn’s method.
    """
    band1 = ds.read(1, masked=True).filled(np.nan)
    dy, dx = np.gradient(band1, ds.res[1], ds.res[0])
    slope_rad = np.arctan(np.hypot(dx, dy))
    return np.degrees(slope_rad) * 100 / 45  # % slope (approx.)

# ---------------------------------------------------------------------------#
# Ruggedness index                                                            #
# ---------------------------------------------------------------------------#
def ruggedness_index(ds: DatasetReader) -> float:
    """
    Return a single float ∈ [0, 1] describing how rugged the DEM is.

    Simple metric: mean(|slope|) / 45°, capped at 1.0.
    """
    slope_pct = slope_percent(ds)
    mean_deg = np.nanmean(slope_pct) * 0.45  # % → degrees (approx.)
    return max(0.0, min(mean_deg / 45.0, 1.0))

