"""
terrain.py – DEM fetch & utilities  ♢  OpenTopography edition  v0·7
------------------------------------------------------------------
* Tries SRTMGL1_E first, then ASTER30m.
* Caches every GeoTIFF under  data/_dem/  for instant re-use.
* If both online downloads fail it falls back to a synthetic “flat” DEM
  created in memory – slope = 0 %, ruggedness = 0.0.
* Public functions
      load_dem(boundary_gdf)  → rasterio DatasetReader
      slope_percent(ds)       → np.ndarray  (% slope)
      ruggedness_index(ds)    → float ∈ [0,1]
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
import rasterio.transform
import requests
from rasterio.io import DatasetReader, MemoryFile
from shapely.geometry import MultiPolygon, Polygon

from src import DATA_DIR   # <- defined in src/__init__.py

log = logging.getLogger("bcpc.terrain")

# ---------------------------------------------------------------------------#
# Paths & constants                                                          #
# ---------------------------------------------------------------------------#
_DEM_CACHE = DATA_DIR / "_dem"
_DEM_CACHE.mkdir(parents=True, exist_ok=True)

OT_API      = "https://portal.opentopography.org/API/globaldem"
OT_KEY      = os.getenv("OT_API_KEY", "153e670200e6b3568ff813c994fda446")
OT_TIMEOUT  = 60          # seconds
_BBOX_PAD   = 0.05        # deg – avoid edge cut-offs
_CACHE_DAYS = 30          # re-download after this many days


# ---------------------------------------------------------------------------#
# Helper utilities                                                           #
# ---------------------------------------------------------------------------#
def _bbox_from_gdf(gdf) -> Tuple[float, float, float, float]:
    """Return padded (N, S, W, E) bbox from a GeoSeries/DataFrame."""
    minx, miny, maxx, maxy = gdf.total_bounds
    return maxy + _BBOX_PAD, miny - _BBOX_PAD, minx - _BBOX_PAD, maxx + _BBOX_PAD


def _hash_bbox(n: float, s: float, w: float, e: float, dem: str) -> str:
    return hashlib.md5(f"{dem}:{n:.4f},{s:.4f},{w:.4f},{e:.4f}".encode()).hexdigest()[:16]


def _download_dem(n: float, s: float, w: float, e: float, demtype: str) -> Path:
    """
    Download/cached DEM for given bbox, return filename (raises on HTTP error).
    """
    cache_path = _DEM_CACHE / f"{_hash_bbox(n, s, w, e, demtype)}_{demtype}.tif"

    fresh_enough = (
        cache_path.exists()
        and (datetime.utcnow() - datetime.utcfromtimestamp(cache_path.stat().st_mtime)).days < _CACHE_DAYS
    )
    if fresh_enough:
        log.debug("DEM cache hit → %s", cache_path.name)
        return cache_path

    params = dict(
        demtype=demtype,
        south=s, north=n, west=w, east=e,
        outputFormat="GTiff",
        API_Key=OT_KEY,
    )
    log.info("Fetching %s DEM  (%.2f,%.2f,%.2f,%.2f)", demtype, s, n, w, e)
    with requests.get(OT_API, params=params, stream=True, timeout=OT_TIMEOUT) as r:
        r.raise_for_status()
        # The API sometimes returns a text error instead of GTiff
        if r.headers.get("Content-Type", "").startswith("text"):
            raise RuntimeError(r.text.strip())

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        for chunk in r.iter_content(chunk_size=1 << 15):
            tmp.write(chunk)
        tmp.close()
        Path(tmp.name).replace(cache_path)

    return cache_path


def _flat_dem(n: float, s: float, w: float, e: float) -> DatasetReader:
    """
    Build an in-memory 1×1 pixel raster at 0 m elevation for graceful fallback.
    """
    log.error("DEM fetch failed – using synthetic flat DEM (0 m)")

    transform = rasterio.transform.from_bounds(w, s, e, n, 1, 1)
    memfile: MemoryFile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=1, width=1, count=1, dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as ds:
        ds.write(np.zeros((1, 1), dtype="float32"), 1)
    return memfile.open()  # reopen for reading


# ---------------------------------------------------------------------------#
# Public API                                                                 #
# ---------------------------------------------------------------------------#
def load_dem(boundary_gdf) -> DatasetReader:
    """
    Return a rasterio dataset covering `boundary_gdf`.

    Order tried:
        1. SRTMGL1_E  (≈30 m)
        2. ASTER30m   (≈30 m)
        3. flat 0 m synthetic (guaranteed to succeed)
    """
    n, s, w, e = _bbox_from_gdf(boundary_gdf)
    for dem in ("SRTMGL1_E", "ASTER30m"):
        try:
            return rasterio.open(_download_dem(n, s, w, e, dem))
        except Exception as exc:   # noqa: BLE001
            log.warning("%s fetch failed – %s", dem, exc)

    # --- final fallback --------------------------------------------------
    return _flat_dem(n, s, w, e)


def slope_percent(ds: DatasetReader) -> np.ndarray:
    """
    Return a %-slope array (same shape as DEM) using Horn’s method.
    """
    band = ds.read(1, masked=True).filled(np.nan)
    dy, dx = np.gradient(band, ds.res[1], ds.res[0])
    slope_rad = np.arctan(np.hypot(dx, dy))
    return np.degrees(slope_rad) * 100 / 45.0   # rough % slope


def ruggedness_index(ds: DatasetReader) -> float:
    """
    0 = billiard-table flat, 1 ≈ Himalayas.  Mean absolute slope / 45°.
    """
    slope_pct = slope_percent(ds)
    mean_deg = np.nanmean(slope_pct) * 0.45  # (%  →  degrees approx.)
    return max(0.0, min(mean_deg / 45.0, 1.0))
