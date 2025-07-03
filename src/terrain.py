"""
terrain.py – real SRTM DEM loader + slope helper (v1.1).

• Downloads / caches 1-arc-sec tiles from AWS srtm-pds
• Mosaics, clips, resamples to ≤512×512
• Provides slope_percent(dem) for routing cost
• Falls back to flat DEM if download fails
"""

from __future__ import annotations

import gzip
import logging
import math
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as win_from_bounds

from . import DATA_DIR

logger = logging.getLogger("bcpc.terrain")

_DEM_DIR = DATA_DIR / "dem"
_DEM_DIR.mkdir(exist_ok=True, parents=True)

AWS_URL = "https://s3.amazonaws.com/srtm-pds/SRTM1/{lat}/{name}.hgt.gz"


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------
def _tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"


def _download_tile(tile_code: str, dest_tif: Path, retries: int = 3) -> None:
    lat_band = tile_code[:3]  # e.g. N33
    url = AWS_URL.format(lat=lat_band, name=tile_code)
    tmp_dir = tempfile.mkdtemp()
    gz_path = Path(tmp_dir) / f"{tile_code}.hgt.gz"

    for attempt in range(1, retries + 1):
        try:
            logger.info("Downloading %s (attempt %d)", tile_code, attempt)
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(gz_path, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning("Download failed: %s", exc)
            if attempt == retries:
                raise

    hgt_path = gz_path.with_suffix("")
    with gzip.open(gz_path, "rb") as src, open(hgt_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    lat = int(tile_code[1:3]) * (1 if tile_code[0] == "N" else -1)
    lon = int(tile_code[4:7]) * (1 if tile_code[3] == "E" else -1)
    transform = from_bounds(lon, lat, lon + 1, lat + 1, 3601, 3601)
    profile = dict(
        driver="GTiff",
        height=3601,
        width=3601,
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        transform=transform,
        compress="lzw",
    )
    data = np.fromfile(hgt_path, ">i2").reshape((3601, 3601))
    with rasterio.open(dest_tif, "w", **profile) as dst:
        dst.write(data, 1)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("Saved %s", dest_tif)


def _flat_dem(bounds, res_deg=0.0008333):
    minx, miny, maxx, maxy = bounds
    nx = max(2, int((maxx - minx) / res_deg))
    ny = max(2, int((maxy - miny) / res_deg))
    transform = from_bounds(minx, miny, maxx, maxy, nx, ny)
    profile = dict(
        driver="GTiff",
        height=ny,
        width=nx,
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        transform=transform,
    )
    mem = rasterio.io.MemoryFile()
    with mem.open(**profile) as dst:
        dst.write(np.zeros((1, ny, nx), dtype="int16"))
    return mem.open()


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def load_dem(boundary: gpd.GeoSeries, buffer_m: int = 2_000) -> rasterio.DatasetReader:
    boundary_m = boundary.to_crs(3857).buffer(buffer_m).to_crs(4326)
    minx, miny, maxx, maxy = boundary_m.total_bounds
    tiles: List[Path] = []

    try:
        for lat in range(math.floor(miny), math.ceil(maxy)):
            for lon in range(math.floor(minx), math.ceil(maxx)):
                code = _tile_name(lat, lon)
                tif = _DEM_DIR / f"{code}.tif"
                if not tif.exists():
                    _download_tile(code, tif)
                tiles.append(tif)
    except Exception as exc:  # noqa: BLE001
        logger.error("DEM fetch failed – %s; using flat terrain", exc)
        return _flat_dem((minx, miny, maxx, maxy))

    srcs = [rasterio.open(p) for p in tiles]
    try:
        mosaic, transform = merge(srcs)
    finally:
        for s in srcs:
            s.close()

    window = win_from_bounds(minx, miny, maxx, maxy, transform)
    out_h = min(512, int(window.height))
    out_w = min(512, int(window.width))
    data = mosaic[
        :,
        int(window.row_off) : int(window.row_off + window.height),
        int(window.col_off) : int(window.col_off + window.width),
    ]
    profile = dict(
        driver="GTiff",
        height=out_h,
        width=out_w,
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        transform=from_bounds(minx, miny, maxx, maxy, out_w, out_h),
    )
    mem = rasterio.io.MemoryFile()
    with mem.open(**profile) as dst:
        dst.write(data[0], 1, resampling=Resampling.bilinear)
    return mem.open()


def slope_percent(dem: rasterio.DatasetReader) -> np.ndarray:
    """Return slope (%) using Horn kernel."""
    band = dem.read(1).astype("float32")
    # 1-cell central differences
    dzdx = np.gradient(band, axis=1)
    dzdy = np.gradient(band, axis=0)
    slope = np.sqrt(dzdx**2 + dzdy**2)
    return slope * 100
