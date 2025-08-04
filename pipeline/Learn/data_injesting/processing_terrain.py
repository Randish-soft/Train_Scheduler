"""
Learn/data_injesting/processing_terrain.py
-----------------------------------------

Step #2 of the Learn pipeline -- fetch and re-grid terrain data.

Dependencies (pip):
    rasterio, elevation, richdem, numpy, geopandas, shapely, tqdm
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import rasterio as rio
import richdem as rd
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import box
from tqdm import tqdm

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

# Default parent folder is the same one used by processing_input.py
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
NE_ADMIN_GPKG = "admin.gpkg"  # relative to country data dir


# --------------------------------------------------------------------------- #
# Dataclass wrapper
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class TerrainArtifacts:
    elevation: Path          # COG GeoTIFF (int16, metres)
    slope: Path              # COG GeoTIFF (uint16, degrees * 100)
    hillshade: Path          # COG GeoTIFF (uint8)
    bbox: Tuple[float, float, float, float]

    def as_dict(self) -> Dict[str, Path | Tuple[float, float, float, float]]:
        return self.__dict__


# --------------------------------------------------------------------------- #
# PUBLIC API
# --------------------------------------------------------------------------- #
def load_terrain(
    country: str,
    *,
    refresh: bool = False,
    data_root: Path | str | None = None,
    target_res_m: int = 90,
) -> Dict[str, Path]:
    """
    Download + build terrain rasters for *country* if not cached.

    Parameters
    ----------
    country : str
        Plain country name (“Belgium”, “Kenya”…).
    refresh : bool
        Force re-download/re-build even if cache exists.
    data_root : Path | str | None
        Base cache folder (defaults to `pipeline/data`).
    target_res_m : int
        Output resolution in metres (90 is fine for national planning,
        drop to 30 if you need extra detail but expect bigger files).

    Returns
    -------
    dict  – keys `elevation`, `slope`, `hillshade`, `bbox`
    """
    data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    slug = country.lower().replace(" ", "_")
    out_dir = data_root / slug / "terrain"
    out_dir.mkdir(parents=True, exist_ok=True)

    elev_fp = out_dir / f"elevation_{target_res_m}m.tif"
    slope_fp = out_dir / f"slope_{target_res_m}m.tif"
    shade_fp = out_dir / f"hillshade_{target_res_m}m.tif"

    if not refresh and elev_fp.exists() and slope_fp.exists() and shade_fp.exists():
        LOG.info("🗻  Using cached terrain for %s (%s)", country, elev_fp.name)
        bbox = _bbox_from_admin(data_root / slug / NE_ADMIN_GPKG)
        return TerrainArtifacts(elev_fp, slope_fp, shade_fp, bbox).as_dict()

    LOG.info("🗻  Building terrain rasters for %s …", country)

    bbox = _bbox_from_admin(data_root / slug / NE_ADMIN_GPKG)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        raw = _download_dem_tiles(bbox, tmp_dir)
        vrt = _mosaic_to_vrt(raw, tmp_dir / "merged.vrt")
        _warp_to_cog(vrt, elev_fp, res=target_res_m)
        _derive_slope_and_hillshade(elev_fp, slope_fp, shade_fp)

    return TerrainArtifacts(elev_fp, slope_fp, shade_fp, bbox).as_dict()


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _bbox_from_admin(gpkg: Path) -> Tuple[float, float, float, float]:
    """Return minx, miny, maxx, maxy for the outer admin boundary."""
    gdf = gpd.read_file(gpkg, layer="admin")
    return gdf.to_crs(4326).total_bounds  # type: ignore[return-value]


def _download_dem_tiles(bbox, dest: Path) -> list[Path]:
    """
    Use the *elevation* package (CLI) to pull 30 m Copernicus tiles; returns list of .tif paths.
    """
    dest.mkdir(parents=True, exist_ok=True)
    west, south, east, north = bbox
    LOG.info("   ⤷ Fetching Copernicus DEM tiles …")
    subprocess.run(
        [
            "eio",
            "clip",
            "-o",
            str(dest),
            "--bounds",
            f"{west},{south},{east},{north}",
            "--product",
            "COP30",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return list(dest.glob("*.tif"))


def _mosaic_to_vrt(tifs: list[Path], out_vrt: Path) -> Path:
    """Create a VRT that mosaics multiple tiles together."""
    if len(tifs) == 1:
        return tifs[0]
    LOG.info("   ⤷ Mosaicking %d tiles → VRT", len(tifs))
    subprocess.run(
        ["gdalbuildvrt", str(out_vrt)] + [str(t) for t in tifs],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return out_vrt


def _warp_to_cog(src: Path, dst: Path, *, res: int) -> None:
    """
    Reproject + resample DEM to `res` metres, write Cloud-Optimised GeoTIFF.
    """
    LOG.info("   ⤷ Re-projecting & compressing → %s", dst.name)
    with rio.open(src) as fsrc:
        dst_crs = "EPSG:3857"  # WebMercator is fine for world-scale, fast pixel area
        transform, width, height = calculate_default_transform(
            fsrc.crs, dst_crs, fsrc.width, fsrc.height, *fsrc.bounds, resolution=res
        )
        kwargs = fsrc.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "compress": "DEFLATE",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
            }
        )
        with rio.open(dst, "w", **kwargs) as fdst:
            reproject(
                source=rio.band(fsrc, 1),
                destination=rio.band(fdst, 1),
                src_transform=fsrc.transform,
                src_crs=fsrc.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )


def _derive_slope_and_hillshade(elev_fp: Path, slope_fp: Path, shade_fp: Path) -> None:
    """
    Use RichDEM (fast) for slope and hill-shade derivations.
    """
    LOG.info("   ⤷ Deriving slope° and hill-shade")
    dem = rd.LoadGDAL(str(elev_fp))
    slope = rd.TerrainAttribute(dem, attrib="slope_degrees")
    hill = rd.Hillshade(dem, azimuth=315, altitude=45)

    for arr, fp, dtype, nodata in [
        (slope, slope_fp, np.uint16, 65535),
        (hill, shade_fp, np.uint8, 0),
    ]:
        _write_single_band(arr, elev_fp, fp, dtype, nodata)


def _write_single_band(data, ref_fp: Path, out_fp: Path, dtype, nodata):
    """Clone metadata from ref DEM, replace array + dtype."""
    with rio.open(ref_fp) as ref:
        meta = ref.meta.copy()
        meta.update({"dtype": dtype, "nodata": nodata})
        with rio.open(out_fp, "w", **meta) as dst:
            dst.write(data.astype(dtype), 1)
