"""
cost_surface.py – build a raster (numpy array) whose cell values represent
                  the *relative* construction cost of putting a new rail line
                  through that cell.

Returned objects
----------------
profile : dict      rasterio-style profile with transform + CRS
cost    : ndarray   float32 array (no-data = np.nan)
"""

from __future__ import annotations
import logging, tempfile, math

from pathlib import Path
from typing   import Tuple

import numpy            as np
import rasterio
from rasterio.enums     import Resampling

log = logging.getLogger("bcpc.costsurf")
class RoutingError(RuntimeError):
    """Raised when no feasible alignment can be found."""
# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _downsample_slope(dem_ds, factor: int = 3) -> Tuple[dict, np.ndarray]:
    """
    Read DEM, compute %-slope, downsample by `factor` using mean resampling.
    """
    # 1) slope (@30 m) -------------------------------------------------------
    arr  = dem_ds.read(1, masked=True).filled(np.nan)
    dy, dx = np.gradient(arr, dem_ds.res[1], dem_ds.res[0])
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))     # degrees

    # 2) reproject & downsample to coarser grid -----------------------------
    new_h = math.ceil(slope.shape[0] / factor)
    new_w = math.ceil(slope.shape[1] / factor)

    with rasterio.Env():        # in-memory VRT
        vrt_opts = {
            "driver": "VRT",
            "width" : new_w,
            "height": new_h,
            "transform": dem_ds.transform * dem_ds.transform.scale(factor, factor),
            "crs": dem_ds.crs,
        }
        with rasterio.io.MemoryFile() as mem:
            with mem.open(**vrt_opts) as dst:
                dst.write(
                    slope[np.newaxis, ...].astype("float32"),
                    indexes = 1
                )
            with mem.open() as ds:
                slope_ds = ds.read(
                    1,
                    out_shape=(new_h, new_w),
                    resampling=Resampling.average,
                    masked=True,
                )

    prof = dict(
        driver    = "GTiff",
        dtype     = "float32",
        count     = 1,
        height    = new_h,
        width     = new_w,
        crs       = dem_ds.crs,
        transform = dem_ds.transform * dem_ds.transform.scale(factor, factor),
        nodata    = np.nan,
    )
    return prof, slope_ds.filled(np.nan)


# --------------------------------------------------------------------------- #
# Public builder                                                              #
# --------------------------------------------------------------------------- #
def build_cost_surface(boundary_gdf, dem_ds):
    """
    Return (profile, cost_array) masked to boundary.

    Cost model (v0·1)
    -----------------
    cell_cost = 1 + (slope_deg / 3)²
      – flat ground ≈ 1
      – 6 % ≈ 5× cost
      – 10 % ≈ 12× cost
    """
    prof, slope = _downsample_slope(dem_ds, factor=3)

    # mask to study-area bbox (+1-cell padding)
    mask_geom   = boundary_gdf.unary_union
    mask_arr, _ = rasterio.mask.mask(
        dem_ds, [mask_geom], crop=True, nodata=np.nan, filled=False
    )

    cost = 1.0 + (slope / 3.0) ** 2          # simple quadratic penalty
    cost[np.isnan(mask_arr[0])] = np.nan     # outside area → blocked

    log.debug("Cost-surface built: %.1f %% cells valid",
              100 * np.isfinite(cost).mean())
    return prof, cost

def trace_route(origin, dest, gdf_boundary, dem, track):
    """
    Trace a route between two cities considering terrain and boundaries.
    
    Args:
        origin: Starting city/point
        dest: Destination city/point
        gdf_boundary: GeoDataFrame with boundary information
        dem: Digital Elevation Model data
        track: Track specifications (standard gauge, etc.)
    
    Returns:
        dict or LineString: Route information including path and metadata
    """
    # TODO: Implement actual routing algorithm
    # For now, return a simple placeholder
    
    # Extract coordinates if origin/dest are objects with geometry
    if hasattr(origin, 'geometry'):
        start_point = origin.geometry
    else:
        start_point = origin
        
    if hasattr(dest, 'geometry'):
        end_point = dest.geometry
    else:
        end_point = dest
    
    # Create a simple direct line for now
    from shapely.geometry import LineString
    
    # If points have coordinates
    if hasattr(start_point, 'x') and hasattr(end_point, 'x'):
        line = LineString([
            (start_point.x, start_point.y),
            (end_point.x, end_point.y)
        ])
    else:
        # Fallback to a dummy line
        line = LineString([(0, 0), (1, 1)])
    
    return line

__all__ = ["build_cost_surface"]
__all__ = ["trace_route", "RoutingError"]
