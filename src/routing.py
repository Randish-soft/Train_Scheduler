"""
routing.py – v1·0
──────────────────
Creates *new* rail corridors by running a least-cost path (8-neighbour
Dijkstra) across a raster cost-surface.  Falls back to the previous
terrain-aware A* on road links when the raster stack cannot be built.
"""

from __future__ import annotations
import logging, math
from typing import Tuple, Sequence
from functools import lru_cache

import numpy as np, geopandas as gpd, shapely.geometry as geom, networkx as nx
from shapely.ops import substring
from pyproj import Geod

# external helpers in your repo
from src.models        import TrackType
from src.terrain       import slope_percent, load_dem          # unchanged
from src.cost_surface  import build_cost_surface               # ← new file we sketched
from src.cost_surface  import rd                              # richdem already imported there

log   = logging.getLogger("bcpc.routing")
_GEOD = Geod(ellps="WGS84")          # single WGS-84 helper

# ────────────────────────────────────────────────────────────────────────────
# Exceptions
# ────────────────────────────────────────────────────────────────────────────
class RoutingError(RuntimeError):
    """Raised when no feasible alignment can be found."""


# ────────────────────────────────────────────────────────────────────────────
# Raster LCP helper
# ────────────────────────────────────────────────────────────────────────────
def _least_cost_path(profile: dict, cost: np.ndarray,
                     origin: Tuple[float,float], dest: Tuple[float,float]
                     ) -> geom.LineString:
    """Return a LineString connecting lon-lat points through the cheapest pixels."""
    rd_cost = rd.rdarray(cost, no_data=np.nan)
    rd_cost.geotransform = (
        profile["transform"][2], profile["transform"][0], 0,
        profile["transform"][5], 0, profile["transform"][4])

    g = rd.CreateCostPathGraph(rd_cost)        # ≈0.3 s for Lebanon @100 m

    row_o, col_o = ~profile["transform"] * origin[::-1]   # lon,lat → row,col
    row_d, col_d = ~profile["transform"] * dest[::-1]

    path = g.Path(int(row_o), int(col_o), int(row_d), int(col_d))
    if not path:
        raise RoutingError("Raster LCP failed – no path")

    # pixel centres back to lon-lat
    coords = [profile["transform"] * (c+0.5, r+0.5) for r,c in path]
    lonlat = [(x, y) for y, x in coords]       # swap back
    return geom.LineString(lonlat)


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────
def trace_route(
        origin_lonlat : Tuple[float,float],
        dest_lonlat   : Tuple[float,float],
        boundary_gdf  : gpd.GeoSeries,
        dem,                                   # rasterio dataset (already open)
        track         : TrackType,
) -> geom.LineString:
    """
    1. Try green-field least-cost path across the cost-surface grid.
    2. If anything goes wrong (missing layers, out-of-memory…), fall back to
       the legacy road-graph A* (kept below, untouched).
    """
    # --------------------------------------------------------------- LCP first
    try:
        profile, cost_arr = build_cost_surface(boundary_gdf, dem)
        line = _least_cost_path(profile, cost_arr, origin_lonlat, dest_lonlat)

        # sanity: if absurdly indirect, refuse it so the caller can fall back
        gc = abs(_GEOD.line_length([origin_lonlat[0], dest_lonlat[0]],
                                   [origin_lonlat[1], dest_lonlat[1]]))
        if line.length > 4 * gc:
            raise RoutingError("LCP too long vs great-circle")

        log.info("Raster LCP succeeded (%.1f km)", line.length/1000)
        return line

    except Exception as exc:           # noqa: BLE001
        log.warning("Raster LCP failed – %s  ➜  trying road graph", exc)

    # ---------------------------------------------------------------- A* road
    from src.routing_legacy import trace_route_legacy   # <-- just move your old
    return trace_route_legacy(                          #      code into that
        origin_lonlat, dest_lonlat, boundary_gdf, dem, track)


