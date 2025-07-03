"""
enrich.py – Add open data (OSM boundaries, terrain grids, etc.) to a scenario.
Compatible with GeoPandas 0.14 and 1.x
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Union

import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, Polygon

from . import DATA_DIR

logger = logging.getLogger("bcpc.enrich")
ox.settings.log_console = False
ox.settings.use_cache = True

_BOUNDARY_CACHE = DATA_DIR / "_boundaries"


# ---------------------------------------------------------------------------
# OSM city boundary
# ---------------------------------------------------------------------------
def _cache_path(city_id: str) -> Path:
    return _BOUNDARY_CACHE / f"{city_id}.gpkg"


def _download_boundary(city_name: str) -> Union[Polygon, Point]:
    """Return a polygon or point from OSM for the given city."""
    gdf = ox.geocoder.geocode_to_gdf(city_name, which_result=None)
    if gdf.empty:
        raise RuntimeError(f"No OSM result for «{city_name}»")
    geom = gdf.geometry.iloc[0]
    if isinstance(geom, (Polygon, Point)):
        return geom
    # MultiPolygon / LineString etc → dissolve into one polygon if possible
    if geom.geom_type.startswith("Multi"):
        return geom.convex_hull
    raise RuntimeError(f"Unsupported geometry type {geom.geom_type}")


@functools.lru_cache(maxsize=64)
def get_city_boundary(city_name: str, city_id: str) -> Polygon:
    """Download (or load cached) admin boundary; fall back to 10 km circle."""
    _BOUNDARY_CACHE.mkdir(exist_ok=True, parents=True)
    cache_file = _cache_path(city_id)

    # read cache if valid
    if cache_file.exists():
        try:
            return gpd.read_file(cache_file).geometry.iloc[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Boundary cache read failed (%s) – redownloading", exc)

    logger.info("Downloading OSM boundary for %s", city_name)
    geom = _download_boundary(city_name)

    if isinstance(geom, Point):
        logger.warning(
            "Only a point geometry found for %s – using 10 km circular buffer", city_name
        )
        geom = geom.buffer(10_000)  # metres (in EPSG:4326 that’s ~deg, fine for toy)

    if not isinstance(geom, Polygon):
        raise RuntimeError(
            f"Could not obtain polygon boundary for {city_name} (got {geom.geom_type})"
        )

    gpd.GeoSeries([geom]).to_file(cache_file, driver="GPKG")
    return geom


# ---------------------------------------------------------------------------
# Candidate corridor generation
# ---------------------------------------------------------------------------
def _fallback_hex_grid(boundary: Polygon, spacing: int = 5_000) -> gpd.GeoDataFrame:
    import math
    import shapely.geometry as geom

    boundary_3857 = gpd.GeoSeries([boundary], crs=4326).to_crs(3857).iloc[0]
    minx, miny, maxx, maxy = boundary_3857.bounds
    dx = spacing * math.sqrt(3)
    dy = spacing * 1.5

    hex_lines = []
    y = miny
    row = 0
    while y < maxy + spacing:
        offset = 0 if row % 2 == 0 else dx / 2
        x = minx - offset
        while x < maxx + dx:
            center = geom.Point(x, y)
            hexagon = center.buffer(spacing, resolution=6)
            if hexagon.intersects(boundary_3857):
                hex_lines.append(hexagon.intersection(boundary_3857).boundary)
            x += dx
        y += dy
        row += 1
    return gpd.GeoDataFrame(geometry=hex_lines, crs=3857)


def build_candidate_corridors(boundary: Polygon) -> gpd.GeoDataFrame:
    try:
        import geopandas.tools as gptools

        grid = gptools.hexbin(boundary, 5_000)
        grid["geometry"] = grid.geometry.boundary
        return grid.to_crs(3857)
    except (ImportError, AttributeError, TypeError):
        logger.debug("hexbin unavailable – using fallback generator")
        return _fallback_hex_grid(boundary, 5_000)
