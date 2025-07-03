"""
enrich.py – Add open data (OSM boundaries, terrain grids, etc.) to a scenario.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path

import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon

from . import DATA_DIR, logger

ox.settings.log_console = False
ox.settings.use_cache = True
_BOUNDARY_CACHE = DATA_DIR / "_boundaries"

def _cached_boundary_path(city_id: str) -> Path:
    return Path(_BOUNDARY_CACHE / f"{city_id}.gpkg")

@functools.lru_cache(maxsize=64)
def get_city_boundary(city_name: str, city_id: str) -> gpd.GeoSeries:
    """Download (or load cache) of administrative boundary from OSM."""
    _BOUNDARY_CACHE.mkdir(exist_ok=True, parents=True)
    cache_file = _cached_boundary_path(city_id)
    if cache_file.exists():
        try:
            gdf = gpd.read_file(cache_file)
            logger.debug("Boundary loaded from cache: %s", cache_file)
            return gdf.geometry.iloc[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache read failed (%s) – redownloading", exc)

    logger.info("Downloading OSM boundary for %s", city_name)
    try:
        gdf = ox.geocoder.geocode_to_gdf(city_name)
    except Exception as exc:  # noqa: BLE001
        logger.error("OSM geocode failed – %s", exc)
        raise
    gdf.to_file(cache_file, driver="GPKG")
    return gdf.geometry.iloc[0]


def clip_nimby(gdf: gpd.GeoDataFrame, nimby: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    """Subtract NIMBY polygons from candidate geometries."""
    if nimby is None or nimby.empty:
        return gdf
    gdf = gdf.overlay(nimby, how="difference")
    return gdf


def build_candidate_corridors(boundary: Polygon) -> gpd.GeoDataFrame:
    """Generate a simple hex grid within boundary as proxy candidate track segments."""
    logger.debug("Building candidate corridor grid")
    try:
        import geopandas.tools as gptools  # lazy import

        grid = gptools.hexbin(boundary, 5000)  # 5 km spacing
        grid["geometry"] = grid.geometry.boundary  # convert polys → lines
        grid = gpd.GeoDataFrame(grid, crs="EPSG:3857")
        return grid
    except Exception as exc:  # noqa: BLE001
        logger.error("Hex grid generation failed – %s", exc)
        raise
