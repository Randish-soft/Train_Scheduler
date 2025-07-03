"""
pipeline.py – BCPC end-to-end for a single scenario row (terrain-aware v0·3)

Changes vs v0·2
---------------
✓ MultiPolygon boundaries supported
✓ Logging format strings fixed
✓ _nx_to_gdf uses real geometry CRS
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import networkx as nx
import shapely.geometry as geom
from shapely.geometry import Polygon, MultiPolygon

from . import OUTPUT_DIR, logger
from .cost import CostBreakdown
from .demand import estimate_demand
from .enrich import get_city_boundary
from .io import save_geojson
from .models import Gauge, TrackType, TrainType
from .optimise import NetworkDesign, optimise_design
from .routing import trace_route, RoutingError
from .terrain import load_dem

logging.getLogger("bcpc").setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Demo catalogues                                                             #
# --------------------------------------------------------------------------- #
STD_TRACK = TrackType("Std-Cat-160", Gauge.STANDARD, True, 160, 1_200, 12_000_000)
EMU       = TrainType("4-car EMU",   Gauge.STANDARD, 400, 160, 10_000_000, 8)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _nx_to_gdf(g: nx.Graph) -> gpd.GeoDataFrame:
    if g.number_of_edges() == 0:
        return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    rows = [
        {
            "u": u,
            "v": v,
            "length_km": d["geometry"].length / 1_000,
            "geometry": d["geometry"],
        }
        for u, v, d in g.edges(data=True)
    ]
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def _select_extreme_points(
    geom_in: Polygon | MultiPolygon,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return two boundary coords with max great-circle separation."""
    if isinstance(geom_in, Polygon):
        polys = [geom_in]
    elif isinstance(geom_in, MultiPolygon):
        polys = list(geom_in.geoms)
    else:
        raise TypeError(f"Unexpected boundary type: {type(geom_in)}")

    coords: list[Tuple[float, float]] = []
    for p in polys:
        coords.extend(p.exterior.coords)

    if len(coords) < 2:
        raise ValueError("Boundary has <2 coords")

    coords = coords[:: max(1, len(coords) // 300)]  # decimate long rings
    max_d, pair = 0.0, (coords[0], coords[1])
    for i, a in enumerate(coords):
        for b in coords[i + 1 :]:
            d = geom.Point(a).distance(geom.Point(b))
            if d > max_d:
                max_d, pair = d, (a, b)
    return pair


# --------------------------------------------------------------------------- #
# Main entry-point                                                            #
# --------------------------------------------------------------------------- #
def run_pipeline(row) -> Tuple[NetworkDesign, CostBreakdown, Path]:
    logger.info("=== Running BCPC for %s ===", row.city_name)

    # 1 ▸ boundary & DEM ----------------------------------------------------
    boundary = get_city_boundary(row.city_name, row.city_id)
    gdf_boundary = gpd.GeoSeries([boundary], crs=4326)
    try:
        dem = load_dem(gdf_boundary)
    except Exception as exc:  # noqa: BLE001
        logger.error("DEM unavailable – aborting %s : %s", row.city_name, exc)
        out = OUTPUT_DIR / f"{row.city_id}_ERROR.txt"
        out.write_text(f"DEM load failed – {exc}\n")
        return NetworkDesign(nx.Graph(), STD_TRACK, EMU, 0, 0), CostBreakdown, out

    # 2 ▸ provisional termini ----------------------------------------------
    origin, dest = _select_extreme_points(boundary)

    # 3 ▸ terrain-aware routing --------------------------------------------
    try:
        line = trace_route(origin, dest, gdf_boundary, dem, STD_TRACK)
    except RoutingError as exc:
        logger.warning("Routing failed: %s – using straight chord", exc)
        line = geom.LineString([origin, dest])

    G = nx.Graph()
    G.add_edge(line.coords[0], line.coords[-1], geometry=line, weight=line.length / 1_000)

    # 4 ▸ demand + optimisation --------------------------------------------
    demand = estimate_demand(row.population, row.daily_commuters, row.tourism_index)
    design = optimise_design(demand, row.budget_total_eur)
    design.graph, design.track, design.train = G, STD_TRACK, EMU

    # 5 ▸ export ------------------------------------------------------------
    if design.graph.number_of_edges() == 0:
        artifact = OUTPUT_DIR / f"{row.city_id}_EMPTY.txt"
        artifact.write_text("Network empty – budget insufficient.\n")
    else:
        try:
            gdf_tracks = _nx_to_gdf(design.graph)
            artifact = OUTPUT_DIR / f"{row.city_id}_tracks.geojson"
            save_geojson(gdf_tracks, artifact)
        except Exception as exc:  # noqa: BLE001
            logger.error("GeoJSON export failed: %s", exc, exc_info=True)
            artifact = OUTPUT_DIR / f"{row.city_id}_ERROR.txt"
            artifact.write_text(f"GeoJSON export failed – {exc}\n")

    # 6 ▸ KPIs --------------------------------------------------------------
    logger.info(
        "%s design done: cost=€%.1f M",
        row.city_name,
        design.cost_eur / 1e6,
    )
    return design, CostBreakdown, artifact
