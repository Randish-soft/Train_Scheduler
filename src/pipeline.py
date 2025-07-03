"""
pipeline.py – Orchestrates BCPC end-to-end for one scenario row.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import networkx as nx

from . import OUTPUT_DIR, logger
from .cost import CostBreakdown
from .demand import estimate_demand
from .enrich import build_candidate_corridors, get_city_boundary
from .io import save_geojson
from .optimise import NetworkDesign, optimise_design

logging.getLogger("bcpc").setLevel(logging.INFO)


def _nx_to_gdf(g: nx.Graph) -> gpd.GeoDataFrame:
    """Convert networkx graph with edge weights=km to GeoDataFrame (straight lines)."""
    import shapely.geometry as geom
    rows = []
    for u, v, data in g.edges(data=True):
        rows.append(
            {
                "u": u,
                "v": v,
                "length_km": data.get("weight", 1.0),
                "geometry": geom.LineString([(u, 0), (v, 0)]),
            }
        )
    return gpd.GeoDataFrame(rows, crs="EPSG:3857")


def run_pipeline(row) -> tuple[NetworkDesign, CostBreakdown, Path]:
    logger.info("=== Running BCPC for %s ===", row.city_name)

    boundary = get_city_boundary(row.city_name, row.city_id)
    _ = build_candidate_corridors(boundary)  # placeholder – unused for toy optimiser

    demand = estimate_demand(row.population, row.daily_commuters, row.tourism_index)
    design = optimise_design(demand, row.budget_total_eur)

    # Export track GeoJSON (straight-line placeholder)
    gdf_tracks = _nx_to_gdf(design.graph)
    out_path = OUTPUT_DIR / f"{row.city_id}_tracks.geojson"
    save_geojson(gdf_tracks, out_path)

    logger.info(
        "%s design done: daily trips=%.1f, cost=€%.1f M",
        row.city_name,
        design.ridership_daily,
        design.cost_eur / 1e6,
        )
    return design, CostBreakdown, out_path
