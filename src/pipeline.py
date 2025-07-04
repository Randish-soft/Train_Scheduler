"""
pipeline.py – BCPC end-to-end for ONE scenario row (terrain-aware v0·8)

New since v0·5
--------------
✓  Optional columns: if `daily_commuters` or `terrain_ruggedness` is blank
   in the CSV they’re **imputed on-the-fly**:
       • commuters  ← gravity-style estimate_commuters(...)
       • ruggedness ← mean slope from the freshly downloaded DEM
✓  Global budget: the loader forward-fills `budget_total_eur`, so every row
   sees the same cap-ex pot.
✓  Thousands-separator in KPI log uses pre-formatting – no logger crash.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Sequence

import geopandas as gpd
import networkx as nx
import numpy as np
import shapely.geometry as geom
from shapely.geometry import Polygon, MultiPolygon

from . import OUTPUT_DIR, logger                    # configured in __init__.py
from .commuter_model import estimate_commuters
from .cost import CostBreakdown, estimate_cost
from .demand import estimate_demand
from .enrich import get_city_boundary
from .scenario_io import ScenarioRow, save_geojson
from .models import Gauge, TrackType, TrainType
from .optimise import NetworkDesign, optimise_design
from .routing import trace_route, RoutingError
from .terrain import load_dem, ruggedness_index

logging.getLogger("bcpc.pipeline").setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Demo catalogues (replace with YAML config later)                            #
# --------------------------------------------------------------------------- #
STD_TRACK = TrackType("Std-Cat-160", Gauge.STANDARD, True, 160, 1_200, 12_000_000)
EMU       = TrainType ("4-car EMU",   Gauge.STANDARD, 400, 160, 10_000_000, 8)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _nx_to_gdf(g: nx.Graph) -> gpd.GeoDataFrame:
    if g.number_of_edges() == 0:
        return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    crs = g.graph.get("crs", "EPSG:4326")
    rows = [
        dict(u=u, v=v,
             length_km=data["geometry"].length / 1_000,
             geometry=data["geometry"])
        for u, v, data in g.edges(data=True)
    ]
    return gpd.GeoDataFrame(rows, crs=crs)


def _select_extreme_points(geom_in: Polygon | MultiPolygon
                           ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if isinstance(geom_in, Polygon):
        polys = [geom_in]
    elif isinstance(geom_in, MultiPolygon):
        polys = list(geom_in.geoms)
    else:
        raise TypeError(f"Unexpected geometry {type(geom_in)}")

    coords: list[Tuple[float, float]] = []
    for p in polys:
        coords.extend(p.exterior.coords)
    coords = coords[:: max(1, len(coords)//500)]        # thin ring

    xy = np.array(coords)
    d2 = np.sum((xy[:, None, :] - xy[None, :, :])**2, axis=-1)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    return tuple(coords[i]), tuple(coords[j])

# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def run_pipeline(row: ScenarioRow,
                 all_rows: Sequence[ScenarioRow]) -> Tuple[NetworkDesign,
                                                            CostBreakdown,
                                                            Path]:
    logger.info("==> %s – starting run", row.city_name)

    # 1 ▸ Boundary + DEM --------------------------------------------------
    boundary = get_city_boundary(row.city_name, row.city_id)
    gdf_boundary = gpd.GeoSeries([boundary], crs=4326)

    dem = load_dem(gdf_boundary)               # raises on hard failure

    # 1b ▸ inject ruggedness if missing
    if row.terrain_ruggedness is None:
        row.terrain_ruggedness = ruggedness_index(dem)
        logger.debug("Imputed ruggedness %.2f for %s",
                     row.terrain_ruggedness, row.city_name)

    # 2 ▸ provisional termini --------------------------------------------
    origin, dest = _select_extreme_points(boundary)

    # 3 ▸ terrain-aware routing ------------------------------------------
    try:
        line = trace_route(origin, dest, gdf_boundary, dem, STD_TRACK)
    except RoutingError as exc:
        logger.warning("Routing failed: %s – fallback to straight chord", exc)
        line = geom.LineString([origin, dest])

    G = nx.Graph(crs="EPSG:4326")
    G.add_edge(line.coords[0], line.coords[-1],
               geometry=line, weight=line.length/1_000)

    # 4 ▸ commuters if missing -------------------------------------------
    if row.daily_commuters is None:
        row.daily_commuters = estimate_commuters(row, all_rows)
        logger.debug("Imputed %d commuters for %s",
                     row.daily_commuters, row.city_name)

    demand_ppd = estimate_demand(row.population,
                                 row.daily_commuters,
                                 row.tourism_index)

    design = optimise_design(demand_ppd, row.budget_total_eur)
    design.graph, design.track, design.train = G, STD_TRACK, EMU

    # 5 ▸ cost breakdown --------------------------------------------------
    km_track = sum(d["geometry"].length for _,_,d in G.edges(data=True))/1_000
    cost_br  = estimate_cost(km_track, len(G.nodes), 1, 5, underground_ratio=0)
    design.cost_eur = cost_br.total()

    # 6 ▸ export ----------------------------------------------------------
    if G.number_of_edges():
        gdf_tracks = _nx_to_gdf(G)
        artifact = OUTPUT_DIR / f"{row.city_id}_tracks.geojson"
        save_geojson(gdf_tracks, artifact)
    else:
        artifact = OUTPUT_DIR / f"{row.city_id}_EMPTY.txt"
        artifact.write_text("Network empty – budget insufficient.\n")

    # 7 ▸ KPI log ---------------------------------------------------------
    logger.info("%s ✓ CAPEX €%.1f M | %d stations | %s pax/day",
                row.city_name,
                design.cost_eur/1e6,
                len(design.graph.nodes),
                f"{design.ridership_daily:,.0f}")

    return design, cost_br, artifact
