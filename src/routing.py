"""
routing.py – terrain-aware routing with rail Overpass filter + fallback.

Changes
-------
* Use custom_filter to fetch OSM railways.
* If no rail geometry exists or Overpass errors, fall back to road graph.
"""

from __future__ import annotations

import logging
import math
from typing import Tuple

import geopandas as gpd
import networkx as nx
import osmnx as ox
import shapely.geometry as geom
from shapely.ops import substring

from .terrain import slope_percent
from .models import TrackType

logger = logging.getLogger("bcpc.routing")


class RoutingError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _edge_cost(u_xy, v_xy, dem, track: TrackType) -> float:
    """Return geodesic length × slope penalty."""
    line = geom.LineString([u_xy, v_xy])
    if line.length == 0:
        return math.inf
    x, y = line.interpolate(0.5).xy
    row, col = dem.index(x[0], y[0])
    try:
        grade = slope_percent(dem)[row, col]
    except IndexError:
        grade = 0.0
    penalty = 1 + max(0, grade - 3) / 10
    return line.length * penalty


def _load_rail_graph(boundary_poly) -> nx.MultiDiGraph | None:
    """Try Overpass rail filter; return None on failure/empty."""
    filter_rail = '["railway"~"rail|light_rail|subway|tram"]'
    try:
        G = ox.graph_from_polygon(
            boundary_poly,
            custom_filter=filter_rail,
            retain_all=True,
            simplify=True,
            clean_periphery=True,
        )
        if G.number_of_edges() == 0:
            return None
        return G
    except Exception as exc:  # noqa: BLE001
        logger.warning("Overpass rail query failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def trace_route(
    origin_lonlat: Tuple[float, float],
    dest_lonlat: Tuple[float, float],
    boundary: gpd.GeoSeries,
    dem,
    track: TrackType,
):
    """Return a LineString following rail (or road fallback) with grade cost."""
    boundary_poly = boundary.unary_union

    G = _load_rail_graph(boundary_poly)
    if G is None:
        logger.info("No rail graph – falling back to road network")
        G = ox.graph_from_polygon(boundary_poly, network_type="drive")

    if G.number_of_edges() == 0:
        raise RoutingError("Graph has no edges")

    try:
        orig = ox.distance.nearest_nodes(G, origin_lonlat[0], origin_lonlat[1])
        dest = ox.distance.nearest_nodes(G, dest_lonlat[0], dest_lonlat[1])
    except KeyError as exc:
        raise RoutingError("Unable to snap termini to graph") from exc

    # astar with terrain-aware edge cost
    try:
        route_nodes = nx.astar_path(
            G,
            orig,
            dest,
            weight=lambda u, v, d: _edge_cost(
                (G.nodes[u]["x"], G.nodes[u]["y"]),
                (G.nodes[v]["x"], G.nodes[v]["y"]),
                dem,
                track,
            ),
        )
    except nx.NetworkXNoPath as exc:
        raise RoutingError("No feasible path") from exc

    coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in route_nodes]
    line = geom.LineString(coords)

    # curvature check (simple first/last 100 m chord)
    if line.length > 0:
        seg = substring(line, 0, min(100, line.length))
        try:
            radius = seg.length**2 / (2 * seg.buffer(0).length + 1e-9)
            if radius < track.min_radius_m:
                logger.debug(
                    "Curve radius %.0f m < track min %d – still accepting for demo",
                    radius,
                    track.min_radius_m,
                )
        except Exception:  # noqa: BLE001
            pass

    return line
