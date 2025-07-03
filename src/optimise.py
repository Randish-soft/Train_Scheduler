"""
optimise.py – Multi-objective rail routing & fleet-sizing optimiser (stub).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx

from .cost import estimate_cost
from .demand import DemandResult

logger = logging.getLogger("bcpc.optimise")


@dataclass
class NetworkDesign:
    graph: nx.Graph
    cost_eur: float
    ridership_daily: float


def simple_star_network(zones: int) -> nx.Graph:
    """Return star graph: node 0 hub, edges to all others (toy example)."""
    g = nx.Graph()
    g.add_nodes_from(range(zones))
    g.add_weighted_edges_from((0, i, 5) for i in range(1, zones))
    return g


def optimise_design(demand: DemandResult, budget_eur: int) -> NetworkDesign:
    """Toy optimiser – picks either star or do-nothing based on budget."""
    zones = len(demand.od)
    g = simple_star_network(zones)
    km_track = g.size(weight="weight")
    n_stations = zones
    # rule-of-thumb: 1 trainset per 4 km
    trainsets = int(max(1, km_track / 4))
    breakdown = estimate_cost(km_track, n_stations, n_yards=1, trainsets=trainsets)
    if breakdown.total() > budget_eur:
        logger.warning("Budget insufficient – returning empty network")
        g = nx.Graph()
        breakdown = estimate_cost(0, 0, 0, 0)
    ridership = demand.od.values.sum() / 2  # symmetric OD
    return NetworkDesign(graph=g, cost_eur=breakdown.total(), ridership_daily=ridership)
