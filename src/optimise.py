"""
optimise.py – toy optimiser with gauge / track / train choices.

Still a stub: selects the cheapest option that fits the budget and returns a
1-km hub-and-spoke graph.  Now uses the correct TrackType signature:
(name, gauge, electrified, speed_max_kmh, min_radius_m, capex_per_km_eur)
"""

from __future__ import annotations

import logging
from typing import List

import networkx as nx

from .models import Gauge, TrackType, TrainType

logger = logging.getLogger("bcpc.optimise")


# ---------------------------------------------------------------------------
# Demo catalogues (now with min-radius) -------------------------------------
STD_TRACK = TrackType(
    name="Std-Catenary-160",
    gauge=Gauge.STANDARD,
    electrified=True,
    speed_max_kmh=160,
    min_radius_m=1_200,
    capex_per_km_eur=12_000_000,
)
METRIC_TRACK = TrackType(
    name="Metric-Diesel-100",
    gauge=Gauge.METRIC,
    electrified=False,
    speed_max_kmh=100,
    min_radius_m=800,
    capex_per_km_eur=7_000_000,
)

EMU = TrainType(
    name="4-car EMU",
    gauge=Gauge.STANDARD,
    capacity=400,
    top_speed_kmh=160,
    purchase_cost_eur=10_000_000,
    opex_per_km_eur=8,
)
DMU = TrainType(
    name="2-car DMU",
    gauge=Gauge.METRIC,
    capacity=140,
    top_speed_kmh=100,
    purchase_cost_eur=4_000_000,
    opex_per_km_eur=6,
)

TRACK_CHOICES: List[TrackType] = [STD_TRACK, METRIC_TRACK]
TRAIN_CHOICES: List[TrainType] = [EMU, DMU]


# ---------------------------------------------------------------------------
# Return container ----------------------------------------------------------
class NetworkDesign:
    def __init__(
        self,
        graph: nx.Graph,
        track: TrackType,
        train: TrainType,
        ridership_daily: float,
        cost_eur: float,
    ) -> None:
        self.graph = graph
        self.track = track
        self.train = train
        self.ridership_daily = ridership_daily
        self.cost_eur = cost_eur


# ---------------------------------------------------------------------------
# Toy optimiser -------------------------------------------------------------
def _star_graph_km() -> nx.Graph:
    """Return a 1-km star centred at (0,0) – placeholder geometry."""
    g = nx.Graph()
    g.add_edge((0, 0), (0.5, 0), weight=0.5)
    g.add_edge((0, 0), (-0.5, 0), weight=0.5)
    return g


def optimise_design(demand_ppd: float, budget_eur: float) -> NetworkDesign:
    """
    Pick the cheapest gauge/stock combo that fits the budget.
    Fallback: empty graph.
    """
    graph = _star_graph_km()
    track = STD_TRACK
    train = EMU

    for tck, trn in zip(TRACK_CHOICES, TRAIN_CHOICES, strict=False):
        capex = tck.capex_per_km_eur * graph.size()
        rolling = trn.purchase_cost_eur * 5  # 5 trainsets
        total = capex + rolling
        if total <= budget_eur:
            track, train, cost = tck, trn, total
            break
    else:
        logger.warning("Budget insufficient – returning empty network")
        return NetworkDesign(nx.Graph(), track, train, demand_ppd, 0.0)

    logger.info(
        "Selected %s + %s at €%.1f M", track.name, train.name, cost / 1e6
    )
    return NetworkDesign(graph, track, train, demand_ppd, cost)
