"""
cost.py – Parametric cost estimation for tracks, stations, rolling stock.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("bcpc.cost")


@dataclass
class CostBreakdown:
    track_eur: float
    station_eur: float
    yard_eur: float
    rolling_stock_eur: float

    def total(self) -> float:
        return sum(vars(self).values())


def estimate_cost(
        km_track: float,
        n_stations: int,
        n_yards: int,
        trainsets: int,
        underground_ratio: float = 0.15,
) -> CostBreakdown:
    """Very rough CAPEX curves – adjust coefficients as real data becomes available."""
    COST_PER_KM_AT_GRADE = 15_000_000
    UNDERGROUND_MULTIPLIER = 5.0
    COST_PER_STATION = 20_000_000
    COST_PER_YARD = 100_000_000
    COST_PER_TRAINSET = 12_000_000

    track_cost = km_track * COST_PER_KM_AT_GRADE * (
            (1 - underground_ratio) + UNDERGROUND_MULTIPLIER * underground_ratio
    )
    station_cost = n_stations * COST_PER_STATION
    yard_cost = n_yards * COST_PER_YARD
    fleet_cost = trainsets * COST_PER_TRAINSET

    breakdown = CostBreakdown(
        track_eur=track_cost,
        station_eur=station_cost,
        yard_eur=yard_cost,
        rolling_stock_eur=fleet_cost,
    )
    logger.debug("Cost breakdown: %s", breakdown)
    return breakdown
