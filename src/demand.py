"""
demand.py – 4-step travel-demand model (simplified, but pluggable).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("bcpc.demand")


@dataclass
class DemandResult:
    """Container for outputs (zone OD matrix and summary stats)."""

    od: pd.DataFrame
    avg_trip_length_km: float
    peak_factor: float


def synthetic_zones(population: int, granularity: int = 10_000) -> np.ndarray:
    """Split population into roughly equal zones. Return zone sizes."""
    zones = max(1, population // granularity)
    base = population // zones
    remainder = population % zones
    out = np.full(zones, base, dtype=int)
    out[:remainder] += 1
    return out


def build_gravity_matrix(zones: np.ndarray, beta: float = 0.1) -> pd.DataFrame:
    """Basic doubly-constrained gravity model with distance decay exp(-β·d)."""
    size = len(zones)
    rng = np.random.default_rng(42)
    # Random centroid coordinates in a unit square – placeholder
    xy = rng.random((size, 2))
    dists = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2) + np.eye(size)
    impedance = np.exp(-beta * dists)
    trips = np.outer(zones, zones) * impedance
    # Iterative proportional fitting to match marginals
    for _ in range(10):
        trips *= zones / trips.sum(axis=1, keepdims=True)
        trips *= zones / trips.sum(axis=0, keepdims=True)
    return pd.DataFrame(trips, index=range(size), columns=range(size))


def estimate_demand(population: int, daily_commuters: int, tourism_idx: float) -> DemandResult:
    zones = synthetic_zones(population)
    od = build_gravity_matrix(zones)
    avg_trip_len = 10.0  # km – placeholder until network distances known
    peak_factor = 1.6 * (1 + tourism_idx)
    return DemandResult(od=od, avg_trip_length_km=avg_trip_len, peak_factor=peak_factor)
