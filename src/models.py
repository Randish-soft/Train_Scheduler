"""Core dataclasses: gauge, track, train, and station profile."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class Gauge(Enum):
    STANDARD = 1_435
    CAPE = 1_067
    METRIC = 1_000
    BROAD = 1_676


@dataclass(frozen=True)
class TrackType:
    name: str
    gauge: Gauge
    electrified: bool
    speed_max_kmh: int
    min_radius_m: int
    capex_per_km_eur: float

    def __post_init__(self):
        if self.speed_max_kmh <= 0:
            raise ValueError("speed_max_kmh must be positive")
        if self.min_radius_m < 300:
            raise ValueError("min_radius_m unrealistically small")


@dataclass(frozen=True)
class TrainType:
    name: str
    gauge: Gauge
    capacity: int
    top_speed_kmh: int
    purchase_cost_eur: float
    opex_per_km_eur: float

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        if self.purchase_cost_eur <= 0:
            raise ValueError("purchase_cost_eur must be > 0")


@dataclass
class StationProfile:
    name: str
    lat: float
    lon: float
    depth_m: int = 0
    tracks: int = 2
    platforms: List[int] = field(default_factory=lambda: [180])
