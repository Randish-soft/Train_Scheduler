"""
io.py – CSV validation and GeoJSON helpers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, List

import geopandas as gpd
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, root_validator

from . import INPUT_DIR

logger = logging.getLogger("bcpc.io")

# ────────────────────────────────────────────────────────────────────────────
CSV_REQUIRED_COLS: Final[List[str]] = [
    "city_id",
    "city_name",
    "population",
    "tourism_index",
    "daily_commuters",
    "terrain_ruggedness",
    "budget_total_eur",
]


class ScenarioRow(BaseModel):
    city_id: str
    city_name: str
    population: int = Field(..., gt=0)
    tourism_index: float = Field(..., ge=0, le=1)
    daily_commuters: int = Field(..., ge=0)
    terrain_ruggedness: float = Field(..., ge=0, le=1)
    budget_total_eur: int = Field(..., gt=0)

    @root_validator(pre=True)
    def _strip_ws(cls, v):  # pylint: disable=no-self-argument
        return {k: val.strip() if isinstance(val, str) else val for k, val in v.items()}


# ────────────────────────────────────────────────────────────────────────────
def load_scenario(csv_path: Path | str):
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    missing = [c for c in CSV_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}")

    rows = []
    for raw in df.to_dict(orient="records"):
        try:
            rows.append(ScenarioRow(**raw))
        except ValidationError as err:
            logger.warning("Skipping invalid row: %s", err)
    if not rows:
        raise ValueError("No valid rows")
    logger.info("Loaded %d scenario rows", len(rows))
    return rows


# ────────────────────────────────────────────────────────────────────────────
def save_geojson(gdf: gpd.GeoDataFrame, out_path: Path | str):
    """
    Write a GeoJSON file and log the result.

    Raises any write error upward; caller handles.
    """
    out_path = Path(out_path)
    gdf.to_file(out_path, driver="GeoJSON")
    logger.info("GeoJSON written to %s (%d features)", out_path, len(gdf))
