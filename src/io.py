"""
io.py – CSV and auxiliary-data ingestion / validation utilities.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import geopandas as gpd
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, root_validator

from . import INPUT_DIR, logger

CSV_REQUIRED_COLS: Final[list[str]] = [
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
    def _strip_ws(cls, values):  # pylint: disable=no-self-argument
        return {k: v.strip() if isinstance(v, str) else v for k, v in values.items()}


def load_scenario(csv_path: str | Path) -> list[ScenarioRow]:
    """Load and validate a scenario CSV. Return list of ScenarioRow objects."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        logger.error("CSV not found: %s", csv_path)
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    missing = [c for c in CSV_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    scenarios: list[ScenarioRow] = []
    for idx, raw in df.iterrows():
        try:
            scenarios.append(ScenarioRow(**raw.to_dict()))
        except ValidationError as err:
            logger.warning("Row %s invalid – %s", idx, err)
            continue
    if not scenarios:
        raise ValueError("No valid rows in scenario CSV")
    logger.info("Loaded %d scenario rows", len(scenarios))
    return scenarios


def load_nimby_polygons(json_path: str | Path | None) -> gpd.GeoDataFrame | None:
    """Load NIMBY polygons from GeoJSON."""
    if json_path is None:
        return None
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    try:
        gdf = gpd.read_file(json_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read NIMBY polygons – %s", exc)
        raise
    if gdf.empty:
        logger.warning("NIMBY layer empty")
        return None
    return gdf


def save_geojson(gdf: gpd.GeoDataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    try:
        gdf.to_file(out_path, driver="GeoJSON")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to write GeoJSON: %s – %s", out_path, exc)
        raise
    logger.info("GeoJSON written to %s (%,d features)", out_path, len(gdf))
