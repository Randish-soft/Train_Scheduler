"""
io.py – CSV validation, light NLP normalisation, and GeoJSON helpers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, List, Optional

import geopandas as gpd
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict
from text_unidecode import unidecode

from . import INPUT_DIR

logger = logging.getLogger("bcpc.io")

# ────────────────────────────────────────────────────────────────────────────
CSV_REQUIRED_COLS: Final[List[str]] = [
    # hard facts we *must* have
    "city_id",
    "city_name",
    "population",
    "tourism_index",
    # model-derived fields – may be blank at ingest
    "daily_commuters",
    "terrain_ruggedness",
    # global budget column (can be blank except first row)
    "budget_total_eur",
]

# ---------------------------------------------------------------------------
def _ascii_title(s: str) -> str:
    """ASCII + Title-case helper (“Zahlé ” → “Zahle”)."""
    return unidecode(s).strip().title()


class ScenarioRow(BaseModel):
    """
    One record of the scenario CSV after initial cleaning.
    Optional fields will be imputed later in the pipeline.
    """

    model_config = ConfigDict(validate_by_name=True)

    # raw vs normalised name
    city_name_raw: str = Field(alias="city_name")
    city_name: str

    city_id: str
    population: int = Field(..., gt=0)
    tourism_index: float = Field(..., ge=0, le=1)

    # may be blank in CSV
    daily_commuters: Optional[int] = Field(None, ge=0)
    terrain_ruggedness: Optional[float] = Field(None, ge=0, le=1)
    budget_total_eur: Optional[int] = Field(None, gt=0)   # forward-filled

    # ── validators ──────────────────────────────────────────────────────
    @field_validator("city_name", mode="before")
    @classmethod
    def _norm_name(cls, v: str) -> str:     # noqa: D401
        return _ascii_title(v)

    @field_validator("*", mode="before")
    @classmethod
    def _strip_strings(cls, v):
        return v.strip() if isinstance(v, str) else v


# ────────────────────────────────────────────────────────────────────────────
def load_scenario(csv_path: Path | str):
    """
    Parse the scenario CSV and return a list[ScenarioRow].

    Blank cells in `daily_commuters`, `terrain_ruggedness`, and
    `budget_total_eur` are accepted. The global budget is forward-filled.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    # Header check
    missing = [c for c in CSV_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}")

    # Forward-fill the shared budget column (first non-null value propagates)
    if df["budget_total_eur"].notna().any():
        df["budget_total_eur"].ffill(inplace=True)
    else:
        raise ValueError("budget_total_eur column is entirely blank")

    rows: list[ScenarioRow] = []
    for raw in df.to_dict(orient="records"):
        # Convert pandas NaN → None so Optional fields validate
        raw = {k: (None if pd.isna(v) else v) for k, v in raw.items()}
        try:
            rows.append(ScenarioRow(**raw))
        except ValidationError as err:
            logger.warning("Skipping invalid row: %s", err)

    if not rows:
        raise ValueError("No valid rows in CSV")

    logger.info("Loaded %d scenario rows", len(rows))
    return rows


# ────────────────────────────────────────────────────────────────────────────
def save_geojson(gdf: gpd.GeoDataFrame, out_path: Path | str):
    """Write a GeoJSON file and log the result."""
    out_path = Path(out_path)
    gdf.to_file(out_path, driver="GeoJSON")
    logger.info("GeoJSON written to %s (%d features)", out_path, len(gdf))
