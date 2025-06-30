import pathlib as pl
import pandas as pd
import json

ROOT   = pl.Path(__file__).resolve().parents[1]
INPUT  = ROOT / "input" / "master-data.csv"
OUTPUT = ROOT / "data"

SCHEMAS = {
    # ─────────── node-level files ───────────
    "Stations": [
        "station_id", "city", "name", "n_tracks",
        "size_m2", "amenities", "overhead_wires"
    ],
    "Population-per-city": ["city", "population"],
    "Railyard-position":   ["yard_id", "city", "lat", "lon"],

    # ─────────── edge-level files ───────────
    "Tracks": [
        "segment_id", "city_a", "city_b", "distance_km",
        "terrain_class", "allowed_train_types"
    ],
    "Frequency": [          # <-- NEW
        "segment_id", "peak_trains_per_hour", "offpeak_trains_per_hour"
    ],

    # ─────────── project-wide files ─────────
    "Budget": [             # <-- NEW
        "fiscal_year", "currency", "capex_million", "opex_million"
    ]
}


def parse_master_csv(path: pl.Path = INPUT) -> dict[str, pd.DataFrame]:
    """
    Split the monolithic master-data.csv into canonical DataFrames
    (one per JSON spec in SCHEMAS).
    """
    df   = pd.read_csv(path)
    data: dict[str, pd.DataFrame] = {}

    # ── node-level
    data["Stations"] = (
        df[SCHEMAS["Stations"]]
        .drop_duplicates("station_id")
        .sort_values("station_id")
        .reset_index(drop=True)
    )

    data["Population-per-city"] = (
        df[SCHEMAS["Population-per-city"]]
        .drop_duplicates("city")
        .reset_index(drop=True)
    )

    data["Railyard-position"] = (
        df[SCHEMAS["Railyard-position"]]
        .dropna(subset=["yard_id"])
        .drop_duplicates("yard_id")
        .reset_index(drop=True)
    )

    # ── edge-level
    data["Tracks"] = (
        df[SCHEMAS["Tracks"]]
        .drop_duplicates("segment_id")
        .sort_values("segment_id")
        .reset_index(drop=True)
    )

    data["Frequency"] = (
        df[SCHEMAS["Frequency"]]
        .drop_duplicates("segment_id")
        .sort_values("segment_id")
        .reset_index(drop=True)
    )

    # ── project-wide
    data["Budget"] = (
        df[SCHEMAS["Budget"]]
        .drop_duplicates("fiscal_year")
        .sort_values("fiscal_year")
        .reset_index(drop=True)
    )

    return data


def materialise_json(dfs: dict[str, pd.DataFrame], out_dir: pl.Path = OUTPUT) -> None:
    """
    Dump every DataFrame to pretty-printed JSON.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for name, frame in dfs.items():
        (out_dir / f"{name}.json").write_text(
            frame.to_json(orient="records", indent=2)
        )
