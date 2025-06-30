import os
import pathlib as pl
import pandas as pd
import json, warnings

# ───────────────────────────────────────────────────────────────
#  Paths
# ───────────────────────────────────────────────────────────────
ROOT   = pl.Path(__file__).resolve().parents[1]

# Allow override via env var:  export MASTER_CSV=/path/to/your.csv
INPUT  = pl.Path(os.getenv("MASTER_CSV", ROOT / "input" / "master-data.csv"))

COSTS  = ROOT / "input" / "cost_params.json"
OUTPUT = ROOT / "data"

# ───────────────────────────────────────────────────────────────
#  A. Parse CSV → canonical DataFrames
# ───────────────────────────────────────────────────────────────
def parse_master_csv(path: pl.Path = INPUT) -> dict[str, pd.DataFrame]:
    df   = pd.read_csv(path)
    data: dict[str, pd.DataFrame] = {}

    # ── node-level -------------------------------------------------------
    data["Stations"] = (
        df[["station_id", "city", "name", "n_tracks",
            "size_m2", "amenities", "overhead_wires"]]
        .dropna(subset=["station_id"])
        .drop_duplicates("station_id")
        .reset_index(drop=True)
    )

    data["Population-per-city"] = (
        df[["city", "population"]]
        .drop_duplicates("city")
        .reset_index(drop=True)
    )

    data["Railyard-position"] = (
        df[["yard_id", "city", "lat", "lon"]]
        .dropna(subset=["yard_id"])
        .drop_duplicates("yard_id")
        .reset_index(drop=True)
    )
    data["City-coords"] = (
        df[["city", "lat", "lon"]]
        .dropna(subset=["lat", "lon"])
        .drop_duplicates("city")
        .reset_index(drop=True)
    )

    # ── edge-level -------------------------------------------------------
    edge_rows = df[df["segment_id"].notna()]      # ← filter out budget-only row

    data["Tracks"] = (
        edge_rows[["segment_id", "city_a", "city_b", "distance_km",
                   "terrain_class", "allowed_train_types"]]
        .drop_duplicates("segment_id")
        .reset_index(drop=True)
    )

    data["Frequency"] = (
        edge_rows[["segment_id", "peak_trains_per_hour",
                   "offpeak_trains_per_hour"]]
        .drop_duplicates("segment_id")
        .reset_index(drop=True)
    )

    # ── project-wide -----------------------------------------------------
    data["Budget"] = (
        df[["fiscal_year", "currency", "capex_million", "opex_million"]]
        .drop_duplicates("fiscal_year")
        .reset_index(drop=True)
    )

    return data


def materialise_json(dfs: dict[str, pd.DataFrame], out_dir: pl.Path = OUTPUT):
    """Dump every DataFrame to pretty-printed JSON."""
    out_dir.mkdir(exist_ok=True, parents=True)
    for name, frame in dfs.items():
        (out_dir / f"{name}.json").write_text(
            frame.to_json(orient="records", indent=2)
        )

# ───────────────────────────────────────────────────────────────
#  B. Back-compat helper (graph.py & others)
# ───────────────────────────────────────────────────────────────
def read_json(name: str) -> pd.DataFrame:
    path = OUTPUT / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found – run parse_master_csv()/materialise_json() first."
        )
    return pd.read_json(path)

# ───────────────────────────────────────────────────────────────
#  C. Cost parameters (with on-disk override)
# ───────────────────────────────────────────────────────────────
DEFAULT_COSTS = {
    "track_cost_per_km": {"coastal": 7.5, "rolling": 9.0, "mountain": 18.0},
    "double_track_multiplier": 1.6,
    "high_speed_multiplier": 2.1,
    "train_cost": {"TER_4car": 15.0, "TGV_8car": 38.0},
    "annual_crew_cost_per_train": 0.45,
    "discount_rate": 0.05
}

def load_cost_params() -> dict:
    if COSTS.exists():
        with open(COSTS) as f:
            user_costs = json.load(f)
        # Python 3.9+ dict union-merge
        return DEFAULT_COSTS | user_costs
    warnings.warn("cost_params.json not found – using baked-in defaults")
    return DEFAULT_COSTS
