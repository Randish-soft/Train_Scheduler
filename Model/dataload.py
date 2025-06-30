import pathlib as pl
import pandas as pd
import json

ROOT   = pl.Path(__file__).resolve().parents[1]
INPUT  = ROOT / "input" / "master-data.csv"
OUTPUT = ROOT / "data"

SCHEMAS = {
    "Stations": [
        "station_id", "city", "name", "n_tracks",
        "size_m2", "amenities", "overhead_wires"
    ],
    "Tracks": [
        "segment_id", "city_a", "city_b", "distance_km",
        "terrain_class", "allowed_train_types"
    ],
    "Population-per-city": ["city", "population"],
    "Railyard-position":   ["yard_id", "city", "lat", "lon"]
    # …etc
}

def parse_master_csv(path: pl.Path = INPUT) -> dict[str, pd.DataFrame]:
    df   = pd.read_csv(path)
    data = {}

    # Stations.json
    cols = SCHEMAS["Stations"]
    data["Stations"] = (
        df[cols]
        .drop_duplicates("station_id")
        .sort_values("station_id")
        .reset_index(drop=True)
    )

    # Tracks.json
    data["Tracks"] = (
        df[SCHEMAS["Tracks"]]
        .drop_duplicates("segment_id")
        .sort_values("segment_id")
        .reset_index(drop=True)
    )

    # Population-per-city.json
    data["Population-per-city"] = (
        df[SCHEMAS["Population-per-city"]]
        .drop_duplicates("city")
        .reset_index(drop=True)
    )

    # Railyard-position.json  – option: infer one yard per city
    data["Railyard-position"] = (
        df[SCHEMAS["Railyard-position"]]
        .dropna(subset=["yard_id"])
        .drop_duplicates("yard_id")
        .reset_index(drop=True)
    )
    return data


def materialise_json(dfs: dict[str, pd.DataFrame], out_dir: pl.Path = OUTPUT):
    out_dir.mkdir(exist_ok=True, parents=True)
    for name, frame in dfs.items():
        frame.to_json(out_dir / f"{name}.json", orient="records", indent=2)
