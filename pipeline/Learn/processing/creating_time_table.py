"""
Learn/processing/creating_time_table.py
---------------------------------------

Builds a static timetable straight from the GTFS feed that
`processing_input.py` downloaded (routes.txt, trips.txt, stop_times.txt,
calendar*.txt).  We:

    1. Pick the first *weekday* service_id that operates in the feed
       (calendar.txt monday==1).                                  ─ GTFS ref :contentReference[oaicite:0]{index=0}
    2. Filter `trips.txt` to those service_ids.
    3. For every **route_id**:
          • collect all its trips for that service;
          • grab the departure at the first stop and the arrival at the
            last stop from `stop_times.txt`;
          • write them to a Parquet `<route_id>.parquet` with:
                trip_id, depart_time_str, arrive_time_str, headsign.
    4. Produce an `index.parquet` so other pipeline steps know where each
       timetable file lives, plus a small `meta.json`.

No headway maths, no runtime estimation—just whatever the operator
published.  (If your GTFS folder is empty we raise a helpful error.)

Dependencies
------------
pandas ≥2.1, pyarrow, tqdm  (all pure-Python)

Typical output tree
-------------------
pipeline/data/belgium/timetable/
    ├── 1A.parquet          # one per route_id
    ├── 94.parquet
    ├── index.parquet
    └── meta.json
"""
from __future__ import annotations

import json
import logging
from datetime import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


# --------------------------------------------------------------------------- #
def build(
    raw_inputs: Dict[str, Path | None],
    *,
    country: str,
    refresh: bool = False,
) -> Dict[str, Path]:
    """
    Parameters
    ----------
    raw_inputs : dict
        The dict returned by `processing_input.load_inputs()`. Must contain
        key ``'gtfs'`` pointing to the unzipped feed directory.
    """
    gtfs_dir: Path | None = raw_inputs.get("gtfs")  # type: ignore[attr-defined]
    if gtfs_dir is None or not gtfs_dir.exists():
        raise FileNotFoundError(
            "No GTFS feed present. Run processing_input with --refresh "
            "or choose a reference country that publishes GTFS."
        )

    slug = country.lower().replace(" ", "_")
    out_dir = DATA_ROOT / slug / "timetable"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_fp = out_dir / "index.parquet"
    meta_fp = out_dir / "meta.json"

    if not refresh and idx_fp.exists():
        LOG.info("🕑  Using cached GTFS timetables for %s", country)
        return {"index": idx_fp, "meta": meta_fp, "folder": out_dir}

    # 1 ────────── load GTFS core text files
    routes = pd.read_csv(gtfs_dir / "routes.txt", dtype=str)
    trips = pd.read_csv(gtfs_dir / "trips.txt", dtype=str)
    st_times = pd.read_csv(gtfs_dir / "stop_times.txt", dtype=str)
    calendar = pd.read_csv(gtfs_dir / "calendar.txt", dtype=str)

    # 2 ────────── pick a weekday service_id (Monday == 1)
    weekday = calendar[calendar["monday"] == "1"].copy()
    if weekday.empty:
        raise ValueError("GTFS has no weekday service in calendar.txt")
    weekday_ids = set(weekday["service_id"].unique())
    trips = trips[trips["service_id"].isin(weekday_ids)]

    # 3 ────────── derive timetable per route
    master_rows: List[dict] = []
    route_ids = trips["route_id"].unique()
    for rid in tqdm(route_ids, desc="Building timetables"):
        t = trips[trips["route_id"] == rid]
        merged = (
            t[["trip_id", "trip_headsign"]]
            .merge(st_times, on="trip_id", how="left")
            .sort_values(["trip_id", "stop_sequence"])
        )

        # first/last stop per trip → depart/arrive
        first = merged.groupby("trip_id").first().reset_index()
        last = merged.groupby("trip_id").last().reset_index()
        sched = first[["trip_id", "trip_headsign"]].copy()
        sched["depart_time"] = first["departure_time"]
        sched["arrive_time"] = last["arrival_time"]

        # write parquet
        route_fp = out_dir / f"{rid}.parquet"
        sched.to_parquet(route_fp, index=False)

        master_rows.append(
            {
                "route_id": rid,
                "file": str(route_fp),
                "trips": len(sched),
                "headsigns": ", ".join(sched["trip_headsign"].unique()[:3]),
            }
        )

    # 4 ────────── index + meta
    idx = pd.DataFrame(master_rows)
    idx.to_parquet(idx_fp, index=False)

    meta = {
        "country": country,
        "routes": len(idx),
        "source_gtfs": str(gtfs_dir),
    }
    meta_fp.write_text(json.dumps(meta, indent=2))

    LOG.info("🕑  Wrote %d route timetables (weekday) → %s", len(idx), idx_fp.name)
    return {"index": idx_fp, "meta": meta_fp, "folder": out_dir}
