"""
Learn/processing/choosing_train_for_route.py
===========================================

Purpose
-------
1.  Collapse the 250 m geometric *segments* (written by plotting_route.py)
    into **operational sections**:

        • speed_class  → low / medium / high
        • plus any segment whose mean slope ≥ 30 ‰ is tagged **mountain**

2.  For every **Line Name**:
        • derive section statistics (length, Vmax, limiting gradient)
        • estimate *peak-hour* passenger demand from the OD-matrix
          → pax_ph ≈ daily_pax × 0.15                             
3.  Choose the **smallest* rolling-stock model that satisfies  
        • max_speed  ≥ line Vmax  
        • max_grade  ≥ line gradient  
        • crush_cap  ≥ pax_ph  

Outputs (written beneath `pipeline/data/<country>/routes/`)
----------------------------------------------------------
sections.gpkg
    – layer **sections** with geometry + attrs  
train_assignment.parquet
    – one row per Line Name with the chosen train model  
train_meta.json
    – snapshot (roster size, #lines, etc.)

Roster data
-----------
Capacity & speed figures are hard-coded below and sourced from:

* **Alstom TGV M / Avelia Horizon** – 740 seats, 320 km/h  :contentReference[oaicite:0]{index=0}  
* **DB ICE 4 (12-car)** – 830 seats, 250 km/h             :contentReference[oaicite:1]{index=1}  
* **Bombardier IC2 / KISS-5** – 466 seats, 160 km/h        :contentReference[oaicite:2]{index=2}  
* **Stadler FLIRT-3 (4-car)** – 244 seats, 160 km/h        :contentReference[oaicite:3]{index=3}

Extend *ROSTER* or replace it with a DB call when you have a larger fleet
database available.

"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from tqdm import tqdm

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

# ---------------------------------------------------------------------------- #
# CONSTANTS & CONFIG
# ---------------------------------------------------------------------------- #
DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
PPD_TO_PHP = 0.15       # daily passengers → peak-hour factor
GRADE_MTN_PPM = 30      # ≥30 ‰ ⇒ mountain section
SECTION_SIMPLIFY_M = 5  # geometry simplification tolerance (metres)
SEGM_CRS = 3857         # metric CRS used by plotting_route

# ---------------------------------------------------------------------------- #
# Roster – expand or load from DB later
# ---------------------------------------------------------------------------- #
@dataclass(slots=True)
class Train:
    id: str
    model: str
    max_speed: int      # km/h
    seats: int
    crush_cap: int
    power: str
    max_grade_ppm: int

ROSTER: List[Train] = [
    Train("TGV_M", "Alstom Avelia Horizon (TGV M)", 320, 740, 900, "electric", 22),
    Train("ICE4_12", "Siemens/DB ICE 4 (12-car)",    250, 830, 1000, "electric", 30),
    Train("IC2_5", "Bombardier/KISS-5 IC2",          160, 466, 570,  "electric", 25),
    Train("FLIRT3_4", "Stadler FLIRT-3 (4-car)",     160, 244, 300,  "electric", 35),
]

ROSTER_DF = pd.DataFrame([t.__dict__ for t in ROSTER]).set_index("id")

# ---------------------------------------------------------------------------- #
# Artifacts dataclass
# ---------------------------------------------------------------------------- #
@dataclass(slots=True)
class Artifacts:
    sections: Path
    assignment: Path
    meta: Path

    def as_dict(self) -> Dict[str, Path]:
        return self.__dict__


# ---------------------------------------------------------------------------- #
# PUBLIC ENTRY
# ---------------------------------------------------------------------------- #
def choose(
    routes: Dict[str, Path],
    demand: Dict[str, Path],
    *,
    country: str,
    refresh: bool = False,
) -> Dict[str, Path]:
    """
    Main entry called by pipeline_initiator – returns dict of file paths.
    """
    slug = country.lower().replace(" ", "_")
    out_dir = DATA_ROOT / slug / "routes"
    out_dir.mkdir(parents=True, exist_ok=True)

    sec_fp  = out_dir / "sections.gpkg"
    asn_fp  = out_dir / "train_assignment.parquet"
    meta_fp = out_dir / "train_meta.json"

    if not refresh and sec_fp.exists() and asn_fp.exists():
        LOG.info("🚆  Using cached train assignment for %s", country)
        return Artifacts(sec_fp, asn_fp, meta_fp).as_dict()

    LOG.info("🚆  Choosing trains for %s …", country)

    # ---------------------------------------------------------------------- #
    # 1) Load inputs
    # ---------------------------------------------------------------------- #
    segs   = gpd.read_file(routes["segments"], layer="segments").to_crs(SEGM_CRS)  # type: ignore[index]
    od     = pd.read_parquet(demand["od"])                                         # type: ignore[index]
    zones  = gpd.read_file(demand["zones"], layer="zones").to_crs(SEGM_CRS)        # type: ignore[index]

    # ---------------------------------------------------------------------- #
    # 2) Collapse 250 m segments ➜ operational sections
    # ---------------------------------------------------------------------- #
    sections = _collapse_segments(segs)
    sections = sections.to_crs(SEGM_CRS)

    # ---------------------------------------------------------------------- #
    # 3) Estimate peak-hour demand per Line
    # ---------------------------------------------------------------------- #
    peak_df = _estimate_peak_pax(sections, od, zones)
    sections = sections.merge(peak_df, on="line_name")

    # ---------------------------------------------------------------------- #
    # 4) Pick rolling-stock for every Line
    # ---------------------------------------------------------------------- #
    assignments = []
    for ln, grp in sections.groupby("line_name"):
        vmax   = int(grp["vmax_kph"].max())
        grade  = int(grp["max_grade_ppm"].max())
        pax_ph = float(grp["peak_pax_ph"].iloc[0])

        train_id = _select_train(vmax, grade, pax_ph)
        row      = ROSTER_DF.loc[train_id].to_dict()
        row.update(line_name=ln, required_speed=vmax,
                   required_grade=grade, peak_pax_ph=pax_ph)
        assignments.append(row)

    asn_df = pd.DataFrame(assignments)
    asn_df.to_parquet(asn_fp, index=False)
    sections.to_file(sec_fp, layer="sections", driver="GPKG")

    meta_fp.write_text(json.dumps({
        "country": country,
        "lines": len(asn_df),
        "roster_size": len(ROSTER_DF),
    }, indent=2))

    LOG.info("🚆  Assigned stock to %d lines  →  %s", len(asn_df), asn_fp.name)
    return Artifacts(sec_fp, asn_fp, meta_fp).as_dict()


# ---------------------------------------------------------------------------- #
# HELPERS
# ---------------------------------------------------------------------------- #
def _collapse_segments(segs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge contiguous segments by (line_name, section_type)."""
    segs = segs.copy()
    segs["section_type"] = np.where(
        segs["slope_deg"] * 10 >= GRADE_MTN_PPM, "mountain", segs["speed_class"]
    )

    def _union(df):
        return df.unary_union.simplify(SECTION_SIMPLIFY_M)

    g = (
        segs.groupby(["line_name", "section_type"])
        .geometry.apply(_union)
        .reset_index()
        .rename(columns={"geometry": "geometry"})
    )
    gdf = gpd.GeoDataFrame(g, geometry="geometry", crs=segs.crs)
    gdf["length_km"]        = gdf.length / 1000
    gdf["vmax_kph"]         = gdf["section_type"].map({"low": 120, "medium": 200,
                                                       "high": 320, "mountain": 100})
    gdf["max_grade_ppm"]    = np.where(gdf["section_type"] == "mountain",
                                       GRADE_MTN_PPM, 25)
    return gdf


def _estimate_peak_pax(
    sections: gpd.GeoDataFrame, od: pd.DataFrame, zones: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Sum daily OD flows whose centroids fall within 3 km of any section."""
    zone_pts = zones.geometry.values
    zone_id  = zones["zone_id"].to_numpy()

    line_buffers = sections.groupby("line_name").geometry.apply(
        lambda g: g.unary_union.buffer(3000)
    )

    results = []
    for ln, buf in line_buffers.items():
        inside = zones.within(buf)
        z_ids  = zone_id[inside]
        flow   = od[
            od["origin_id"].isin(z_ids) & od["dest_id"].isin(z_ids)
        ]["demand_ppd"].sum()
        results.append({"line_name": ln, "peak_pax_ph": flow * PPD_TO_PHP})
    return pd.DataFrame(results)


def _select_train(vmax: int, grade: int, pax: float) -> str:
    """Return ID of first roster entry that satisfies all three constraints."""
    cand = ROSTER_DF[
        (ROSTER_DF["max_speed"]   >= vmax) &
        (ROSTER_DF["max_grade_ppm"] >= grade) &
        (ROSTER_DF["crush_cap"]   >= pax)
    ]
    if cand.empty:
        LOG.warning("⚠  No stock meets %dkm/h, %d‰, %d pax – using fastest big train",
                    vmax, grade, pax)
        return ROSTER_DF.sort_values(["max_speed", "crush_cap"], ascending=False).index[0]
    # choose the smallest (by crush_cap) that works
    return cand.sort_values("crush_cap").index[0]
