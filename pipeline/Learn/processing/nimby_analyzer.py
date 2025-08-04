"""
Learn/processing/nimby_analyzer.py
==================================
Explain which segments might trigger NIMBY resistance and *why*.

Public API
----------
analyze(routes: dict, *, country: str, refresh=False) -> dict

Returns a new *nimby_segments.gpkg* with extra columns:

    noise_risk      (0-100)
    visual_risk     (0-100)
    disruption_risk (0-100)
    nimby_score     (0-100)
    nimby_reason    ("dense urban", "scenic valley", …)

These columns are later consumed by the optimiser to decide on tunnels,
sound-walls or alternative alignments. :contentReference[oaicite:5]{index=5}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.sample import sample_gen
from shapely.geometry import LineString
from tqdm import tqdm

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"

# Risk weights
W_NOISE = 0.4
W_VISUAL = 0.35
W_DISRUPT = 0.25

# Thresholds
POP_NEAR_THRESH = 5000   # persons within 250 m buffer → high noise
SLOPE_SCENIC_DEG = 10    # scenery / valley cut-through


@dataclass(slots=True)
class NimbyArtifacts:
    segments: Path
    meta: Path

    def as_dict(self) -> Dict[str, Path]:
        return self.__dict__


# --------------------------------------------------------------------------- #
def analyze(
    routes: Dict[str, Path],
    *,
    country: str,
    refresh: bool = False,
) -> Dict[str, Path]:
    slug = country.lower().replace(" ", "_")
    out_dir = DATA_ROOT / slug / "routes"
    seg_in = routes["segments"]  # input GPKG from plotting_route
    seg_out = out_dir / "nimby_segments.gpkg"
    meta_fp = out_dir / "nimby_meta.json"

    if (not refresh) and seg_out.exists():
        LOG.info("🙅  Using cached NIMBY analysis for %s", country)
        return NimbyArtifacts(seg_out, meta_fp).as_dict()

    segs = gpd.read_file(seg_in, layer="segments")
    segs = segs.to_crs(3857)  # metric

    # 1) Population within 250 m (reuse WorldPop raster via terrain grid)
    pop_dens = _sample_pop_density(segs, radius_m=250)
    segs["pop_near"] = pop_dens

    # 2) Scenic / visual impact heuristics
    segs["visual_risk"] = np.where(
        (segs["form"].isin(["bridge", "embankment"])) & (segs["slope_deg"] > SLOPE_SCENIC_DEG),
        100,
        30,
    )

    # 3) Noise risk (pop density + speed)
    segs["noise_risk"] = np.clip(segs["pop_near"] / POP_NEAR_THRESH * 100, 0, 100)
    segs["noise_risk"] *= segs["speed_kph"] / segs["speed_kph"].max()

    # 4) Construction disruption (ground vs tunnel)
    segs["disruption_risk"] = np.where(
        segs["form"] == "ground",
        np.clip(segs["pop_near"] / POP_NEAR_THRESH * 100, 0, 100),
        20,
    )

    # 5) Aggregate score & dominant reason
    segs["nimby_score"] = (
        W_NOISE * segs["noise_risk"] + W_VISUAL * segs["visual_risk"] + W_DISRUPT * segs["disruption_risk"]
    ).round(1)

    segs["nimby_reason"] = segs.apply(_explain_reason, axis=1)

    # Persist
    segs.to_file(seg_out, layer="segments", driver="GPKG")
    meta = {
        "country": country,
        "avg_nimby_score": float(segs["nimby_score"].mean()),
        "pct_high_risk": float((segs["nimby_score"] >= 70).mean()) * 100,
    }
    meta_fp.write_text(json.dumps(meta, indent=2))

    return NimbyArtifacts(seg_out, meta_fp).as_dict()


# --------------------------------------------------------------------------- #
# Helper fns
# --------------------------------------------------------------------------- #
def _sample_pop_density(segs: gpd.GeoDataFrame, radius_m: int = 250) -> np.ndarray:
    """
    Buffer each segment centre by `radius_m`, sample WorldPop raster and return
    persons in buffer.
    """
    # WorldPop path stored in demand folder
    country_slug = segs.iloc[0]["line_name"]  # any field to get slug? fall back later
    pop_tif = _worldpop_tif_from_seg(segs)
    with rio.open(pop_tif) as pop_ds:
        pop = []
        for geom in tqdm(segs.geometry, desc="PopSample"):
            mid: LineString = LineString(geom.coords).interpolate(0.5, normalized=True)
            for val in sample_gen(pop_ds, [mid.buffer(radius_m).centroid], indexes=1):
                pop.append(float(val[0] * (radius_m * radius_m * np.pi) / 1_000_000))
    return np.array(pop)


def _worldpop_tif_from_seg(segs: gpd.GeoDataFrame) -> Path:
    country = segs.iloc[0]["line_name"] if "line_name" in segs.columns else "country"
    slug = country.lower().replace(" ", "_")
    return DATA_ROOT / slug / "demand" / f"worldpop_{slug.upper()}.tif"


def _explain_reason(row) -> str:
    reasons = []
    if row["noise_risk"] > 60:
        reasons.append("dense urban")
    if row["visual_risk"] > 60:
        reasons.append("scenic impact")
    if row["disruption_risk"] > 60:
        reasons.append("street disruption")
    return ", ".join(reasons) or "low"
