"""
Learn/data_injesting/processing_demand.py
----------------------------------------

Step #3 of the Learn pipeline – estimate inter-zonal passenger demand.

How it works
~~~~~~~~~~~~
1.  Download **WorldPop 2020** population raster for the target country
    (≈100 m pixels; automatically mosaicked if multiple tiles).
2.  Dissolve your `admin.gpkg` polygons into *zones* (default: Admin-2 level).
3.  For each zone:
      • sum its population (raster zonal stats)  
      • store the centroid geometry  
4.  Build an **O-D matrix** with a gravity model::

       Tij = G · Pi · Pj / dij^β

    where *dij* is great-circle distance (km), *β=1.6* by default and *G* is
    a scaling constant so that the country-wide sum matches a plausible
    per-capita trip rate (e.g. 300 trips/year → 0.82 trips/day).

5.  Write three artefacts under
   `pipeline/data/<country-slug>/demand/`:

   ├── zones.gpkg        (centroids + pop)  
   ├── od.parquet        (origin_id, dest_id, demand_ppd)  
   └── meta.json

You can later recalibrate with survey / ticket data simply by replacing
`od.parquet` or overriding β and trip_rate.
---------------------------------------------------------------------------
Dependencies (pip):
    geopandas, rasterio, rasterstats, pandas, numpy, scipy, tqdm, requests
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import requests
from rasterstats import zonal_stats
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import Point
from tqdm import tqdm

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
NE_ADMIN_GPKG = "admin.gpkg"      # written by processing_input
WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/"
    "{iso3}/ppp/{iso3_lower}_ppp_2020.tif"
)

# --------------------------------------------------------------------------- #
# Dataclass wrapper
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class DemandArtifacts:
    zones: Path        # GeoPackage with centroids + population
    od: Path           # Parquet OD matrix (long format)
    meta: Path         # JSON

    def as_dict(self) -> Dict[str, Path]:
        return self.__dict__


# --------------------------------------------------------------------------- #
# Public entry
# --------------------------------------------------------------------------- #
def estimate_demand(
    raw_inputs: Dict[str, Path | None],
    *,
    country: str,
    refresh: bool = False,
    data_root: Path | str | None = None,
    beta: float = 1.6,
    trip_rate_pppy: int = 300,
) -> Dict[str, Path]:
    """
    Build an OD-demand layer and return file paths.

    Parameters
    ----------
    raw_inputs : dict
        Output of processing_input.py (must include 'admin' GPKG path).
    country : str
        Plain-language country name (“Belgium”, …).
    refresh : bool, default False
        Force rebuild even if cached files already exist.
    beta : float
        Gravity-model exponent (default 1.6 ≈ inter-city elasticity).
    trip_rate_pppy : int
        Target *trips per person per year* to scale the matrix.
    """
    data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    slug = country.lower().replace(" ", "_")
    out_dir = data_root / slug / "demand"
    out_dir.mkdir(parents=True, exist_ok=True)

    zones_fp = out_dir / "zones.gpkg"
    od_fp = out_dir / "od.parquet"
    meta_fp = out_dir / "meta.json"

    if (not refresh) and zones_fp.exists() and od_fp.exists():
        LOG.info("👥  Using cached demand layer for %s", country)
        return DemandArtifacts(zones_fp, od_fp, meta_fp).as_dict()

    LOG.info("👥  Building demand layer for %s …", country)

    # ------------------------------------------------------------------- #
    # 1) Load admin polygons & pick the finest available layer
    # ------------------------------------------------------------------- #
    adm_gpkg = Path(raw_inputs["admin"])  # type: ignore[arg-type]
    layer = _pick_finest_layer(adm_gpkg)
    gdf = gpd.read_file(adm_gpkg, layer=layer).to_crs(4326)

    # ------------------------------------------------------------------- #
    # 2) Download WorldPop 2020 raster for the country
    # ------------------------------------------------------------------- #
    iso3 = gdf.iloc[0]["adm0_a3"]  # Natural-Earth column
    pop_raster = _download_worldpop(iso3, out_dir)

    # ------------------------------------------------------------------- #
    # 3) Zonal statistics → population per zone
    # ------------------------------------------------------------------- #
    LOG.info("   ⤷ Zonal stats (population)…")
    stats = zonal_stats(
        gdf.geometry,
        pop_raster,
        stats=["sum"],
        raster_out=False,
        all_touched=True,
        nodata=-99999,
        progress=False,
    )
    gdf["population"] = [s["sum"] or 0 for s in stats]
    gdf["population"] = gdf["population"].astype(int)

    # Remove empty zones (e.g. uninhabited islands)
    gdf = gdf[gdf["population"] > 0].reset_index(drop=True)

    # Centroids for distance matrix
    gdf["centroid"] = gdf.geometry.centroid
    gdf["x"] = gdf.centroid.x
    gdf["y"] = gdf.centroid.y

    # IDs
    gdf["zone_id"] = np.arange(len(gdf), dtype=int)

    # ------------------------------------------------------------------- #
    # 4) Build OD matrix with gravity model
    # ------------------------------------------------------------------- #
    LOG.info("   ⤷ Gravity model (β=%.2f)…", beta)
    coords = gdf[["x", "y"]].to_numpy()
    dist_km = _haversine_matrix(coords)  # (N, N)
    pops = gdf["population"].to_numpy()
    prod = np.outer(pops, pops)
    Tij = prod / np.power(dist_km, beta, where=dist_km > 0)
    np.fill_diagonal(Tij, 0)  # no intra-zone

    # Scaling constant so that ΣTij matches pop * trip_rate
    trips_total = pops.sum() * (trip_rate_pppy / 365.0)
    G = trips_total / Tij.sum()
    Tij *= G

    # Long format → parquet
    LOG.info("   ⤷ Writing OD matrix …")
    origin, dest = np.nonzero(Tij)
    flow = Tij[origin, dest].astype(np.float32)
    od_df = pd.DataFrame(
        {"origin_id": origin.astype(np.int32), "dest_id": dest.astype(np.int32), "demand_ppd": flow}
    )
    od_df.to_parquet(od_fp, index=False)

    # Write zones file (one layer with centroids)
    LOG.info("   ⤷ Writing zones file …")
    zones = gpd.GeoDataFrame(
        gdf[["zone_id", "population", "centroid"]], geometry="centroid", crs=4326
    )
    zones.to_file(zones_fp, layer="zones", driver="GPKG")

    # Snapshot metadata
    meta = {
        "country": country,
        "beta": beta,
        "trip_rate_pppy": trip_rate_pppy,
        "zones": len(zones),
        "total_population": int(pops.sum()),
        "total_trips_ppd": float(flow.sum()),
    }
    meta_fp.write_text(json.dumps(meta, indent=2))

    return DemandArtifacts(zones_fp, od_fp, meta_fp).as_dict()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pick_finest_layer(gpkg: Path) -> str:
    """Return the Natural-Earth layer with the finest admin level present."""
    import fiona

    layers = fiona.listlayers(gpkg)
    for cand in ("admin", "ne_10m_admin_2_counties", "ne_10m_admin_1_states_provinces"):
        if cand in layers:
            return cand
    raise FileNotFoundError("No suitable admin layer inside admin.gpkg")


def _download_worldpop(iso3: str, out_dir: Path) -> Path:
    url = WORLDPOP_URL.format(iso3=iso3, iso3_lower=iso3.lower())
    tif = out_dir / f"worldpop_{iso3}.tif"
    if tif.exists():
        LOG.info("   ⤷ Using cached WorldPop raster")
        return tif
    LOG.info("   ⤷ Downloading WorldPop raster …")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tif.open("wb") as fh, tqdm(
            total=int(r.headers.get("content-length", 0)),
            unit="B",
            unit_scale=True,
            desc=tif.name,
        ) as bar:
            for chunk in r.iter_content(1024 * 1024):
                fh.write(chunk)
                bar.update(len(chunk))
    return tif


def _haversine_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Return an NxN matrix of great-circle distances (km) given Nx2 lon/lat deg.
    """
    lon, lat = np.deg2rad(coords[:, 0]), np.deg2rad(coords[:, 1])
    pts = np.column_stack([lon, lat])
    # pairwise haversine via scipy
    d = pdist(pts, _haversine_pair)
    return squareform(d, force="no", checks=False)


def _haversine_pair(p1: np.ndarray, p2: np.ndarray) -> float:
    r = 6371.0
    dl = p2[1] - p1[1]
    dp = p2[0] - p1[0]
    a = math.sin(dl / 2) ** 2 + math.cos(p1[1]) * math.cos(p2[1]) * math.sin(dp / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
