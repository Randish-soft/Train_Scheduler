"""
Learn/processing_input.py
-------------------------

Step #1 of the Learn pipeline: fetch & normalise every raw dataset we’ll
need for a reference country.

Public API
~~~~~~~~~~
load_inputs(country: str,
            refresh: bool = False,
            data_root: Path | str | None = None) -> dict

Returns a mapping like::

    {
        "osm": <Path to rail_network.parquet>,
        "gtfs": <Path to extracted GTFS dir | None>,
        "admin": <Path to admin.gpkg>,
        "metadata": <Path to meta.json>
    }

The rest of the pipeline never has to care _how_ these files appeared.

Dependencies:  osmnx, geopandas, pandas, requests, tqdm
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
from tqdm import tqdm

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Defaults & helpers
# --------------------------------------------------------------------------- #
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
OSM_COLUMNS = ["geometry", "railway", "service", "highspeed", "gauge"]


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _download(url: str, dest: Path, chunk: int = 1024 * 1024) -> Path:
    """Stream-download with a progress bar."""
    _mkdir(dest.parent)
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0) or 0)
    with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar, dest.open("wb") as fh:
        for part in r.iter_content(chunk):
            fh.write(part)
            bar.update(len(part))
    return dest


# --------------------------------------------------------------------------- #
# Dataclass wrapper so callers can do `artifacts.osm`, etc.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class InputArtifacts:
    osm: Path
    gtfs: Path | None
    admin: Path
    metadata: Path

    def as_dict(self) -> Dict[str, Path | None]:
        return self.__dict__


# --------------------------------------------------------------------------- #
# PUBLIC ENTRY POINT
# --------------------------------------------------------------------------- #
def load_inputs(
    country: str,
    *,
    refresh: bool = False,
    data_root: Path | str | None = None,
) -> Dict[str, Path | None]:
    """
    Fetch / cache all inputs for *country* and return the file locations.

    Parameters
    ----------
    country : str
        Plain-language place name (“Belgium”, “United Kingdom”) understood by
        Nominatim.
    refresh : bool, default False
        Force re-download even if files already exist.
    data_root : str | Path | None
        Where to store caches.  Defaults to ``pipeline/data``.
    """
    data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    slug = country.lower().replace(" ", "_")
    out_dir = data_root / slug
    _mkdir(out_dir)

    LOG.info("📥  Loading inputs for %s  (refresh=%s)", country, refresh)

    # ------------------------------------------------------------------- #
    # 1) OpenStreetMap rail network
    # ------------------------------------------------------------------- #
    osm_fp = out_dir / "rail_network.parquet"
    if refresh or not osm_fp.exists():
        LOG.info("   ⤷ Fetching OSM rail network …")
        _fetch_osm_rail(country, osm_fp)
    else:
        LOG.info("   ⤷ Using cached rail network  (%s)", osm_fp.name)

    # ------------------------------------------------------------------- #
    # 2) GTFS
    # ------------------------------------------------------------------- #
    gtfs_dir = out_dir / "gtfs"
    if refresh or not gtfs_dir.exists():
        try:
            LOG.info("   ⤷ Fetching GTFS feed …")
            _fetch_gtfs(country, gtfs_dir)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("   ⚠  No GTFS feed: %s", exc)
            gtfs_dir = None
    else:
        LOG.info("   ⤷ Using cached GTFS feed   (%s)", gtfs_dir)

    # ------------------------------------------------------------------- #
    # 3) Admin boundaries
    # ------------------------------------------------------------------- #
    admin_fp = out_dir / "admin.gpkg"
    if refresh or not admin_fp.exists():
        LOG.info("   ⤷ Fetching Natural-Earth admin boundaries …")
        _fetch_admin(country, admin_fp)
    else:
        LOG.info("   ⤷ Using cached boundaries  (%s)", admin_fp.name)

    # ------------------------------------------------------------------- #
    # 4) Metadata snapshot
    # ------------------------------------------------------------------- #
    meta_fp = out_dir / "meta.json"
    if refresh or not meta_fp.exists():
        meta = {
            "country": country,
            "fetched_at": pd.Timestamp.utcnow().isoformat(),
            "osm_file": str(osm_fp),
            "gtfs_dir": str(gtfs_dir) if gtfs_dir else None,
            "admin_file": str(admin_fp),
        }
        meta_fp.write_text(json.dumps(meta, indent=2))

    return InputArtifacts(osm_fp, gtfs_dir, admin_fp, meta_fp).as_dict()


# --------------------------------------------------------------------------- #
# IMPLEMENTATIONS
# --------------------------------------------------------------------------- #
def _fetch_osm_rail(country: str, dest: Path) -> None:
    """Download every feature with ``railway=*`` inside the country boundary."""
    tags = {"railway": True}
    gdf: gpd.GeoDataFrame = ox.geometries_from_place(country, tags=tags)
    if gdf.empty:
        raise RuntimeError(f"No OSM rail features found for {country!r}")
    gdf = gdf.to_crs(4326)[OSM_COLUMNS]
    _mkdir(dest.parent)
    gdf.to_parquet(dest, index=False)


def _fetch_gtfs(country: str, dest_dir: Path) -> None:
    """Grab the first MobilityData feed for *country* (if any) and unzip it."""
    index_url = "https://storage.googleapis.com/mdb-latest/index.json"
    feeds = requests.get(index_url, timeout=30).json()
    matches = [f for f in feeds if f["location_country_code"] == country.upper()]
    if not matches:
        raise FileNotFoundError("No GTFS feed registered for this country.")
    url = matches[0]["urls"]["latest"]

    zip_fp = dest_dir.with_suffix(".zip")
    _download(url, zip_fp)
    _mkdir(dest_dir)
    shutil.unpack_archive(zip_fp, dest_dir, "zip")
    zip_fp.unlink(missing_ok=True)


def _fetch_admin(country: str, dest: Path) -> None:
    """Slice Natural-Earth Admin-1 & Admin-2 layers to the target country."""
    level1 = (
        "https://naturalearth.s3.amazonaws.com/10m_cultural/"
        "ne_10m_admin_1_states_provinces.zip"
    )
    level2 = (
        "https://naturalearth.s3.amazonaws.com/10m_cultural/"
        "ne_10m_admin_2_counties.zip"
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _download(level1, tmp / "adm1.zip")
        _download(level2, tmp / "adm2.zip")
        for z in tmp.glob("*.zip"):
            shutil.unpack_archive(z, tmp, "zip")

        try:
            _ogr_slice(tmp, dest, country)
        except (FileNotFoundError, subprocess.SubprocessError):
            LOG.debug("   ↪ ogr2ogr unavailable – falling back to geopandas.")
            _gp_slice(tmp, dest, country)


def _ogr_slice(tmp: Path, dest: Path, country: str) -> None:
    """Use ogr2ogr (fast) to filter big shapefiles by adm0_name."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for shp in tmp.glob("*.shp"):
        subprocess.run(
            [
                "ogr2ogr",
                "-f",
                "GPKG",
                "-append" if dest.exists() else "-nln",
                "admin",
                dest.as_posix(),
                shp.as_posix(),
                "-where",
                f"adm0_name = '{country}'",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


def _gp_slice(tmp: Path, dest: Path, country: str) -> None:
    """Slower fallback using pure Python / GeoPandas."""
    gdfs: List[gpd.GeoDataFrame] = []
    for shp in tmp.glob("*.shp"):
        gdf = gpd.read_file(shp)
        gdfs.append(gdf[gdf["adm0_name"] == country])
    gpd.pd.concat(gdfs).to_crs(4326).to_file(dest, layer="admin", driver="GPKG")
