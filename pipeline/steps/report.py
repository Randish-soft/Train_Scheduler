# pipeline/steps/report.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ..flows import get_logger

LOG = get_logger(__name__)


# --------------------
# Helpers (file IO)
# --------------------
def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Write a list of dict rows to CSV with header from the union of keys.
    """
    import csv

    _ensure_parent(path)
    # Collect header keys in a stable order
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    LOG.info("Wrote CSV: %s (%d rows)", path, len(rows))


def _write_json(path: Path, payload: Any) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    LOG.debug("Wrote JSON: %s", path)


# --------------------
# Geometry utilities
# --------------------
def _geom_to_geojson_mapping(geom: Any) -> Optional[Dict[str, Any]]:
    """
    Convert a geometry-ish object to a GeoJSON geometry mapping.
    Accepts:
      - shapely geometry
      - already-GeoJSON-like dict with 'type' and 'coordinates'
      - dict with 'coordinates' only (assume LineString)
      - list of coords (assume LineString; coords must be [lon, lat] or [lon, lat, z])
    Returns None if conversion fails.
    """
    # Already a GeoJSON-like geometry?
    if isinstance(geom, dict) and "type" in geom and "coordinates" in geom:
        return geom

    # Shapely?
    try:
        from shapely.geometry.base import BaseGeometry  # type: ignore
        from shapely.geometry import mapping  # type: ignore

        if isinstance(geom, BaseGeometry):
            return mapping(geom)
    except Exception:
        pass

    # Bare coordinates (list) — assume LineString
    if isinstance(geom, (list, tuple)) and geom and isinstance(geom[0], (list, tuple)):
        return {"type": "LineString", "coordinates": list(geom)}

    # Dictionary with coordinates only
    if isinstance(geom, dict) and "coordinates" in geom and isinstance(geom["coordinates"], (list, tuple)):
        return {"type": geom.get("type", "LineString"), "coordinates": list(geom["coordinates"])}

    return None


def _feature_from_line(line: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a line dict (from routes['lines']) into a GeoJSON Feature.
    Expects keys like 'id', 'name', 'geometry', plus any extra metadata.
    """
    g = _geom_to_geojson_mapping(line.get("geometry"))
    if g is None:
        return None

    props = {k: v for k, v in line.items() if k not in {"geometry"}}
    return {"type": "Feature", "geometry": g, "properties": props}


def _flatten_coords(coords: Union[List, Tuple]) -> Iterable[Tuple[float, float]]:
    """
    Recursively flatten GeoJSON coordinates into (lat, lon) tuples.
    Supports LineString, MultiLineString, Polygon, MultiPolygon.
    """
    if not coords:
        return
    # Single position?
    if isinstance(coords[0], (float, int)) and (len(coords) == 2 or len(coords) == 3):
        lon, lat = coords[0], coords[1]
        yield (lat, lon)
        return
    # Nested
    for part in coords:
        yield from _flatten_coords(part)


def _bounds_from_feature_collection(fc: Dict[str, Any]) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Compute ((min_lat, min_lon), (max_lat, max_lon)) for a FeatureCollection.
    Returns None if no coordinates found.
    """
    min_lat = min_lon = float("inf")
    max_lat = max_lon = float("-inf")

    for feat in fc.get("features", []):
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        for lat, lon in _flatten_coords(coords):
            if lat < min_lat:
                min_lat = lat
            if lat > max_lat:
                max_lat = lat
            if lon < min_lon:
                min_lon = lon
            if lon > max_lon:
                max_lon = lon

    if min_lat == float("inf"):
        return None
    return (min_lat, min_lon), (max_lat, max_lon)


def _stable_color(key: str) -> str:
    """
    Deterministic hex color from a string key.
    """
    h = 0
    for ch in (key or "line"):
        h = (h * 31 + ord(ch)) & 0xFFFFFF
    return f"#{h:06x}"


# --------------------
# Visualizer (Folium)
# --------------------
def _build_folium_map(fc: Dict[str, Any], out_html: Path) -> Optional[Path]:
    """
    Render a FeatureCollection to a Folium HTML map.
    Returns out_html if written, otherwise None (e.g., if folium missing).
    """
    try:
        import folium  # type: ignore
    except Exception as e:
        LOG.warning("Folium not available (%s). Skipping HTML map.", e)
        return None

    bounds = _bounds_from_feature_collection(fc)
    # Default center on Belgium if no bounds
    if bounds:
        (min_lat, min_lon), (max_lat, max_lon) = bounds
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
    else:
        center_lat, center_lon = 50.5039, 4.4699

    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, control_scale=True, prefer_canvas=True)

    def style_function(feature: Dict[str, Any]) -> Dict[str, Any]:
        props = feature.get("properties", {})
        line_key = str(props.get("line_id") or props.get("id") or props.get("name") or "line")
        return {"color": _stable_color(line_key), "weight": 4, "opacity": 0.9}

    def tooltip_function(feature: Dict[str, Any]):
        props = feature.get("properties", {})
        name = props.get("name") or props.get("line_id") or props.get("id") or "alignment"
        return folium.Tooltip(str(name))

    folium.GeoJson(
        fc,
        name="Alignments",
        style_function=style_function,
        tooltip=tooltip_function,
        highlight_function=lambda f: {"weight": 6},
        embed=False,
    ).add_to(m)

    if bounds:
        (min_lat, min_lon), (max_lat, max_lon) = bounds
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    folium.LayerControl().add_to(m)

    _ensure_parent(out_html)
    m.save(str(out_html))
    LOG.info("Wrote HTML map: %s", out_html)
    return out_html


# --------------------
# Public API
# --------------------
def build_reports(
    config: Dict[str, Any],
    routes: Dict[str, Any],
    timetable: Dict[str, Any],
    models: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create standard pipeline reports/exports:
      - CSV: cost book
      - CSV: timetable
      - GeoJSON: line alignments
      - HTML: interactive map (if folium available)

    Returns a dict pointing to produced artifacts.
    """
    artifacts_dir = Path(config.get("artifacts_dir") or config.get("artifacts") or "artifacts")
    reports_dir = artifacts_dir / "reports"
    geo_dir = artifacts_dir / "geo"

    # ---- Timetable CSV ----
    timetable_rows: List[Dict[str, Any]] = []
    tt = timetable.get("timetable", [])
    if isinstance(tt, list):
        for row in tt:
            if isinstance(row, dict):
                timetable_rows.append(row)
    timetable_csv = reports_dir / "timetable.csv"
    _write_csv(timetable_csv, timetable_rows)

    # ---- Cost Book CSV (placeholder: keep existing shape if provided) ----
    cost_rows: List[Dict[str, Any]] = []
    cb = timetable.get("cost_book") or routes.get("cost_book") or []
    if isinstance(cb, list) and cb and isinstance(cb[0], dict):
        cost_rows = cb
    else:
        # Minimal placeholder so downstream users have a file to inspect
        cost_rows = [{
            "scenario": config.get("scenario_name"),
            "lines_count": len(routes.get("lines", [])),
        }]
    cost_book_csv = reports_dir / "cost_book.csv"
    _write_csv(cost_book_csv, cost_rows)

    # ---- Alignments GeoJSON ----
    feats: List[Dict[str, Any]] = []
    for line in routes.get("lines", []):
        if not isinstance(line, dict):
            continue
        feat = _feature_from_line(line)
        if feat:
            feats.append(feat)

    fc = {"type": "FeatureCollection", "features": feats}
    alignments_geojson = geo_dir / "alignments.geojson"
    _ensure_parent(alignments_geojson)
    _write_json(alignments_geojson, fc)
    LOG.info("Wrote alignments GeoJSON: %s", alignments_geojson)

    # ---- HTML Map (optional) ----
    map_html_path = reports_dir / "map.html"
    map_written = _build_folium_map(fc, map_html_path)

    result = {
        "reports": {
            "cost_book_csv": str(cost_book_csv),
            "timetable_csv": str(timetable_csv),
            "alignments_geojson": str(alignments_geojson),
        }
    }
    if map_written:
        result["reports"]["map_html"] = str(map_written)

    return result
