# pipeline/steps/visualize.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..flows import get_logger

LOG = get_logger(__name__)

# --------------------
# IO helpers
# --------------------
def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# --------------------
# Geometry + styling utils
# --------------------
def _stable_color(key: str) -> str:
    """Deterministic color from a string key."""
    h = 0
    for ch in key:
        h = (h * 31 + ord(ch)) & 0xFFFFFF
    return f"#{h:06x}"


def _bounds_from_geoms(geoms) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Compute ((min_lat, min_lon), (max_lat, max_lon)) from a GeoSeries of geometries.
    Works with LineString/MultiLineString/Point, etc.
    """
    try:
        import shapely
        from shapely.geometry import base as _base
    except Exception:
        return None

    min_lat = min_lon = float("inf")
    max_lat = max_lon = float("-inf")

    for g in geoms:
        if g is None:
            continue
        # Iterate coordinates
        try:
            for x, y in shapely.get_coordinates(g):
                lon, lat = float(x), float(y)
                if lat < min_lat:
                    min_lat = lat
                if lat > max_lat:
                    max_lat = lat
                if lon < min_lon:
                    min_lon = lon
                if lon > max_lon:
                    max_lon = lon
        except Exception:
            # Fallback: use bounds
            try:
                minx, miny, maxx, maxy = g.bounds
                if miny < min_lat:
                    min_lat = miny
                if maxy > max_lat:
                    max_lat = maxy
                if minx < min_lon:
                    min_lon = minx
                if maxx > max_lon:
                    max_lon = maxx
            except Exception:
                continue

    if min_lat == float("inf"):
        return None
    return (min_lat, min_lon), (max_lat, max_lon)


def _classify_track(row: Dict[str, Any]) -> str:
    """
    Classify an OSM railway edge into Underground / Overpass/Elevated / Surface
    using common OSM tags: tunnel, bridge, layer.
    """
    # Normalize helpers
    def _is_yes(v):
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        return s in {"yes", "true", "1"}

    def _to_int(v, default=0):
        try:
            return int(v)
        except Exception:
            return default

    # Underground
    if _is_yes(row.get("tunnel")) or _to_int(row.get("layer"), 0) < 0:
        return "Underground"

    # Overpass / Elevated
    if _is_yes(row.get("bridge")) or _to_int(row.get("layer"), 0) > 0:
        return "Overpass/Elevated"

    # Default
    return "Surface"


def _is_rail_edge(row: Dict[str, Any]) -> bool:
    """
    Keep only railway edges (rail/light_rail/tram/subway).
    """
    val = str(row.get("railway", "")).lower()
    return val in {"rail", "light_rail", "tram", "subway"}


def _is_station_node(row: Dict[str, Any]) -> bool:
    """
    Identify station-like nodes from OSM tags.
    """
    val = str(row.get("railway", "")).lower()
    return val in {"station", "halt", "stop", "stop_position", "tram_stop"}


# --------------------
# Folium map builder
# --------------------
def _build_folium_map(
    gdf_edges,
    gdf_nodes,
    out_html: Path,
    country_name: Optional[str] = None,
) -> Optional[Path]:
    """
    Render a terrain basemap with:
      - railway edges colored by class (Underground/Overpass/Elevated/Surface)
      - station nodes as black dots
      - legend + layer controls
    """
    try:
        import folium
        from folium import features
    except Exception as e:
        LOG.warning("Folium not available (%s). Skipping learn map.", e)
        return None

    # Compute map center/bounds from edges; fallback center ~ Belgium
    bounds = _bounds_from_geoms(gdf_edges.geometry if gdf_edges is not None else [])
    if bounds:
        (min_lat, min_lon), (max_lat, max_lon) = bounds
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
    else:
        center_lat, center_lon = 50.5039, 4.4699

    # Base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, control_scale=True)

    # Terrain / hillshade layers (as overlays so they can sit "on top" if desired)
    # ESRI World Hillshade (overlay)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Hillshade",
        name="Terrain (Hillshade)",
        overlay=True,
        control=True,
        opacity=0.8,
    ).add_to(m)

    # OpenTopoMap as alternative base
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="&copy; OpenTopoMap (CC-BY-SA)",
        name="OpenTopoMap",
        overlay=False,
        control=True,
    ).add_to(m)

    # OSM Standard for context
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False, control=True).add_to(m)

    # Split edges by class
    classes = ["Underground", "Overpass/Elevated", "Surface"]
    class_colors = {
        "Underground": "#e74c3c",        # red
        "Overpass/Elevated": "#f1c40f",  # yellow
        "Surface": "#2ecc71",            # green
    }

    for cls in classes:
        sub = gdf_edges[gdf_edges["__track_class__"] == cls]
        if sub.empty:
            continue

        def style_function(_):
            return {"color": class_colors.get(cls, _stable_color(cls)), "weight": 4, "opacity": 0.95}

        gj = folium.GeoJson(
            sub.__geo_interface__,
            name=f"Tracks — {cls}",
            style_function=style_function,
            highlight_function=lambda f: {"weight": 6},
            overlay=True,
        )
        gj.add_to(m)

    # Stations: small black dots
    if gdf_nodes is not None and not gdf_nodes.empty:
        station_group = folium.FeatureGroup(name="Stations", overlay=True, control=True, show=True)
        for _, row in gdf_nodes.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            try:
                lat = geom.y
                lon = geom.x
            except Exception:
                continue
            name = row.get("name") or row.get("ref") or "Station"
            folium.CircleMarker(
                location=(lat, lon),
                radius=2.5,
                color="#000000",
                fill=True,
                fill_color="#000000",
                fill_opacity=1.0,
                weight=1,
                tooltip=str(name),
            ).add_to(station_group)
        station_group.add_to(m)

    # Legend (fixed HTML overlay)
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 18px; left: 18px; z-index: 9999;
        background: rgba(255,255,255,0.92);
        padding: 10px 12px; border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.2);
        font-size: 13px; line-height: 1.4;
    ">
      <div style="font-weight:600; margin-bottom:6px;">Railway Legend</div>
      <div><span style="display:inline-block;width:12px;height:3px;background:#e74c3c;margin-right:6px;vertical-align:middle;"></span> Underground</div>
      <div><span style="display:inline-block;width:12px;height:3px;background:#f1c40f;margin-right:6px;vertical-align:middle;"></span> Overpass / Elevated</div>
      <div><span style="display:inline-block;width:12px;height:3px;background:#2ecc71;margin-right:6px;vertical-align:middle;"></span> Surface</div>
      <div style="margin-top:4px;"><span style="display:inline-block;width:8px;height:8px;background:#000;border-radius:50%;margin-right:8px;vertical-align:middle;"></span> Station</div>
    </div>
    """
    folium.map.CustomPane("legend").add_to(m)  # ensures above tiles
    folium.Marker(
        location=[-90, -180],  # off-map placeholder; we'll attach raw HTML instead
        icon=folium.DivIcon(html=legend_html),
    ).add_to(m)

    # Fit bounds to edges (one country only)
    if bounds:
        (min_lat, min_lon), (max_lat, max_lon) = bounds
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    folium.LayerControl(collapsed=False).add_to(m)

    _ensure_parent(out_html)
    m.save(str(out_html))
    LOG.info("Wrote learn HTML map: %s", out_html)
    return out_html


# --------------------
# Public API
# --------------------
def build_learn_map(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds a learn-stage visualization:
      - reads ingest artifacts: edges.parquet + nodes.parquet
      - filters for railway features
      - classifies edges into Underground / Overpass/Elevated / Surface
      - renders terrain + tracks + stations to artifacts/reports/learn_map.html
    Returns {"map_html": "..."} if created (or empty dict if skipped).
    """
    artifacts_dir = Path(config.get("artifacts_dir") or config.get("artifacts") or "artifacts")
    ingest_dir = artifacts_dir / "ingest"
    reports_dir = artifacts_dir / "reports"

    edges_pq = ingest_dir / "edges.parquet"
    nodes_pq = ingest_dir / "nodes.parquet"
    out_html = reports_dir / "learn_map.html"

    try:
        import geopandas as gpd
    except Exception as e:
        LOG.warning("GeoPandas not available (%s). Skipping learn map.", e)
        return {}

    if not edges_pq.exists():
        LOG.warning("Edges parquet not found at %s. Skipping learn map.", edges_pq)
        return {}

    # Load edges
    try:
        gdf_edges = gpd.read_parquet(edges_pq)
        if gdf_edges.empty or "geometry" not in gdf_edges:
            LOG.warning("Edges parquet has no geometry/rows. Skipping learn map.")
            return {}
    except Exception as e:
        LOG.warning("Failed to read edges parquet (%s). Skipping learn map.", e)
        return {}

    # Keep only rails
    try:
        # Normalize for row-wise checks
        sel = []
        for _, r in gdf_edges.drop(columns=["geometry"], errors="ignore").iterrows():
            sel.append(_is_rail_edge(dict(r)))
        import pandas as pd
        mask = pd.Series(sel, index=gdf_edges.index) if len(sel) == len(gdf_edges) else None
        if mask is not None:
            gdf_edges = gdf_edges[mask]
        if gdf_edges.empty:
            LOG.warning("No railway edges after filtering. Skipping learn map.")
            return {}
    except Exception:
        # If anything odd, proceed without filtering
        pass

    # Classify tracks
    try:
        gdf_edges["__track_class__"] = [
            _classify_track(dict(r)) for _, r in gdf_edges.drop(columns=["geometry"], errors="ignore").iterrows()
        ]
    except Exception:
        # Best effort default
        gdf_edges["__track_class__"] = "Surface"

    # Load station nodes (optional)
    gdf_nodes = None
    if nodes_pq.exists():
        try:
            gdf_nodes_full = gpd.read_parquet(nodes_pq)
            if "geometry" in gdf_nodes_full and not gdf_nodes_full.empty:
                import pandas as pd
                mask_nodes = []
                for _, r in gdf_nodes_full.drop(columns=["geometry"], errors="ignore").iterrows():
                    mask_nodes.append(_is_station_node(dict(r)))
                mask_nodes = pd.Series(mask_nodes, index=gdf_nodes_full.index)
                gdf_nodes = gdf_nodes_full[mask_nodes]
            else:
                gdf_nodes = None
        except Exception:
            gdf_nodes = None

    # Build map
    written = _build_folium_map(
        gdf_edges=gdf_edges,
        gdf_nodes=gdf_nodes,
        out_html=out_html,
        country_name=str(config.get("scenario_name") or ""),
    )

    return {"map_html": str(written)} if written else {}
