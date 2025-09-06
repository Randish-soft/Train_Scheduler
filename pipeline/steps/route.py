# pipeline/steps/route.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import log
from ..flows import import_optional

pd = import_optional("pandas")
gpd = import_optional("geopandas")
np = import_optional("numpy")
nx = import_optional("networkx")
joblib = import_optional("joblib")
shapely = import_optional("shapely")

# -------------------- Constants --------------------
FEATURES_DIRNAME = "features"
INGEST_DIRNAME = "ingest"
RUN_ROUTES_JSON = "routes_summary.json"

# -------------------- Paths --------------------
@dataclass
class Paths:
    artifacts_dir: Path
    features_path: Optional[Path]
    edges_path: Optional[Path]
    runs_dir: Path

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> "Paths":
        try:
            artifacts_root = Path(cfg.get("artifacts_dir") or cfg.get("artifacts") or "artifacts")
            feats = artifacts_root / FEATURES_DIRNAME / "edge_features.parquet"
            feats = feats if feats.exists() else None
            edges = artifacts_root / INGEST_DIRNAME / "edges.parquet"
            if not edges.exists():
                alt = edges.with_suffix(".csv")
                edges = alt if alt.exists() else None
            p = Paths(
                artifacts_dir=artifacts_root,
                features_path=feats,
                edges_path=edges,
                runs_dir=artifacts_root / "runs",
            )
            p.runs_dir.mkdir(parents=True, exist_ok=True)
            return p
        except Exception as e:
            log.error("Failed to resolve routing paths: %s", e, exc_info=True)
            raise


# -------------------- I/O helpers --------------------
def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        log.debug("Wrote JSON: %s", path)
    except Exception as e:
        log.error("Failed to write %s: %s", path, e, exc_info=True)
        raise


def _load_table(path: Path):
    try:
        if str(path).endswith(".parquet"):
            if pd is None:
                raise RuntimeError("pandas required to read parquet.")
            return pd.read_parquet(path)
        if str(path).endswith(".csv"):
            if pd is None:
                raise RuntimeError("pandas required to read csv.")
            return pd.read_csv(path)
        if gpd:
            return gpd.read_file(path)  # gpkg/geojson, etc.
        raise RuntimeError(f"Unsupported format: {path.suffix}")
    except Exception as e:
        log.error("Failed to read table %s: %s", path, e, exc_info=True)
        raise


# -------------------- Models / cost --------------------
def _load_cost_model(models: Dict[str, Any]):
    """
    Best-effort loader for a pickled cost model (xgboost/sklearn/etc.).
    Returns None if unavailable or load fails.
    """
    try:
        path = None
        for k in ("cost", "cost_model", "capex"):
            if k in models:
                path = models[k]
                break
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            log.warning("Declared cost model not found: %s", p)
            return None
        if not joblib:
            log.warning("joblib not available; cannot load cost model at %s", p)
            return None
        model = joblib.load(p)  # type: ignore
        log.info("Loaded cost model: %s", p)
        return model
    except Exception as e:
        log.warning("Failed to load cost model; using heuristic costs. %s", e)
        return None


def _heuristic_cost_row(row: Dict[str, Any]) -> float:
    """
    Very rough CAPEX €/km heuristic by structure + slope + urbanity.
    Tunable defaults to keep pipeline running without a trained model.
    """
    base = 8_000_000.0  # at-grade, rural €/km
    structure = str(row.get("structure", "")).lower()
    slope = row.get("slope_pct")
    urban = str(row.get("env", "")).lower() == "urban"

    if "tunnel" in structure:
        base = 45_000_000.0
    elif "bridge" in structure:
        base = 25_000_000.0
    elif "elevated" in structure or "viaduct" in structure:
        base = 18_000_000.0

    if isinstance(slope, (int, float)) and not (np and np.isnan(slope)):
        if slope > 2.5:
            base *= 1.25
        if slope > 4.0:
            base *= 1.5

    if urban:
        base *= 1.35

    return float(base)


def _capex_for_edge_km(row: Dict[str, Any], model) -> float:
    if model is None:
        return _heuristic_cost_row(row)
    try:
        # Build a tiny single-row feature vector the model likely understands.
        # We keep it defensive—if predict fails, fallback to heuristic.
        cols = [
            "len_km",
            "curvature_rad_per_m",
            "slope_pct",
            "max_speed_kph",
            "track_count",
            "is_tunnel",
            "is_bridge",
            "is_elevated",
            "is_urban",
        ]
        x = [[
            float(row.get("len_km", 0.0) or 0.0),
            float(row.get("curvature_rad_per_m") or 0.0) if row.get("curvature_rad_per_m") is not None else 0.0,
            float(row.get("slope_pct") or 0.0) if row.get("slope_pct") is not None else 0.0,
            float(row.get("max_speed_kph") or 0.0) if row.get("max_speed_kph") is not None else 0.0,
            float(row.get("track_count") or 1.0) if row.get("track_count") is not None else 1.0,
            1.0 if str(row.get("structure", "")).lower() == "tunnel" else 0.0,
            1.0 if str(row.get("structure", "")).lower() == "bridge" else 0.0,
            1.0 if str(row.get("structure", "")).lower() == "elevated" else 0.0,
            1.0 if str(row.get("env", "")).lower() == "urban" else 0.0,
        ]]
        y = model.predict(x)  # type: ignore[attr-defined]
        y = float(y[0])
        # guardrail: clamp to reasonable bounds
        y = max(2_000_000.0, min(y, 120_000_000.0))
        return y
    except Exception as e:
        log.warning("Model prediction failed; fallback heuristic. %s", e)
        return _heuristic_cost_row(row)


# -------------------- Graph build & routing --------------------
def _build_graph_from_features(feats) -> Optional[Any]:
    """
    Build a NetworkX graph from a features table that has ('u','v','len_km') columns.
    """
    if nx is None:
        log.warning("networkx not available; cannot build routing graph.")
        return None
    try:
        required = ["u", "v", "len_km"]
        for c in required:
            if c not in feats.columns:
                raise ValueError(f"Features missing required column '{c}' for routing.")
        G = nx.DiGraph()
        for _, r in feats.iterrows():
            u = r.get("u")
            v = r.get("v")
            if pd.isna(u) or pd.isna(v):
                continue
            G.add_edge(u, v, **r.to_dict())
        log.info("Built routing graph with %d nodes, %d edges.", G.number_of_nodes(), G.number_of_edges())
        return G
    except Exception as e:
        log.error("Failed to build graph from features: %s", e, exc_info=True)
        return None


def _select_od_pairs(config: Dict[str, Any], feats) -> List[Tuple[Any, Any]]:
    """
    Choose origin-destination pairs for routing.
    Priority:
      1) config['od_pairs'] as [(u,v), ...]
      2) config['od_coords'] as [{'name':..., 'x':..., 'y':...}, ...] -> nearest nodes
      3) Heuristic: pick two largest-degree nodes in the graph (if available)
    """
    # Direct node ids
    if "od_pairs" in config and isinstance(config["od_pairs"], list) and len(config["od_pairs"]) > 0:
        return [tuple(p) for p in config["od_pairs"] if isinstance(p, (list, tuple)) and len(p) == 2]

    # Coordinate-based (needs geometry & nearest search)
    if "od_coords" in config and gpd and "geometry" in feats.columns:
        try:
            from shapely.geometry import Point
            nodes = pd.DataFrame({"node": pd.concat([feats["u"], feats["v"]]).unique()})
            # crude proxy: take start points of edges for u and v sets
            starts = feats.drop_duplicates("u")[["u", "geometry"]].rename(columns={"u": "node"})
            ends = feats.drop_duplicates("v")[["v", "geometry"]].rename(columns={"v": "node"})
            ndgeo = pd.concat([starts, ends]).drop_duplicates("node")
            ndgeo = gpd.GeoDataFrame(ndgeo, geometry="geometry", crs="EPSG:3857")
            pairs = []
            for i in range(0, len(config["od_coords"]) - 1, 2):
                a, b = config["od_coords"][i], config["od_coords"][i + 1]
                pa = Point(a["x"], a["y"])
                pb = Point(b["x"], b["y"])
                na = ndgeo.distance(gpd.GeoSeries([pa], crs=ndgeo.crs).iloc[0]).idxmin()
                nb = ndgeo.distance(gpd.GeoSeries([pb], crs=ndgeo.crs).iloc[0]).idxmin()
                pairs.append((ndgeo.loc[na, "node"], ndgeo.loc[nb, "node"]))
            if pairs:
                return pairs
        except Exception as e:
            log.warning("OD from coordinates failed; %s", e)

    # Heuristic fallback: high-degree nodes
    try:
        if nx is not None:
            G = _build_graph_from_features(feats)
            if G and G.number_of_nodes() >= 2:
                deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
                return [(deg[0][0], deg[1][0])]
    except Exception:
        pass

    # Last resort: single dummy pair
    return [(None, None)]


def _k_shortest_path(G, source, target, weight: str = "weight", k: int = 1):
    """
    Generator for up to k simple shortest paths. Falls back to single path if k==1.
    """
    if G is None or source is None or target is None:
        return []
    try:
        if k <= 1:
            path = nx.shortest_path(G, source=source, target=target, weight=weight)
            return [path]
        # Yen's algorithm if available
        try:
            from networkx.algorithms.simple_paths import shortest_simple_paths
            gen = shortest_simple_paths(G, source, target, weight=weight)
            paths = []
            for i, p in enumerate(gen):
                paths.append(p)
                if len(paths) >= k:
                    break
            return paths
        except Exception:
            path = nx.shortest_path(G, source=source, target=target, weight=weight)
            return [path]
    except Exception as e:
        log.error("Shortest path failed (%s -> %s): %s", source, target, e, exc_info=True)
        return []


# -------------------- Line assembly --------------------
def _assemble_line_from_path(feats, path_nodes: List[Any], capex_model) -> Dict[str, Any]:
    """
    Given a node sequence, pull matching edges (u->v) in order, compute per-edge costs,
    aggregate to sections by 'structure' and 'max_speed_kph', and form a line dict.
    """
    if not path_nodes or len(path_nodes) < 2:
        return {"id": "L0", "geometry": None, "meta": {"notes": "empty path"}, "sections": []}

    # Build a quick lookup for edges by (u,v)
    try:
        key = ("u", "v")
        if key[0] not in feats.columns or key[1] not in feats.columns:
            raise ValueError("Features missing u/v columns.")
        # Multi-index lookup
        kv = feats.set_index(list(key))
    except Exception as e:
        log.warning("Failed to index features by (u,v): %s", e)
        kv = None

    edges_seq = []
    total_len_km = 0.0
    total_capex = 0.0
    sections: List[Dict[str, Any]] = []

    # Reconstruct edge sequence
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        row = None
        if kv is not None:
            try:
                r = kv.loc[(u, v)]
                # If multiple rows match, take the shortest
                if hasattr(r, "iloc"):
                    r = r.iloc[r["len_km"].astype(float).argmin()]
                row = r.to_dict()
            except Exception:
                row = None
        if row is None:
            # fallback: pick any edge with same u/v from feats
            try:
                cand = feats[(feats["u"] == u) & (feats["v"] == v)]
                if len(cand) > 0:
                    r = cand.iloc[cand["len_km"].astype(float).argmin()]
                    row = r.to_dict()
            except Exception:
                row = None

        if row is None:
            log.warning("Missing edge for step %s->%s; skipping.", u, v)
            continue

        # Costs
        capex_per_km = _capex_for_edge_km(row, capex_model)
        len_km = float(row.get("len_km", 0.0) or (row.get("length_m", 0.0) or 0.0) / 1000.0)
        capex = capex_per_km * len_km

        edges_seq.append({
            "u": u, "v": v,
            "len_km": len_km,
            "structure": row.get("structure"),
            "speed_kph": row.get("max_speed_kph"),
            "capex_eur": capex,
            "capex_eur_km": capex_per_km,
        })
        total_len_km += len_km
        total_capex += capex

    # Aggregate into sections by (structure, speed_kph)
    cur = None
    for e in edges_seq:
        key = (e.get("structure"), e.get("speed_kph"))
        if cur is None or cur["structure"] != key[0] or cur["speed_kph"] != key[1]:
            if cur is not None:
                sections.append(cur)
            cur = {
                "seq": len(sections),
                "structure": key[0],
                "speed_kph": key[1],
                "len_km": 0.0,
                "capex_eur": 0.0,
            }
        cur["len_km"] += float(e["len_km"])
        cur["capex_eur"] += float(e["capex_eur"])
    if cur is not None:
        sections.append(cur)

    # Geometry (optional)
    geom = None
    if gpd is not None and "geometry" in feats.columns and shapely is not None:
        try:
            from shapely.ops import linemerge
            ls_parts = []
            for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                seg = feats[(feats["u"] == u) & (feats["v"] == v)]
                if len(seg) > 0 and seg.iloc[0].get("geometry") is not None:
                    ls_parts.append(seg.iloc[0]["geometry"])
            if ls_parts:
                geom = linemerge(ls_parts) if len(ls_parts) > 1 else ls_parts[0]
        except Exception as e:
            log.warning("Failed to assemble geometry for line: %s", e)

    return {
        "id": "L1",
        "geometry": geom,  # may be None
        "meta": {
            "name": "AutoLine-1",
            "length_km": float(total_len_km),
            "capex_eur": float(total_capex),
            "notes": "Auto-generated by route step",
        },
        "sections": sections,
    }


# -------------------- Public API --------------------
def build_routes(config: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build one or more line candidates:
      - Load features (preferred) or edges (fallback)
      - Construct a NetworkX graph with edge weights = capex/km * len_km (+ small penalty by curvature)
      - Solve shortest path for requested OD pairs (config), defaulting to a heuristic pair
      - Assemble line dict(s) with sections, costs, and optional geometry
    """
    if pd is None:
        raise RuntimeError("pandas is required by route step but not available.")

    paths = Paths.from_config(config)

    # Load features OR edges
    feats = None
    used_features = False
    if paths.features_path:
        try:
            feats = _load_table(paths.features_path)
            used_features = True
            log.info("Routing will use features: %s", paths.features_path)
        except Exception as e:
            log.warning("Failed to load features; trying ingest edges. %s", e)

    if feats is None:
        if not paths.edges_path or not Path(paths.edges_path).exists():
            log.error("No features or edges available for routing.")
            # Hard fallback: straight-line demo if coords in config; else empty.
            return _fallback_straight_line(config)
        try:
            feats = _load_table(paths.edges_path)
            log.info("Routing will use ingest edges: %s", paths.edges_path)
            # Ensure basic columns compatible with features step
            if "length_m" in feats.columns and "len_km" not in feats.columns:
                feats["len_km"] = feats["length_m"].astype(float) / 1000.0
            for c in ("curvature_rad_per_m", "slope_pct", "env", "structure", "max_speed_kph", "track_count"):
                if c not in feats.columns:
                    feats[c] = None
        except Exception as e:
            log.error("Failed to load edges for routing: %s", e, exc_info=True)
            return _fallback_straight_line(config)

    # Build graph
    G = _build_graph_from_features(feats)
    if G is None or G.number_of_edges() == 0:
        log.warning("Graph unavailable/empty; using straight-line fallback.")
        return _fallback_straight_line(config)

    # Edge weights = capex_per_km * len_km + curvature penalty
    capex_model = _load_cost_model(models)
    try:
        for u, v, data in G.edges(data=True):
            cost_per_km = _capex_for_edge_km(data, capex_model)
            length_km = float(data.get("len_km", 0.0))
            curvature = float(data.get("curvature_rad_per_m") or 0.0)
            penalty = 250_000.0 * curvature * length_km  # small curvature discouragement
            G[u][v]["weight"] = cost_per_km * length_km + penalty
    except Exception as e:
        log.warning("Failed setting custom weights; defaulting to len_km. %s", e)
        for u, v, data in G.edges(data=True):
            G[u][v]["weight"] = float(data.get("len_km", 1.0))

    # Determine OD pairs
    od_pairs = _select_od_pairs(config, feats)
    lines = []
    for idx, (o, d) in enumerate(od_pairs):
        if o is None or d is None:
            log.warning("OD pair %d invalid; skipping graph search.", idx)
            continue
        paths = _k_shortest_path(G, o, d, weight="weight", k=int(config.get("k_paths", 1)))
        if not paths:
            log.warning("No path found for OD pair %s -> %s", o, d)
            continue
        # Take the best path for now
        best = paths[0]
        line = _assemble_line_from_path(feats, best, capex_model)
        line["id"] = f"L{idx+1}"
        lines.append(line)

    if not lines:
        log.warning("No graph-based lines built; using straight-line fallback.")
        return _fallback_straight_line(config)

    summary = {
        "lines": lines,
        "used_features": bool(used_features),
        "n_lines": len(lines),
        "notes": "Routing completed",
    }
    try:
        _safe_write_json(paths.runs_dir / RUN_ROUTES_JSON, summary)
    except Exception as e:
        log.warning("Failed to write routes summary JSON: %s", e, exc_info=True)

    return summary


# -------------------- Fallback: straight line --------------------
def _fallback_straight_line(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    If we can't build a graph, we still emit a valid structure so downstream
    steps (timetable/report) can run. Requires two coords in config['od_coords'].
    """
    if not (gpd and shapely):
        log.warning("GeoPandas/Shapely not available; returning geometry=None fallback.")
        return {
            "lines": [{
                "id": "L1",
                "geometry": None,
                "meta": {"name": "Fallback Line", "length_km": None, "capex_eur": None, "notes": "no-geo"},
                "sections": [{"seq": 0, "structure": None, "speed_kph": None, "len_km": None, "capex_eur": None}],
            }],
            "used_features": False,
            "n_lines": 1,
            "notes": "Fallback without geometry",
        }

    try:
        from shapely.geometry import LineString, Point
        crs = "EPSG:3857"
        coords = config.get("od_coords") or []
        if len(coords) >= 2:
            a, b = coords[0], coords[1]
            pa = (float(a["x"]), float(a["y"]))
            pb = (float(b["x"]), float(b["y"]))
        else:
            # Brussels Central → Antwerp-Centraal (rough coords in EPSG:3857)
            pa = (420000, 6500000)
            pb = (450000, 6570000)

        ls = LineString([pa, pb])
        length_km = float(ls.length / 1000.0)

        lines = [{
            "id": "L1",
            "geometry": ls,
            "meta": {
                "name": "Fallback Line",
                "length_km": length_km,
                "capex_eur": 8_000_000.0 * length_km,  # heuristic
                "notes": "straight-line fallback",
            },
            "sections": [{
                "seq": 0,
                "structure": "at_grade",
                "speed_kph": 120,
                "len_km": length_km,
                "capex_eur": 8_000_000.0 * length_km,
                "capex_eur_km": 8_000_000.0,
            }],
        }]

        return {
            "lines": lines,
            "used_features": False,
            "n_lines": 1,
            "notes": "Fallback straight line",
        }
    except Exception as e:
        log.warning("Straight-line fallback failed: %s", e)
        return {
            "lines": [{
                "id": "L1",
                "geometry": None,
                "meta": {"name": "Fallback Line", "length_km": None, "capex_eur": None, "notes": "no-geo"},
                "sections": [{"seq": 0, "structure": None, "speed_kph": None, "len_km": None, "capex_eur": None}],
            }],
            "used_features": False,
            "n_lines": 1,
            "notes": "Fallback without geometry",
        }
