# pipeline/steps/timetable.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import log
from ..flows import import_optional

pd = import_optional("pandas")
np = import_optional("numpy")

RUN_TIMETABLE_JSON = "timetable_plan.json"


# ---------- Helpers ----------
def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        log.debug("Wrote JSON: %s", path)
    except Exception as e:
        log.error("Failed to write JSON %s: %s", path, e, exc_info=True)
        raise


def _parse_hhmm(s: str) -> Tuple[int, int]:
    try:
        hh, mm = s.split(":")
        h, m = int(hh), int(mm)
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError
        return h, m
    except Exception:
        raise ValueError(f"Invalid time string '{s}', expected HH:MM.")


def _service_window(window: str) -> Tuple[time, time]:
    """
    Parses "HH:MM-HH:MM" to (start_time, end_time). If end < start, wraps past midnight.
    """
    try:
        start_s, end_s = window.split("-")
        sh, sm = _parse_hhmm(start_s.strip())
        eh, em = _parse_hhmm(end_s.strip())
        return time(sh, sm), time(eh, em)
    except Exception:
        # safe default
        log.warning("Invalid operating_hours '%s'; using 06:00-22:00 default.", window)
        return time(6, 0), time(22, 0)


def _secs(hours: float) -> int:
    try:
        return int(round(hours * 3600.0))
    except Exception:
        return 0


def _minutes_to_str(mins: int) -> str:
    h = mins // 60
    m = mins % 60
    return f"{h:02d}:{m:02d}"


def _time_to_minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def _minutes_to_time(m: int) -> time:
    m = m % (24 * 60)
    return time(m // 60, m % 60)


def _default_speed(config: Dict[str, Any]) -> float:
    try:
        sc = config.get("speed_classes_kph") or []
        if isinstance(sc, list) and sc:
            return float(max(sc))
    except Exception:
        pass
    return 100.0  # fallback kph


def _estimate_runtime_seconds(line: Dict[str, Any], default_speed_kph: float, dwell_s_default: int, stops: int) -> int:
    """
    Sum section running times + total dwell. If section speed missing, use default_speed_kph.
    dwell applies at intermediate stops only (stops-1).
    """
    total_hours = 0.0
    try:
        sections = line.get("sections") or []
        if sections:
            for s in sections:
                len_km = float(s.get("len_km") or s.get("length_km") or 0.0)
                v = s.get("speed_kph")
                if v is None or (isinstance(v, (int, float)) and v <= 0):
                    v = default_speed_kph
                total_hours += (len_km / float(v)) if v else 0.0
        else:
            # aggregate via meta length
            length_km = float(line.get("meta", {}).get("length_km") or 0.0)
            total_hours += (length_km / default_speed_kph) if default_speed_kph > 0 else 0.0
    except Exception as e:
        log.warning("Runtime calc encountered an issue; falling back to meta length. %s", e)
        length_km = float(line.get("meta", {}).get("length_km") or 0.0)
        total_hours = (length_km / default_speed_kph) if default_speed_kph > 0 else 0.0

    total_secs = _secs(total_hours)
    # dwell at intermediate stops only
    dwell_total = max(0, stops - 1) * int(dwell_s_default)
    return max(0, total_secs + dwell_total)


def _infer_stop_names(line: Dict[str, Any], n_stops: int) -> List[str]:
    """
    Makes placeholder stop names if none are provided by the route.
    """
    base = line.get("meta", {}).get("name") or line.get("id", "Line")
    return [f"{base}_S{i}" for i in range(n_stops)]


def _infer_stop_count(config: Dict[str, Any], line: Dict[str, Any]) -> int:
    """
    If the route already provides explicit stops, use those.
    Otherwise estimate from length and min spacing.
    """
    try:
        if "stops" in line and isinstance(line["stops"], list) and len(line["stops"]) >= 2:
            return max(2, len(line["stops"]))
    except Exception:
        pass

    try:
        length_km = float(line.get("meta", {}).get("length_km") or 0.0)
    except Exception:
        length_km = 0.0

    spacing_km = 3.0  # default heuristic
    try:
        spacing_km = float(config.get("stations", {}).get("min_spacing_km") or spacing_km)
    except Exception:
        pass

    if length_km <= 0.0:
        return 2
    # stops = endpoints + internal stops ~ floor(length/spacing) + 1
    internal = max(0, int(length_km // max(spacing_km, 0.5)))
    return max(2, 1 + internal + 1)


def _expand_departures(headway_min: int, start: time, end: time, cap: int = 2000) -> List[time]:
    """
    Expand a list of departure times within [start, end] inclusive (wrap past midnight if end<start).
    Cap the total count to avoid runaway lists.
    """
    dep_times: List[time] = []
    if headway_min <= 0:
        headway_min = 15
        log.warning("Non-positive headway; defaulting to 15 min.")

    s = _time_to_minutes(start)
    e = _time_to_minutes(end)
    wrap = False
    if e < s:
        e += 24 * 60
        wrap = True

    t = s
    while t <= e and len(dep_times) < cap:
        dep_times.append(_minutes_to_time(t))
        t += headway_min
    if len(dep_times) >= cap:
        log.warning("Departure expansion hit cap (%d). Consider larger headway or shorter window.", cap)
    return dep_times


# ---------- Public API ----------
def build_timetable(
    config: Dict[str, Any],
    routes: Dict[str, Any],
    models: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a basic but realistic-ish timetable:
      - Determine headway and operating window from config['service']
      - Estimate run-time from sections (speed, length) + dwell
      - Infer number of stops if not provided in routes
      - Create directional trips (A->B and B->A) with even headways
    Output schema:
      {
        "timetable": [
          {"line_id":"L1","trip":"L1_A","depart":"06:00","arrive":"06:52","headway_min":15,"stops":["S0","S1",...]},
          {"line_id":"L1","trip":"L1_B","depart":"06:00","arrive":"06:51","headway_min":15,"stops":[...]},
          ...
        ],
        "meta": {"headway_min": 15, "operating_hours": "06:00-22:00"}
      }
    """
    artifacts_root = Path(config.get("artifacts_dir") or config.get("artifacts") or "artifacts")
    runs_dir = artifacts_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    service = config.get("service", {}) or {}
    headway_min = int(service.get("headway_min", 15) or 15)
    window = service.get("operating_hours", "06:00-22:00")
    dwell_s_default = int(service.get("dwell_s_default", 35) or 35)
    start_t, end_t = _service_window(window)
    default_speed = _default_speed(config)

    lines = routes.get("lines", []) if isinstance(routes, dict) else []
    if not lines:
        log.warning("No lines provided to timetable builder; emitting empty timetable.")
        result = {"timetable": [], "meta": {"headway_min": headway_min, "operating_hours": window}}
        _safe_write_json(runs_dir / RUN_TIMETABLE_JSON, result)
        return result

    timetable_rows: List[Dict[str, Any]] = []
    for line in lines:
        try:
            line_id = line.get("id", "L?")
            # Stops (infer if absent)
            n_stops = _infer_stop_count(config, line)
            stops = line.get("stops") if isinstance(line.get("stops"), list) and len(line["stops"]) >= 2 else _infer_stop_names(line, n_stops)

            # Runtime (one-way)
            rt_secs = _estimate_runtime_seconds(line, default_speed, dwell_s_default, n_stops)
            rt_min = max(1, int(round(rt_secs / 60.0)))

            # Headway expansion for the day
            departures = _expand_departures(headway_min, start_t, end_t, cap=2000)

            # Build both directions: A (forward) and B (reverse)
            for dep in departures:
                arr_minutes = (_time_to_minutes(dep) + rt_min) % (24 * 60)
                arr_str = _minutes_to_str(arr_minutes)
                timetable_rows.append({
                    "line_id": line_id,
                    "trip": f"{line_id}_A",
                    "depart": f"{dep.hour:02d}:{dep.minute:02d}",
                    "arrive": arr_str,
                    "headway_min": headway_min,
                    "runtime_min": rt_min,
                    "dwell_s_default": dwell_s_default,
                    "stops": stops,
                })
                # reverse direction uses same runtime for now
                timetable_rows.append({
                    "line_id": line_id,
                    "trip": f"{line_id}_B",
                    "depart": f"{dep.hour:02d}:{dep.minute:02d}",
                    "arrive": arr_str,
                    "headway_min": headway_min,
                    "runtime_min": rt_min,
                    "dwell_s_default": dwell_s_default,
                    "stops": list(reversed(stops)),
                })
        except Exception as e:
            log.error("Failed to build timetable for a line: %s", e, exc_info=True)

    # Limit very large outputs to keep files manageable (report step still aggregates)
    MAX_ROWS = int(config.get("timetable_max_rows", 4000))
    if len(timetable_rows) > MAX_ROWS:
        log.warning("Timetable has %d rows; truncating to %d.", len(timetable_rows), MAX_ROWS)
        timetable_rows = timetable_rows[:MAX_ROWS]

    result = {
        "timetable": timetable_rows,
        "meta": {
            "headway_min": headway_min,
            "operating_hours": window,
            "assumed_speed_kph": default_speed,
            "dwell_s_default": dwell_s_default,
        },
    }
    try:
        _safe_write_json(runs_dir / RUN_TIMETABLE_JSON, result)
    except Exception as e:
        log.warning("Failed to write timetable plan JSON: %s", e, exc_info=True)

    log.info("Timetable built: %d rows across %d line(s).", len(timetable_rows), len(lines))
    return result
