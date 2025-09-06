# pipeline/flows/learn_flow.py
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Optional

from . import (
    get_logger,
    load_yaml,
    validate_config,
    ensure_dirs,
    import_optional,
)

LOG = get_logger(__name__)

# ----- Optional Prefect -----
prefect = import_optional("prefect")
prefect_flow = None
prefect_task = None
if prefect:
    try:
        from prefect import flow as _pf_flow
        from prefect import task as _pf_task
        prefect_flow = _pf_flow
        prefect_task = _pf_task
    except Exception as e:
        LOG.warning("Prefect import present but unusable (%s). Running without Prefect.", e)
        prefect = None

# ----- Optional Steps -----
steps_ingest = import_optional("pipeline.steps.ingest")
steps_features = import_optional("pipeline.steps.features")
steps_train = import_optional("pipeline.steps.train")
steps_visualize = import_optional("pipeline.steps.visualize")  # NEW

# Graceful fallbacks (minimal stubs)
if steps_ingest is None:  # pragma: no cover
    class _IngestFallback:
        @staticmethod
        def run_ingest(config: Dict[str, Any]) -> Dict[str, Any]:
            LOG.warning("Using ingest fallback. Producing no DEM and no network.")
            return {"edges": str(Path("artifacts/ingest/edges.parquet")),
                    "nodes": str(Path("artifacts/ingest/nodes.parquet")),
                    "dem": None}
    steps_ingest = _IngestFallback()

if steps_features is None:  # pragma: no cover
    class _FeaturesFallback:
        @staticmethod
        def build_features(config: Dict[str, Any], ingest_artifacts: Dict[str, Any]) -> Dict[str, Any]:
            LOG.warning("Using features fallback. Producing an empty features parquet.")
            out = Path(config.get("artifacts_dir") or "artifacts") / "features" / "edge_features.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                import pandas as pd
                pd.DataFrame([]).to_parquet(out)
            except Exception:
                out.write_text("")  # last resort
            return {"edge_features": str(out)}
    steps_features = _FeaturesFallback()

if steps_train is None:  # pragma: no cover
    class _TrainFallback:
        @staticmethod
        def train_models(config: Dict[str, Any], feat_artifacts: Dict[str, Any]) -> Dict[str, Any]:
            LOG.warning("Using train fallback. Producing placeholder model paths.")
            mdir = Path(config.get("artifacts_dir") or "artifacts") / "models"
            mdir.mkdir(parents=True, exist_ok=True)
            return {"cost": "cost.pkl", "speed": "speed.pkl", "station": "station.pkl"}
    steps_train = _TrainFallback()

# ----- JSON helpers -----
def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    try:
        import numpy as _np
        if isinstance(o, (_np.integer, _np.int_, _np.int32, _np.int64)):
            return int(o)
        if isinstance(o, (_np.floating, _np.float_, _np.float32, _np.float64)):
            return float(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except Exception:
        pass
    try:
        from shapely.geometry.base import BaseGeometry
        from shapely.geometry import mapping
        if isinstance(o, BaseGeometry):
            return mapping(o)
    except Exception:
        pass
    return str(o)


def _safe_save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    tmp.replace(path)


# ----- Timer -----
def _timeit(log: logging.Logger, label: str):
    class _Timer:
        def __enter__(self_):
            self_.t0 = time.perf_counter()
            log.info("▶ START: %s", label)
            return self_
        def __exit__(self_, exc_type, exc, tb):
            dt = time.perf_counter() - self_.t0
            if exc:
                log.error("✖ FAIL: %s (%.3fs) — %s", label, dt, exc, exc_info=True)
                return False
            log.info("✔ DONE: %s (%.3fs)", label, dt)
            return True
    return _Timer()


# ----- Steps wrappers -----
def _step_load_and_validate_config(config_path: Path, schema_path: Optional[Path]) -> Dict[str, Any]:
    with _timeit(LOG, "Load & validate config"):
        cfg = load_yaml(config_path)
        schema_default = Path("pipeline/config/schema/scenario.schema.json")
        sch = schema_path or (schema_default if schema_default.exists() else None)
        ok, err = validate_config(cfg, sch)
        if not ok:
            raise ValueError(f"Config schema validation failed: {err}")
        if not isinstance(cfg, dict):
            raise TypeError(f"Config must be a mapping, got {type(cfg)}")
        return cfg


def _step_prepare_directories(cfg: Dict[str, Any]) -> Dict[str, str]:
    with _timeit(LOG, "Ensure artifact directories"):
        artifacts_dir = Path(cfg.get("artifacts_dir") or cfg.get("artifacts") or "artifacts")
        models_dir = artifacts_dir / "models"
        geo_dir = artifacts_dir / "geo"
        reports_dir = artifacts_dir / "reports"
        runs_dir = artifacts_dir / "runs"
        ensure_dirs(artifacts_dir, models_dir, geo_dir, reports_dir, runs_dir)
        return {
            "artifacts_dir": str(artifacts_dir),
            "models_dir": str(models_dir),
            "runs_dir": str(runs_dir),
        }


def _step_ingest(cfg: Dict[str, Any]) -> Dict[str, Any]:
    with _timeit(LOG, "Ingest"):
        return steps_ingest.run_ingest(cfg)


def _step_features(cfg: Dict[str, Any], ingest_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    with _timeit(LOG, "Feature engineering"):
        return steps_features.build_features(cfg, ingest_artifacts)


def _step_visualize(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    NEW: Build learn-stage map. Runs best-effort; logs a warning if it cannot run.
    """
    with _timeit(LOG, "Visualization (learn map)"):
        if not steps_visualize or not hasattr(steps_visualize, "build_learn_map"):
            LOG.warning("Visualization step not available (module missing).")
            return {}
        try:
            return steps_visualize.build_learn_map(cfg)
        except Exception as e:
            LOG.warning("Visualization step failed: %s", e, exc_info=True)
            return {}


def _step_train(cfg: Dict[str, Any], feature_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    with _timeit(LOG, "Train models"):
        result = steps_train.train_models(cfg, feature_artifacts)
        # Normalize paths to just filenames in the models dir for brevity
        out = {}
        for k, v in result.items():
            out[k] = v
        return out


# ----- Run (sync) -----
def _run_learn_sync(config_path: Path, schema_path: Optional[Path]) -> Dict[str, Any]:
    LOG.info("Running Learn pipeline (sync mode)")
    cfg = _step_load_and_validate_config(config_path, schema_path)
    dirs = _step_prepare_directories(cfg)
    ing = _step_ingest(cfg)
    feat = _step_features(cfg, ing)
    viz = _step_visualize(cfg)        # <-- NEW
    models = _step_train(cfg, feat)

    # Summary
    artifacts_dir = Path(cfg.get("artifacts_dir") or cfg.get("artifacts") or "artifacts")
    runs_dir = artifacts_dir / "runs"
    summary = {
        "scenario": cfg.get("scenario_name"),
        "artifacts_dir": str(artifacts_dir),
        "models_dir": str(Path(dirs["models_dir"])),
        "outputs": {
            "ingest": str(runs_dir / "ingest.json"),
            "features": str(runs_dir / "features.json"),
            "train": str(runs_dir / "train.json"),
        },
    }
    if viz.get("map_html"):
        summary["outputs"]["learn_map_html"] = viz["map_html"]

    _safe_save_json(runs_dir / "summary.json", summary)
    LOG.info("Learn pipeline finished: %s", json.dumps(summary, indent=2))
    return summary


# ----- Prefect or plain entrypoints -----
if prefect and prefect_flow and prefect_task:

    @prefect_task(name="load_config", retries=1, retry_delay_seconds=2)
    def pf_load_and_validate_config(config_path: str, schema_path: Optional[str]) -> Dict[str, Any]:
        return _step_load_and_validate_config(Path(config_path), Path(schema_path) if schema_path else None)

    @prefect_task(name="prepare_dirs")
    def pf_prepare_dirs(cfg: Dict[str, Any]) -> Dict[str, str]:
        return _step_prepare_directories(cfg)

    @prefect_task(name="ingest", retries=1, retry_delay_seconds=3)
    def pf_ingest(cfg: Dict[str, Any]) -> Dict[str, Any]:
        return _step_ingest(cfg)

    @prefect_task(name="features", retries=1, retry_delay_seconds=3)
    def pf_features(cfg: Dict[str, Any], ing: Dict[str, Any]) -> Dict[str, Any]:
        return _step_features(cfg, ing)

    @prefect_task(name="visualize", retries=0)
    def pf_visualize(cfg: Dict[str, Any]) -> Dict[str, Any]:
        return _step_visualize(cfg)

    @prefect_task(name="train", retries=1, retry_delay_seconds=3)
    def pf_train(cfg: Dict[str, Any], feat: Dict[str, Any]) -> Dict[str, Any]:
        return _step_train(cfg, feat)

    @prefect_flow(name="learn_flow")
    def learn_flow(config_path: str, schema_path: Optional[str] = None) -> Dict[str, Any]:
        cfg = pf_load_and_validate_config.submit(config_path, schema_path).result()
        dirs = pf_prepare_dirs.submit(cfg).result()
        ing = pf_ingest.submit(cfg).result()
        feat = pf_features.submit(cfg, ing).result()
        viz = pf_visualize.submit(cfg).result()
        models = pf_train.submit(cfg, feat).result()

        artifacts_dir = Path(cfg.get("artifacts_dir") or cfg.get("artifacts") or "artifacts")
        runs_dir = artifacts_dir / "runs"
        summary = {
            "scenario": cfg.get("scenario_name"),
            "artifacts_dir": str(artifacts_dir),
            "models_dir": str(Path(dirs["models_dir"])),
            "outputs": {
                "ingest": str(runs_dir / "ingest.json"),
                "features": str(runs_dir / "features.json"),
                "train": str(runs_dir / "train.json"),
            },
        }
        if viz.get("map_html"):
            summary["outputs"]["learn_map_html"] = viz["map_html"]
        _safe_save_json(runs_dir / "summary.json", summary)
        LOG.info("Learn pipeline finished: %s", json.dumps(summary, indent=2))
        return summary

else:

    def learn_flow(config_path: str, schema_path: Optional[str] = None) -> Dict[str, Any]:
        return _run_learn_sync(Path(config_path), Path(schema_path) if schema_path else None)


# ----- CLI -----
def _parse_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(description="Run the Learn pipeline flow.")
    p.add_argument("--config", required=True, help="Path to scenario YAML (e.g., pipeline/config/belgium.example.yaml)")
    p.add_argument("--schema", default=None, help="Optional JSON Schema path")
    p.add_argument("--log-level", default=os.getenv("PIPELINE_LOG_LEVEL", "INFO"), help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return p.parse_args(argv)


def _set_global_log_level(level: str) -> None:
    try:
        lvl = getattr(logging, level.upper(), logging.INFO)
        logging.getLogger().setLevel(lvl)
        for h in logging.getLogger().handlers:
            h.setLevel(lvl)
        LOG.setLevel(lvl)
        LOG.info("Log level set to %s", level.upper())
    except Exception as e:
        LOG.warning("Failed to set global log level to %s: %s", level, e)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    _set_global_log_level(args.log_level)
    try:
        summary = learn_flow(config_path=args.config, schema_path=args.schema)
        print(json.dumps(summary, indent=2, default=str))
        return 0
    except KeyboardInterrupt:
        LOG.error("Interrupted by user (Ctrl+C).")
        return 130
    except SystemExit as e:
        LOG.error("SystemExit: %s", e)
        return int(getattr(e, "code", 1) or 1)
    except Exception as e:
        LOG.error("Learn flow crashed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
