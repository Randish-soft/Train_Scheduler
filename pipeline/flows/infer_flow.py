# pipeline/flows/infer_flow.py
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from . import (
    get_logger,
    load_yaml,
    validate_config,
    ensure_dirs,
    import_optional,
)

# ----- Optional dependencies (degrade gracefully) -----
prefect = import_optional("prefect")  # module or None
prefect_flow = None
prefect_task = None
if prefect:
    try:
        # Prefect 2.x API
        from prefect import flow as _pf_flow
        from prefect import task as _pf_task

        prefect_flow = _pf_flow
        prefect_task = _pf_task
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning("Prefect import present but unusable (%s). Running without Prefect.", e)
        prefect = None

# Steps (optional modules; provide fallbacks)
steps_route = import_optional("pipeline.steps.route")
steps_timetable = import_optional("pipeline.steps.timetable")
steps_report = import_optional("pipeline.steps.report")

if steps_route is None:
    class _RouteFallback:  # pragma: no cover
        @staticmethod
        def build_routes(config: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
            log = get_logger(__name__)
            log.warning("Using route fallback: returning trivial line.")
            return {
                "lines": [{"id": "L0", "geometry": None, "meta": {"note": "fallback"}}],
                "notes": "route fallback"
            }
    steps_route = _RouteFallback()

if steps_timetable is None:
    class _TimetableFallback:  # pragma: no cover
        @staticmethod
        def build_timetable(config: Dict[str, Any], routes: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
            log = get_logger(__name__)
            log.warning("Using timetable fallback: returning dummy timetable.")
            return {
                "timetable": [{"line_id": "L0", "depart": "06:00", "headway_min": 20}],
                "notes": "timetable fallback"
            }
    steps_timetable = _TimetableFallback()

if steps_report is None:
    class _ReportFallback:  # pragma: no cover
        @staticmethod
        def build_reports(config: Dict[str, Any], routes: Dict[str, Any], timetable: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
            log = get_logger(__name__)
            log.warning("Using report fallback: returning minimal report pointers.")
            return {
                "reports": {"cost_book_csv": "artifacts/reports/cost_book.csv", "timetable_csv": "artifacts/reports/timetable.csv"},
                "notes": "report fallback"
            }
    steps_report = _ReportFallback()

LOG = get_logger(__name__)


# ----- Utilities -----
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


def _safe_save_json(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        LOG.debug("Saved JSON: %s", path)
    except Exception as e:
        LOG.error("Failed to save JSON to %s: %s", path, e, exc_info=True)
        raise


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        LOG.debug("Loaded JSON: %s", path)
        return data
    except FileNotFoundError:
        LOG.error("JSON file not found: %s", path)
        raise
    except PermissionError:
        LOG.error("Permission denied reading %s", path, exc_info=True)
        raise
    except json.JSONDecodeError as e:
        LOG.error("Invalid JSON in %s: %s", path, e, exc_info=True)
        raise
    except Exception as e:
        LOG.error("Unexpected error reading %s: %s", path, e, exc_info=True)
        raise


# ----- Context -----
from dataclasses import dataclass, field

@dataclass
class InferContext:
    config_path: Path
    schema_path: Optional[Path] = None
    use_models_dir: Optional[Path] = None  # allow overriding where to load models from

    config: Dict[str, Any] = field(default_factory=dict)

    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    models_dir: Path = field(init=False)
    geo_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    run_log_dir: Path = field(init=False)

    def __post_init__(self):
        self.models_dir = self.artifacts_dir / "models"
        self.geo_dir = self.artifacts_dir / "geo"
        self.reports_dir = self.artifacts_dir / "reports"
        self.run_log_dir = self.artifacts_dir / "runs"

    def apply_config_paths(self):
        try:
            ad = self.config.get("artifacts_dir") or self.config.get("artifacts") or str(self.artifacts_dir)
            self.artifacts_dir = Path(ad)
            self.models_dir = self.artifacts_dir / "models"
            self.geo_dir = self.artifacts_dir / "geo"
            self.reports_dir = self.artifacts_dir / "reports"
            self.run_log_dir = self.artifacts_dir / "runs"
            if self.use_models_dir:
                self.models_dir = Path(self.use_models_dir)
            LOG.debug("Artifacts root set to: %s | Models dir: %s", self.artifacts_dir, self.models_dir)
        except Exception as e:
            LOG.error("Failed to apply config paths: %s", e, exc_info=True)
            raise


# ----- Steps wrappers -----
def _step_load_and_validate_config(ctx: InferContext) -> Dict[str, Any]:
    with _timeit(LOG, "Load & validate config"):
        cfg = load_yaml(ctx.config_path)
        schema_default = Path("pipeline/config/schema/scenario.schema.json")
        schema_path = ctx.schema_path or (schema_default if schema_default.exists() else None)

        ok, err = validate_config(cfg, schema_path)
        if not ok:
            raise ValueError(f"Config schema validation failed: {err}")

        if not isinstance(cfg, dict):
            raise TypeError(f"Config must be a mapping, got {type(cfg)}")

        LOG.debug("Config content preview: %s", json.dumps({k: cfg.get(k) for k in list(cfg)[:10]}, indent=2))
        return cfg


def _step_prepare_directories(ctx: InferContext) -> None:
    with _timeit(LOG, "Ensure artifact directories"):
        ensure_dirs(ctx.artifacts_dir, ctx.models_dir, ctx.geo_dir, ctx.reports_dir, ctx.run_log_dir)


def _step_load_models(ctx: InferContext) -> Dict[str, Any]:
    with _timeit(LOG, "Load models"):
        # Flexible: allow models.json index, or look for standard files in models_dir
        index_path = ctx.models_dir / "models.json"
        models: Dict[str, Any] = {}
        try:
            if index_path.exists():
                models = _safe_read_json(index_path)
                LOG.info("Loaded model index: %s", index_path)
            else:
                # Heuristics for expected artifacts
                candidates = {
                    "cost": ["cost.pkl", "cost_model.pkl"],
                    "speed": ["speed.pkl", "speed_model.pkl"],
                    "station": ["station.pkl", "station_model.pkl"],
                }
                for key, names in candidates.items():
                    for n in names:
                        p = ctx.models_dir / n
                        if p.exists():
                            models[key] = str(p)
                            LOG.info("Discovered model '%s' at %s", key, p)
                            break
                    if key not in models:
                        LOG.warning("Model '%s' not found in %s (searched: %s)", key, ctx.models_dir, ", ".join(names))
            if not models:
                raise FileNotFoundError(f"No models found under {ctx.models_dir}")
        except Exception as e:
            LOG.error("Failed to load models: %s", e, exc_info=True)
            raise
        # Persist a copy into run logs for provenance
        _safe_save_json(ctx.run_log_dir / "models_loaded.json", models)
        return models


def _step_route(ctx: InferContext, models: Dict[str, Any]) -> Dict[str, Any]:
    with _timeit(LOG, "Routing / line generation"):
        try:
            result = steps_route.build_routes(ctx.config, models)
            if not isinstance(result, dict):
                raise TypeError("Route step must return a dict.")
            _safe_save_json(ctx.run_log_dir / "routes.json", result)
            return result
        except Exception as e:
            LOG.error("Route step failed: %s", e, exc_info=True)
            raise


def _step_timetable(ctx: InferContext, models: Dict[str, Any], routes: Dict[str, Any]) -> Dict[str, Any]:
    with _timeit(LOG, "Timetable synthesis"):
        try:
            result = steps_timetable.build_timetable(ctx.config, routes, models)
            if not isinstance(result, dict):
                raise TypeError("Timetable step must return a dict.")
            _safe_save_json(ctx.run_log_dir / "timetable.json", result)
            return result
        except Exception as e:
            LOG.error("Timetable step failed: %s", e, exc_info=True)
            raise


def _step_report(ctx: InferContext, models: Dict[str, Any], routes: Dict[str, Any], timetable: Dict[str, Any]) -> Dict[str, Any]:
    with _timeit(LOG, "Reporting / exports"):
        try:
            result = steps_report.build_reports(ctx.config, routes, timetable, models)
            if not isinstance(result, dict):
                raise TypeError("Report step must return a dict.")
            _safe_save_json(ctx.run_log_dir / "report.json", result)
            return result
        except Exception as e:
            LOG.error("Report step failed: %s", e, exc_info=True)
            raise


# ----- Orchestration (Prefect or plain) -----
def _run_infer_sync(ctx: InferContext) -> Dict[str, Any]:
    LOG.info("Running Inference pipeline (sync mode)")
    ctx.config = _step_load_and_validate_config(ctx)
    ctx.apply_config_paths()
    _step_prepare_directories(ctx)

    models = _step_load_models(ctx)
    routes = _step_route(ctx, models)
    timetable = _step_timetable(ctx, models, routes)
    reports = _step_report(ctx, models, routes, timetable)

    summary = {
        "scenario": ctx.config.get("scenario_name"),
        "artifacts_dir": str(ctx.artifacts_dir),
        "models_dir": str(ctx.models_dir),
        "outputs": {
            "routes": str(ctx.run_log_dir / "routes.json"),
            "timetable": str(ctx.run_log_dir / "timetable.json"),
            "report": str(ctx.run_log_dir / "report.json"),
        },
        "reports": reports.get("reports", {}),
    }
    _safe_save_json(ctx.run_log_dir / "summary_infer.json", summary)
    LOG.info("Inference pipeline finished: %s", json.dumps(summary, indent=2))
    return summary


if prefect and prefect_flow and prefect_task:
    @prefect_task(name="load_config", retries=1, retry_delay_seconds=2)
    def pf_load_and_validate_config(ctx_dict: Dict[str, Any]) -> Dict[str, Any]:
        ctx = InferContext(**ctx_dict)
        return _step_load_and_validate_config(ctx)

    @prefect_task(name="prepare_dirs", retries=0)
    def pf_prepare_directories(ctx_dict: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        ctx = InferContext(**ctx_dict)
        ctx.config = config
        ctx.apply_config_paths()
        _step_prepare_directories(ctx)
        return {
            "artifacts_dir": str(ctx.artifacts_dir),
            "models_dir": str(ctx.models_dir),
            "run_log_dir": str(ctx.run_log_dir),
        }

    @prefect_task(name="load_models", retries=1, retry_delay_seconds=3)
    def pf_load_models(ctx_dict: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        ctx = InferContext(**ctx_dict)
        ctx.config = config
        ctx.apply_config_paths()
        return _step_load_models(ctx)

    @prefect_task(name="route", retries=2, retry_delay_seconds=5)
    def pf_route(ctx_dict: Dict[str, Any], config: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
        ctx = InferContext(**ctx_dict)
        ctx.config = config
        ctx.apply_config_paths()
        return _step_route(ctx, models)

    @prefect_task(name="timetable", retries=2, retry_delay_seconds=5)
    def pf_timetable(ctx_dict: Dict[str, Any], config: Dict[str, Any], models: Dict[str, Any], routes: Dict[str, Any]) -> Dict[str, Any]:
        ctx = InferContext(**ctx_dict)
        ctx.config = config
        ctx.apply_config_paths()
        return _step_timetable(ctx, models, routes)

    @prefect_task(name="report", retries=1, retry_delay_seconds=3)
    def pf_report(ctx_dict: Dict[str, Any], config: Dict[str, Any], models: Dict[str, Any], routes: Dict[str, Any], timetable: Dict[str, Any]) -> Dict[str, Any]:
        ctx = InferContext(**ctx_dict)
        ctx.config = config
        ctx.apply_config_paths()
        return _step_report(ctx, models, routes, timetable)

    @prefect_flow(name="infer_flow")
    def infer_flow(config_path: str, schema_path: Optional[str] = None, use_models_dir: Optional[str] = None) -> Dict[str, Any]:
        LOG.info("Prefect Inference flow starting.")
        ctx_dict = {
            "config_path": Path(config_path),
            "schema_path": Path(schema_path) if schema_path else None,
            "use_models_dir": Path(use_models_dir) if use_models_dir else None,
        }

        config = pf_load_and_validate_config.submit(ctx_dict).result()
        _ = pf_prepare_directories.submit(ctx_dict, config).result()
        models = pf_load_models.submit(ctx_dict, config).result()
        routes = pf_route.submit(ctx_dict, config, models).result()
        timetable = pf_timetable.submit(ctx_dict, config, models, routes).result()
        reports = pf_report.submit(ctx_dict, config, models, routes, timetable).result()

        # Final summary
        ctx = InferContext(**ctx_dict)
        ctx.config = config
        ctx.apply_config_paths()
        summary = {
            "scenario": ctx.config.get("scenario_name"),
            "artifacts_dir": str(ctx.artifacts_dir),
            "models_dir": str(ctx.models_dir),
            "outputs": {
                "routes": str(ctx.run_log_dir / "routes.json"),
                "timetable": str(ctx.run_log_dir / "timetable.json"),
                "report": str(ctx.run_log_dir / "report.json"),
            },
            "reports": reports.get("reports", {}),
        }
        _safe_save_json(ctx.run_log_dir / "summary_infer.json", summary)
        LOG.info("Prefect Inference flow finished: %s", json.dumps(summary, indent=2))
        return summary
else:
    def infer_flow(config_path: str, schema_path: Optional[str] = None, use_models_dir: Optional[str] = None) -> Dict[str, Any]:
        ctx = InferContext(
            config_path=Path(config_path),
            schema_path=Path(schema_path) if schema_path else None,
            use_models_dir=Path(use_models_dir) if use_models_dir else None,
        )
        return _run_infer_sync(ctx)


# ----- CLI Entrypoint -----
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Inference pipeline flow.")
    p.add_argument("--config", required=True, help="Path to scenario YAML (e.g., pipeline/config/belgium.example.yaml)")
    p.add_argument("--schema", default=None, help="Optional JSON Schema path (defaults to pipeline/config/schema/scenario.schema.json if present)")
    p.add_argument("--models-dir", default=None, help="Override models directory (defaults to <artifacts_dir>/models)")
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
        summary = infer_flow(
            config_path=args.config,
            schema_path=args.schema,
            use_models_dir=args.models_dir,
        )
        print(json.dumps(summary, indent=2, default=str))
        return 0
    except KeyboardInterrupt:
        LOG.error("Interrupted by user (Ctrl+C).")
        return 130
    except SystemExit as e:
        LOG.error("SystemExit: %s", e)
        return int(getattr(e, "code", 1) or 1)
    except Exception as e:
        LOG.error("Inference flow crashed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
