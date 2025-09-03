# pipeline/flows/learn_flow.py
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
        # Older/newer/partial installs: disable decorators
        logger = get_logger(__name__)
        logger.warning("Prefect import present but unusable (%s). Running without Prefect.", e)
        prefect = None

# Steps (optional modules; provide fallbacks)
steps_ingest = import_optional("pipeline.steps.ingest")
steps_features = import_optional("pipeline.steps.features")
steps_train = import_optional("pipeline.steps.train")
if steps_ingest is None:
    class _IngestFallback:  # pragma: no cover
        @staticmethod
        def run_ingest(config: Dict[str, Any]) -> Dict[str, Any]:
            log = get_logger(__name__)
            log.warning("Using ingest fallback: no-op output.")
            return {"ingested": True, "notes": "fallback ingest"}
    steps_ingest = _IngestFallback()

if steps_features is None:
    class _FeaturesFallback:  # pragma: no cover
        @staticmethod
        def build_features(config: Dict[str, Any], ingest_artifacts: Dict[str, Any]) -> Dict[str, Any]:
            log = get_logger(__name__)
            log.warning("Using features fallback: no-op output.")
            return {"features": True, "notes": "fallback features"}
    steps_features = _FeaturesFallback()

if steps_train is None:
    class _TrainFallback:  # pragma: no cover
        @staticmethod
        def train_models(config: Dict[str, Any], feature_artifacts: Dict[str, Any]) -> Dict[str, Any]:
            log = get_logger(__name__)
            log.warning("Using train fallback: no-op output.")
            return {
                "models": {
                    "cost": "cost.pkl",
                    "speed": "speed.pkl",
                    "station": "station.pkl",
                },
                "notes": "fallback models",
            }
    steps_train = _TrainFallback()

LOG = get_logger(__name__)


# ----- Utilities -----
def _timeit(log: logging.Logger, label: str):
    """
    Context-style timing helper:
        with _timeit(LOG, "step name"): ...
    Logs duration at INFO and catches unexpected exceptions to re-raise after logging.
    """
    class _Timer:
        def __enter__(self_):
            self_.t0 = time.perf_counter()
            log.info("▶ START: %s", label)
            return self_
        def __exit__(self_, exc_type, exc, tb):
            dt = time.perf_counter() - self_.t0
            if exc:
                log.error("✖ FAIL: %s (%.3fs) — %s", label, dt, exc, exc_info=True)
                return False  # re-raise
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


@dataclass
class LearnContext:
    config_path: Path
    schema_path: Optional[Path] = None
    config: Dict[str, Any] = field(default_factory=dict)

    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    models_dir: Path = field(init=False)
    geo_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    run_log_dir: Path = field(init=False)

    def __post_init__(self):
        # config not yet loaded; initialize dirs with defaults (may be overwritten after load)
        self.models_dir = self.artifacts_dir / "models"
        self.geo_dir = self.artifacts_dir / "geo"
        self.reports_dir = self.artifacts_dir / "reports"
        self.run_log_dir = self.artifacts_dir / "runs"

    def apply_config_paths(self):
        # After config is loaded, reflect any custom artifacts_dir
        try:
            ad = self.config.get("artifacts_dir") or self.config.get("artifacts") or str(self.artifacts_dir)
            self.artifacts_dir = Path(ad)
            self.models_dir = self.artifacts_dir / "models"
            self.geo_dir = self.artifacts_dir / "geo"
            self.reports_dir = self.artifacts_dir / "reports"
            self.run_log_dir = self.artifacts_dir / "runs"
            LOG.debug("Artifacts root set to: %s", self.artifacts_dir)
        except Exception as e:
            LOG.error("Failed to apply config paths: %s", e, exc_info=True)
            raise


# ----- Core steps wrapped with logging and error handling -----
def _step_load_and_validate_config(ctx: LearnContext) -> Dict[str, Any]:
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


def _step_prepare_directories(ctx: LearnContext) -> None:
    with _timeit(LOG, "Ensure artifact directories"):
        ensure_dirs(ctx.artifacts_dir, ctx.models_dir, ctx.geo_dir, ctx.reports_dir, ctx.run_log_dir)


def _step_ingest(ctx: LearnContext) -> Dict[str, Any]:
    with _timeit(LOG, "Ingest"):
        try:
            result = steps_ingest.run_ingest(ctx.config)  # expected shape: dict
            if not isinstance(result, dict):
                raise TypeError("Ingest step must return a dict.")
            _safe_save_json(ctx.run_log_dir / "ingest.json", result)
            return result
        except Exception as e:
            LOG.error("Ingest step failed: %s", e, exc_info=True)
            raise


def _step_features(ctx: LearnContext, ingest_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    with _timeit(LOG, "Feature engineering"):
        try:
            result = steps_features.build_features(ctx.config, ingest_artifacts)
            if not isinstance(result, dict):
                raise TypeError("Features step must return a dict.")
            _safe_save_json(ctx.run_log_dir / "features.json", result)
            return result
        except Exception as e:
            LOG.error("Features step failed: %s", e, exc_info=True)
            raise


def _step_train(ctx: LearnContext, feature_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    with _timeit(LOG, "Train models"):
        try:
            result = steps_train.train_models(ctx.config, feature_artifacts)
            if not isinstance(result, dict):
                raise TypeError("Train step must return a dict.")

            # Persist model pointers/metadata into run logs
            _safe_save_json(ctx.run_log_dir / "train.json", result)

            # If step provides model filenames, ensure they exist or warn
            models = result.get("models", {})
            for name, rel in models.items():
                p = Path(rel)
                if not p.is_absolute():
                    p = ctx.models_dir / rel
                if not p.exists():
                    LOG.warning("Model artifact not found yet: %s (reported by '%s')", p, name)
            return result
        except Exception as e:
            LOG.error("Train step failed: %s", e, exc_info=True)
            raise


# ----- Orchestration (Prefect or plain) -----
def _run_learn_sync(ctx: LearnContext) -> Dict[str, Any]:
    """
    Plain Python runner; used if Prefect isn't available.
    """
    LOG.info("Running Learn pipeline (sync mode)")
    ctx.config = _step_load_and_validate_config(ctx)
    ctx.apply_config_paths()
    _step_prepare_directories(ctx)

    ingest_art = _step_ingest(ctx)
    feat_art = _step_features(ctx, ingest_art)
    train_art = _step_train(ctx, feat_art)

    summary = {
        "scenario": ctx.config.get("scenario_name"),
        "artifacts_dir": str(ctx.artifacts_dir),
        "models_dir": str(ctx.models_dir),
        "outputs": {
            "ingest": ctx.run_log_dir / "ingest.json",
            "features": ctx.run_log_dir / "features.json",
            "train": ctx.run_log_dir / "train.json",
        },
    }
    _safe_save_json(ctx.run_log_dir / "summary.json", summary)
    LOG.info("Learn pipeline finished: %s", json.dumps(summary, indent=2, default=str))
    return summary


if prefect and prefect_flow and prefect_task:
    # Prefect-enabled tasks
    @prefect_task(name="load_config", retries=1, retry_delay_seconds=2)
    def pf_load_and_validate_config(ctx_dict: Dict[str, Any]) -> Dict[str, Any]:
        ctx = LearnContext(**ctx_dict)
        cfg = _step_load_and_validate_config(ctx)
        return cfg

    @prefect_task(name="prepare_dirs", retries=0)
    def pf_prepare_directories(ctx_dict: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        ctx = LearnContext(**ctx_dict)
        ctx.config = config
        ctx.apply_config_paths()
        _step_prepare_directories(ctx)
        return {
            "artifacts_dir": str(ctx.artifacts_dir),
            "models_dir": str(ctx.models_dir),
            "run_log_dir": str(ctx.run_log_dir),
        }

    @prefect_task(name="ingest", retries=2, retry_delay_seconds=5)
    def pf_ingest(ctx_dict: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        ctx = LearnContext(**ctx_dict)
        ctx.config = config
        ctx.apply_config_paths()
        return _step_ingest(ctx)

    @prefect_task(name="features", retries=2, retry_delay_seconds=5)
    def pf_features(ctx_dict: Dict[str, Any], config: Dict[str, Any], ingest_artifacts: Dict[str, Any]) -> Dict[str, Any]:
        ctx = LearnContext(**ctx_dict)
        ctx.config = config
        ctx.apply_config_paths()
        return _step_features(ctx, ingest_artifacts)

    @prefect_task(name="train", retries=1, retry_delay_seconds=5)
    def pf_train(ctx_dict: Dict[str, Any], config: Dict[str, Any], feature_artifacts: Dict[str, Any]) -> Dict[str, Any]:
        ctx = LearnContext(**ctx_dict)
        ctx.config = config
        ctx.apply_config_paths()
        return _step_train(ctx, feature_artifacts)

    @prefect_flow(name="learn_flow")
    def learn_flow(config_path: str, schema_path: Optional[str] = None) -> Dict[str, Any]:
        LOG.info("Prefect Learn flow starting.")
        ctx_dict = {
            "config_path": Path(config_path),
            "schema_path": Path(schema_path) if schema_path else None,
        }

        config = pf_load_and_validate_config.submit(ctx_dict).result()
        dirs_info = pf_prepare_directories.submit(ctx_dict, config).result()
        ingest_art = pf_ingest.submit(ctx_dict, config).result()
        feat_art = pf_features.submit(ctx_dict, config, ingest_art).result()
        train_art = pf_train.submit(ctx_dict, config, feat_art).result()

        # Write summary at the end
        ctx = LearnContext(**ctx_dict)
        ctx.config = config
        ctx.apply_config_paths()
        summary = {
            "scenario": ctx.config.get("scenario_name"),
            "artifacts_dir": str(ctx.artifacts_dir),
            "models_dir": str(ctx.models_dir),
            "outputs": {
                "ingest": str(ctx.run_log_dir / "ingest.json"),
                "features": str(ctx.run_log_dir / "features.json"),
                "train": str(ctx.run_log_dir / "train.json"),
            },
        }
        _safe_save_json(ctx.run_log_dir / "summary.json", summary)
        LOG.info("Prefect Learn flow finished: %s", json.dumps(summary, indent=2))
        return summary
else:
    def learn_flow(config_path: str, schema_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Compatibility wrapper so callers can always import and call learn_flow(...).
        """
        ctx = LearnContext(config_path=Path(config_path), schema_path=Path(schema_path) if schema_path else None)
        return _run_learn_sync(ctx)


# ----- CLI Entrypoint -----
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Learn pipeline flow.")
    p.add_argument("--config", required=True, help="Path to scenario YAML (e.g., pipeline/config/belgium.example.yaml)")
    p.add_argument("--schema", default=None, help="Optional JSON Schema path (defaults to pipeline/config/schema/scenario.schema.json if present)")
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
        # Pretty print essential outputs for terminal users
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
