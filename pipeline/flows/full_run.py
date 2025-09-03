# pipeline/flows/full_run.py
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from . import (
    get_logger,
    ensure_dirs,
    import_optional,
)

# Import subflows (these provide both Prefect-enabled and sync fallbacks)
from .learn_flow import learn_flow as _learn_flow
from .infer_flow import infer_flow as _infer_flow

LOG = get_logger(__name__)

# ----- Optional Prefect integration -----
prefect = import_optional("prefect")  # module or None
prefect_flow = None
if prefect:
    try:
        from prefect import flow as _pf_flow
        prefect_flow = _pf_flow
    except Exception as e:
        LOG.warning("Prefect detected but unusable (%s). Proceeding without Prefect.", e)
        prefect = None


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


def _derive_artifacts_dir_from_config(config_path: str) -> Path:
    """
    Fast, best-effort extraction of artifacts_dir from YAML without strict parsing,
    falling back to default 'artifacts' if not found.
    """
    try:
        from . import load_yaml  # lazy import to keep responsibilities local
        data = load_yaml(Path(config_path))
        ad = data.get("artifacts_dir") or data.get("artifacts") or "artifacts"
        p = Path(ad)
        LOG.debug("Derived artifacts_dir=%s from %s", p, config_path)
        return p
    except Exception:
        LOG.warning("Could not derive artifacts_dir from %s, using default 'artifacts'.", config_path)
        return Path("artifacts")


# ----- Core (sync) -----
def _run_full_sync(
    config_path: str,
    schema_path: Optional[str],
    models_dir_override: Optional[str],
    skip_learn: bool,
    skip_infer: bool,
    continue_on_error: bool,
) -> Dict[str, Any]:
    """
    Sequential runner that executes learn → infer. Honors skip flags and error policy.
    """
    LOG.info("Running Full pipeline (sync mode)")
    artifacts_dir = _derive_artifacts_dir_from_config(config_path)
    runs_dir = artifacts_dir / "runs"
    ensure_dirs(artifacts_dir, runs_dir)

    summary: Dict[str, Any] = {
        "config": config_path,
        "schema": schema_path,
        "artifacts_dir": str(artifacts_dir),
        "runs": {},
    }

    learn_result: Optional[Dict[str, Any]] = None
    infer_result: Optional[Dict[str, Any]] = None

    # LEARN
    if not skip_learn:
        with _timeit(LOG, "LEARN"):
            try:
                learn_result = _learn_flow(config_path=config_path, schema_path=schema_path)
                summary["runs"]["learn"] = learn_result
            except Exception as e:
                LOG.error("Learn stage failed: %s", e, exc_info=True)
                summary["runs"]["learn_error"] = str(e)
                if not continue_on_error:
                    _safe_save_json(runs_dir / "summary_full.json", summary)
                    return summary
    else:
        LOG.info("Skipping LEARN stage as requested.")
        summary["runs"]["learn_skipped"] = True

    # Determine models_dir for inference
    models_dir = None
    if models_dir_override:
        models_dir = models_dir_override
        LOG.info("Using models dir override for inference: %s", models_dir)
    elif learn_result:
        # try to read from learn summary if available
        models_dir = learn_result.get("models_dir")
        if models_dir:
            LOG.info("Using models dir from LEARN summary: %s", models_dir)

    # INFER
    if not skip_infer:
        with _timeit(LOG, "INFER"):
            try:
                infer_result = _infer_flow(
                    config_path=config_path,
                    schema_path=schema_path,
                    use_models_dir=models_dir,
                )
                summary["runs"]["infer"] = infer_result
            except Exception as e:
                LOG.error("Infer stage failed: %s", e, exc_info=True)
                summary["runs"]["infer_error"] = str(e)
                if not continue_on_error:
                    _safe_save_json(runs_dir / "summary_full.json", summary)
                    return summary
    else:
        LOG.info("Skipping INFER stage as requested.")
        summary["runs"]["infer_skipped"] = True

    _safe_save_json(runs_dir / "summary_full.json", summary)
    LOG.info("Full pipeline finished:\n%s", json.dumps(summary, indent=2))
    return summary


# ----- Prefect Orchestration -----
if prefect and prefect_flow:

    @prefect_flow(name="full_run")
    def full_run(
        config_path: str,
        schema_path: Optional[str] = None,
        models_dir_override: Optional[str] = None,
        skip_learn: bool = False,
        skip_infer: bool = False,
        continue_on_error: bool = False,
    ) -> Dict[str, Any]:
        """
        Prefect-enabled flow that orchestrates learn and infer subflows.
        """
        try:
            return _run_full_sync(
                config_path=config_path,
                schema_path=schema_path,
                models_dir_override=models_dir_override,
                skip_learn=skip_learn,
                skip_infer=skip_infer,
                continue_on_error=continue_on_error,
            )
        except Exception as e:
            LOG.error("Full run crashed: %s", e, exc_info=True)
            raise
else:

    def full_run(
        config_path: str,
        schema_path: Optional[str] = None,
        models_dir_override: Optional[str] = None,
        skip_learn: bool = False,
        skip_infer: bool = False,
        continue_on_error: bool = False,
    ) -> Dict[str, Any]:
        """
        Non-Prefect wrapper ensuring a stable API for callers.
        """
        return _run_full_sync(
            config_path=config_path,
            schema_path=schema_path,
            models_dir_override=models_dir_override,
            skip_learn=skip_learn,
            skip_infer=skip_infer,
            continue_on_error=continue_on_error,
        )


# ----- CLI Entrypoint -----
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Learn+Infer pipeline in one go.")
    p.add_argument("--config", required=True, help="Path to scenario YAML (e.g., pipeline/config/belgium.example.yaml)")
    p.add_argument("--schema", default=None, help="Optional JSON Schema path (defaults to pipeline/config/schema/scenario.schema.json if present)")
    p.add_argument("--models-dir", default=None, help="Override models directory for inference (defaults to <artifacts_dir>/models or learn output)")
    p.add_argument("--skip-learn", action="store_true", help="Skip the LEARN stage")
    p.add_argument("--skip-infer", action="store_true", help="Skip the INFER stage")
    p.add_argument("--continue-on-error", action="store_true", help="If a stage fails, continue to the next (best-effort)")
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
        summary = full_run(
            config_path=args.config,
            schema_path=args.schema,
            models_dir_override=args.models_dir,
            skip_learn=args.skip_learn,
            skip_infer=args.skip_infer,
            continue_on_error=args.continue_on_error,
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
        LOG.error("Full run crashed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
