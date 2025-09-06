# pipeline/cli.py
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict

from .flows import get_logger
from .flows.learn_flow import learn_flow
from .flows.infer_flow import infer_flow
from .flows.full_run import full_run

log = get_logger("pipeline.cli")


def _set_log_level(level: str) -> None:
    try:
        lvl = getattr(logging, level.upper(), logging.INFO)
        logging.getLogger().setLevel(lvl)
        for h in logging.getLogger().handlers:
            h.setLevel(lvl)
        log.setLevel(lvl)
        log.info("Log level set to %s", level.upper())
    except Exception as e:
        log.warning("Failed to set log level %s: %s", level, e)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Scheduler Pipeline CLI")
    sub = p.add_subparsers(dest="command", required=True)

    # learn
    pl = sub.add_parser("learn", help="Run training pipeline")
    pl.add_argument("--config", required=True, help="Path to scenario YAML config")
    pl.add_argument("--schema", default=None, help="Optional JSON schema path")
    pl.add_argument("--log-level", default=os.getenv("PIPELINE_LOG_LEVEL", "INFO"))

    # infer
    pi = sub.add_parser("infer", help="Run inference pipeline")
    pi.add_argument("--config", required=True, help="Path to scenario YAML config")
    pi.add_argument("--schema", default=None, help="Optional JSON schema path")
    pi.add_argument("--models-dir", default=None, help="Override models directory for inference")
    pi.add_argument("--log-level", default=os.getenv("PIPELINE_LOG_LEVEL", "INFO"))

    # full run
    pf = sub.add_parser("full", help="Run learn + infer pipeline")
    pf.add_argument("--config", required=True, help="Path to scenario YAML config")
    pf.add_argument("--schema", default=None, help="Optional JSON schema path")
    pf.add_argument("--models-dir", default=None, help="Override models directory for inference")
    pf.add_argument("--skip-learn", action="store_true", help="Skip the learn stage")
    pf.add_argument("--skip-infer", action="store_true", help="Skip the infer stage")
    pf.add_argument("--continue-on-error", action="store_true", help="Continue even if a stage fails")
    pf.add_argument("--log-level", default=os.getenv("PIPELINE_LOG_LEVEL", "INFO"))

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _set_log_level(args.log_level)

    try:
        if args.command == "learn":
            out = learn_flow(config_path=args.config, schema_path=args.schema)
        elif args.command == "infer":
            out = infer_flow(config_path=args.config, schema_path=args.schema, use_models_dir=args.models_dir)
        elif args.command == "full":
            out = full_run(
                config_path=args.config,
                schema_path=args.schema,
                models_dir_override=args.models_dir,
                skip_learn=args.skip_learn,
                skip_infer=args.skip_infer,
                continue_on_error=args.continue_on_error,
            )
        else:
            log.error("Unknown command: %s", args.command)
            return 1

        print(json.dumps(out, indent=2, default=str))
        return 0
    except KeyboardInterrupt:
        log.error("Interrupted by user (Ctrl+C).")
        return 130
    except Exception as e:
        log.error("Pipeline command failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
