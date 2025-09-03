# pipeline/steps/__init__.py
"""
Pipeline Steps
==============

Each file in this package implements an **atomic step** of the pipeline
(ingest, features, train, route, timetable, report). Steps are meant to be
*pure functions* with clear input/output contracts, returning Python dicts
that can be serialized to JSON.

Common guidelines:
- Use the shared logger (`log`) for all messages.
- Catch and re-raise exceptions after logging with context.
- Never silently swallow errors.
- Each step should be testable in isolation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ..flows import get_logger

log = get_logger("pipeline.steps")

# convenience exports
try:
    from .ingest import run_ingest
    from .features import build_features
    from .train import train_models
    from .route import build_routes
    from .timetable import build_timetable
    from .report import build_reports
except Exception as e:
    # Fail-safe: expose nothing if imports blow up
    log.warning("Pipeline steps not fully available: %s", e)

__all__ = [
    "log",
    "run_ingest",
    "build_features",
    "train_models",
    "build_routes",
    "build_timetable",
    "build_reports",
]
