# pipeline/flows/__init__.py
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

LOG_NAME = "pipeline.flows"


def get_logger(name: Optional[str] = None, level: Optional[str] = None) -> logging.Logger:
    """
    Create or fetch a namespaced logger with sane defaults and no duplicate handlers.
    Respects env var PIPELINE_LOG_LEVEL if provided.
    """
    logger_name = name or LOG_NAME
    logger = logging.getLogger(logger_name)

    if level is None:
        level = os.getenv("PIPELINE_LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        fmt = (
            "%(asctime)s | %(levelname)s | %(name)s | "
            "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
        )
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


log = get_logger()


def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Lazy YAML loader with defensive error handling.
    """
    try:
        import yaml  # lazy import so flows work even if yaml isn't installed yet
    except Exception as e:
        log.error("PyYAML not installed or failed to import: %s", e, exc_info=True)
        raise

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        log.info("Loaded YAML config: %s", path)
        return data
    except FileNotFoundError:
        log.error("Config file not found: %s", path)
        raise
    except PermissionError:
        log.error("Permission denied reading config: %s", path, exc_info=True)
        raise
    except yaml.YAMLError as e:
        log.error("Invalid YAML in %s: %s", path, e, exc_info=True)
        raise
    except Exception as e:
        log.error("Unexpected error loading YAML %s: %s", path, e, exc_info=True)
        raise


def validate_config(config: Dict[str, Any], schema_path: Optional[Path]) -> Tuple[bool, Optional[str]]:
    """
    Validate a scenario dict against a JSON Schema file (if provided).
    Returns (ok, error_message). If schema_path is None or missing, we log a warning and pass.
    """
    if not schema_path:
        log.warning("No schema path provided; skipping validation.")
        return True, None

    if not schema_path.exists():
        log.warning("Schema not found at %s; skipping validation.", schema_path)
        return True, None

    try:
        import jsonschema  # lazy import
    except Exception as e:
        log.warning("jsonschema not installed (%s); skipping validation.", e)
        return True, None

    try:
        with schema_path.open("r", encoding="utf-8") as f:
            schema = json.load(f)
    except Exception as e:
        msg = f"Failed to load schema {schema_path}: {e}"
        log.error(msg, exc_info=True)
        return False, msg

    try:
        jsonschema.validate(instance=config, schema=schema)
        log.info("Config validated against schema: %s", schema_path)
        return True, None
    except jsonschema.ValidationError as e:
        msg = f"Config validation error at {'/'.join(map(str, e.path)) or '<root>'}: {e.message}"
        log.error(msg)
        return False, msg
    except jsonschema.SchemaError as e:
        msg = f"Invalid JSON Schema {schema_path}: {e}"
        log.error(msg, exc_info=True)
        return False, msg
    except Exception as e:
        msg = f"Unexpected validation error: {e}"
        log.error(msg, exc_info=True)
        return False, msg


def ensure_dirs(*paths: Path) -> None:
    """
    Create directories if missing; raise on permission errors.
    """
    for p in paths:
        try:
            p.mkdir(parents=True, exist_ok=True)
            log.debug("Ensured directory: %s", p)
        except PermissionError:
            log.error("Permission denied creating directory: %s", p, exc_info=True)
            raise
        except Exception as e:
            log.error("Failed to create directory %s: %s", p, e, exc_info=True)
            raise


def safe_write_text(path: Path, content: str) -> None:
    """
    Atomically write text with basic error handling.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
        log.debug("Wrote text file: %s", path)
    except Exception as e:
        log.error("Failed writing file %s: %s", path, e, exc_info=True)
        raise


def import_optional(module_path: str, fallback: Optional[Any] = None) -> Any:
    """
    Try importing a module path (e.g., 'src.routing'); if missing, return fallback and log.
    """
    try:
        parts = module_path.split(":")
        module = __import__(parts[0], fromlist=["*"])
        obj = module
        for attr in parts[1:]:
            obj = getattr(obj, attr)
        log.debug("Imported optional module: %s", module_path)
        return obj
    except Exception as e:
        log.warning("Optional import failed for %s (%s). Using fallback.", module_path, e)
        return fallback
