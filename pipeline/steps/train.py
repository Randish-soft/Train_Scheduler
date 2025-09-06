# pipeline/steps/train.py
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import log
from ..flows import import_optional

pd = import_optional("pandas")
np = import_optional("numpy")
sk = import_optional("sklearn")
joblib = import_optional("joblib")
xgb = import_optional("xgboost")

FEATURES_DIRNAME = "features"
MODELS_DIRNAME = "models"
RUN_TRAIN_JSON = "train_summary.json"

COST_MODEL_NAME = "cost.pkl"
SPEED_MODEL_NAME = "speed.pkl"
STATION_MODEL_NAME = "station.pkl"
MODELS_INDEX = "models.json"


# -------------------- Picklable fallback models --------------------
class PicklableMeanRegressor:
    """Constant-output regressor (mean of y)."""
    def __init__(self):
        self.mu = 10_000_000.0
    def fit(self, X, y):
        import numpy as _np
        if len(y):
            self.mu = float(_np.nanmean(y))
        return self
    def predict(self, X):
        import numpy as _np
        return _np.full((len(X),), self.mu, dtype=float)


class PicklableMajorityClassifier:
    """Constant-output classifier (most frequent class)."""
    def __init__(self):
        self.mode = 0
    def fit(self, X, y):
        from collections import Counter
        cnt = Counter(y)
        self.mode = cnt.most_common(1)[0][0] if cnt else 0
        return self
    def predict(self, X):
        import numpy as _np
        return _np.array([self.mode] * len(X))
    def predict_proba(self, X):
        import numpy as _np
        return _np.ones((len(X), 1))


class PicklableLinearScore:
    """
    Tiny, picklable baseline with predict_proba([neg,pos]).
    Uses two features with fixed coefficients.
    """
    def __init__(self, coef0=0.9, coef1=0.1):
        self.coef_ = [float(coef0), float(coef1)]
    def fit(self, X, y):
        return self
    def predict_proba(self, X):
        import numpy as _np
        z = X[:, 0] * self.coef_[0] + X[:, 1] * self.coef_[1]
        z = 1 / (1 + _np.exp(-z))
        return _np.vstack([1 - z, z]).T


# -------------------- Paths --------------------
@dataclass
class Paths:
    artifacts_dir: Path
    features_path: Path
    models_dir: Path
    runs_dir: Path

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> "Paths":
        try:
            artifacts_root = Path(cfg.get("artifacts_dir") or cfg.get("artifacts") or "artifacts")
            feats = artifacts_root / FEATURES_DIRNAME / "edge_features.parquet"
            if not feats.exists():
                alt = feats.with_suffix(".csv")
                feats = alt if alt.exists() else feats
            p = Paths(
                artifacts_dir=artifacts_root,
                features_path=feats,
                models_dir=artifacts_root / MODELS_DIRNAME,
                runs_dir=artifacts_root / "runs",
            )
            p.models_dir.mkdir(parents=True, exist_ok=True)
            p.runs_dir.mkdir(parents=True, exist_ok=True)
            return p
        except Exception as e:
            log.error("Failed to construct training paths: %s", e, exc_info=True)
            raise


# -------------------- IO helpers --------------------
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


def _save_model(obj: Any, out_path: Path) -> Path:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if joblib:
            joblib.dump(obj, out_path)  # type: ignore
        else:
            with out_path.open("wb") as f:
                pickle.dump(obj, f)
        log.info("Saved model: %s", out_path)
        return out_path
    except Exception as e:
        log.error("Failed to save model %s: %s", out_path, e, exc_info=True)
        raise


def _load_features(path: Path):
    if not pd:
        raise RuntimeError("pandas is required for training but not available.")
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    try:
        if str(path).endswith(".parquet"):
            if hasattr(pd, "read_parquet"):
                return pd.read_parquet(path)
            raise RuntimeError("Parquet support not available (install pyarrow).")
        return pd.read_csv(path)
    except Exception as e:
        log.error("Failed to load features from %s: %s", path, e, exc_info=True)
        raise


# -------------------- Model pickers --------------------
def _pick_regressor() -> Any:
    if xgb:
        try:
            return xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                n_jobs=0, tree_method="hist", objective="reg:squarederror",
            )
        except Exception:
            pass
    if sk:
        try:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        except Exception:
            try:
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(random_state=42)
            except Exception:
                try:
                    from sklearn.linear_model import Ridge
                    return Ridge(alpha=1.0)  # type: ignore[arg-type]
                except Exception:
                    pass
    log.warning("Falling back to PicklableMeanRegressor baseline.")
    return PicklableMeanRegressor()


def _pick_classifier() -> Any:
    if xgb:
        try:
            return xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                n_jobs=0, tree_method="hist", objective="multi:softprob",
            )
        except Exception:
            pass
    if sk:
        try:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        except Exception:
            try:
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(max_iter=200, n_jobs=-1)  # type: ignore[arg-type]
            except Exception:
                pass
    log.warning("Falling back to PicklableMajorityClassifier baseline.")
    return PicklableMajorityClassifier()


def _safe_numeric(series, default=0.0):
    try:
        return series.astype(float).fillna(default)
    except Exception:
        return series


def _bin_speed(y: Any) -> Tuple[Any, List[int]]:
    import numpy as _np
    try:
        arr = _np.array([_np.nan if v in (None, "", "None") else float(v) for v in y])
        edges = [0, 80, 120, 160, 200, 10_000]
        classes = _np.digitize(arr, edges, right=True)
        rep = [60, 100, 140, 180, 220, 250]
        mapped = _np.array([rep[c] if (0 <= c < len(rep)) else 120 for c in classes])
        return mapped, edges
    except Exception:
        return y, [0, 200, 10_000]


# -------------------- Trainers --------------------
def _train_cost_model(df) -> Tuple[Any, Dict[str, Any]]:
    import numpy as _np
    if "capex_eur_km" in df.columns and df["capex_eur_km"].notna().any():
        y = df["capex_eur_km"]
    else:
        base = 8_000_000.0
        y = _np.full(len(df), base, dtype=float)
        struct = df.get("structure")
        slope = df.get("slope_pct")
        env = df.get("env")
        if struct is not None:
            s = struct.astype(str).str.lower()
            y[s.str.contains("tunnel", na=False)] = 45_000_000.0
            y[s.str.contains("bridge", na=False)] = 25_000_000.0
            y[s.str.contains("elevated|viaduct", na=False)] = 18_000_000.0
        if slope is not None:
            try:
                slope = slope.astype(float)
                y[slope > 2.5] *= 1.25
                y[slope > 4.0] *= 1.5
            except Exception:
                pass
        if env is not None:
            e = env.astype(str).str.lower()
            y[e == "urban"] *= 1.35

    X_cols = [
        "len_km", "curvature_rad_per_m", "slope_pct", "max_speed_kph",
        "track_count", "is_tunnel", "is_bridge", "is_elevated", "is_urban"
    ]
    for c in X_cols:
        if c not in df.columns:
            df[c] = 0
    X = df[X_cols].copy()
    for c in ["len_km", "curvature_rad_per_m", "slope_pct", "max_speed_kph", "track_count"]:
        X[c] = _safe_numeric(X[c], 0.0)
    for c in ["is_tunnel", "is_bridge", "is_elevated", "is_urban"]:
        X[c] = X[c].astype(float)

    model = _pick_regressor()
    try:
        model.fit(X.values, y.values if hasattr(y, "values") else y)
    except Exception as e:
        log.warning("Primary regressor fit failed; switching to PicklableMeanRegressor. %s", e)
        model = PicklableMeanRegressor().fit(X.values, y.values if hasattr(y, "values") else y)

    metrics = {}
    try:
        if sk:
            from sklearn.model_selection import train_test_split
            Xtr, Xte, ytr, yte = train_test_split(
                X.values, y.values if hasattr(y, "values") else y,
                test_size=0.2, random_state=42
            )
            yhat = model.predict(Xte)
            if np is not None:
                metrics["mape"] = float(np.mean(np.abs((yte - yhat) / (yte + 1e-9))))
            else:
                metrics["mape"] = None
        else:
            metrics["mape"] = None
    except Exception:
        metrics["mape"] = None

    return model, {"target_rows": int(len(X)), "features": X_cols, "metrics": metrics}


def _train_speed_model(df) -> Tuple[Any, Dict[str, Any]]:
    if "max_speed_kph" in df.columns and df["max_speed_kph"].notna().any():
        y_raw = df["max_speed_kph"]
    else:
        y_raw = df.get("max_speed_kph", None)
        if y_raw is None:
            y_raw = pd.Series([120] * len(df))
    y, edges = _bin_speed(y_raw)

    X_cols = ["curvature_rad_per_m", "slope_pct", "track_count", "is_tunnel", "is_bridge", "is_elevated", "is_urban"]
    for c in X_cols:
        if c not in df.columns:
            df[c] = 0
    X = df[X_cols].copy()
    for c in ["curvature_rad_per_m", "slope_pct", "track_count"]:
        X[c] = _safe_numeric(X[c])

    clf = _pick_classifier()
    try:
        clf.fit(X.values, y)
    except Exception as e:
        log.warning("Speed classifier fit failed; falling back to PicklableMajorityClassifier. %s", e)
        clf = PicklableMajorityClassifier().fit(X.values, y)

    return clf, {"classes_info": "representative kph classes", "train_rows": int(len(X))}


def _train_station_model(df) -> Tuple[Any, Dict[str, Any]]:
    import numpy as _np
    X_cols = ["is_urban", "len_km"]
    for c in X_cols:
        if c not in df.columns:
            df[c] = 0
    X = df[X_cols].astype(float)
    y = (df["is_urban"].astype(float) * 0.7 + (df["len_km"].astype(float) > df["len_km"].median()).astype(float) * 0.3) > 0.5
    y = y.astype(int)

    # If single-class labels, use a picklable dummy classifier
    try:
        if np is not None and len(np.unique(y.values)) < 2 and sk:
            from sklearn.dummy import DummyClassifier
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X.values, y.values)
            return dummy, {"features": X_cols, "note": "dummy most_frequent (single-class labels)"}
    except Exception:
        pass

    if sk:
        try:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=200, n_jobs=-1)  # type: ignore[arg-type]
            model.fit(X.values, y.values)
            return model, {"features": X_cols, "note": "proxy logistic model"}
        except Exception as e:
            log.warning("LogisticRegression failed; using picklable linear baseline. %s", e)

    model = PicklableLinearScore().fit(X.values, y.values)
    return model, {"features": X_cols, "note": "picklable linear scorer"}


# -------------------- Public API --------------------
def train_models(config: Dict[str, Any], feature_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train and persist:
      - cost model (€/km regressor)
      - feasible speed classifier (kph class)
      - station utility scorer (probability/score)
    Returns a dict with relative model filenames so the flow can prefix the models_dir.
    """
    paths = Paths.from_config(config)
    feats_path_str = feature_artifacts.get("features_path") or str(paths.features_path)
    feats_path = Path(feats_path_str)
    df = _load_features(feats_path)

    essentials = ["len_km", "curvature_rad_per_m", "slope_pct", "max_speed_kph", "track_count", "is_tunnel", "is_bridge", "is_elevated", "is_urban"]
    for c in essentials:
        if c not in df.columns:
            df[c] = 0

    metrics: Dict[str, Any] = {}

    try:
        cost_model, cost_info = _train_cost_model(df.copy())
        cost_path = paths.models_dir / COST_MODEL_NAME
        _save_model(cost_model, cost_path)
        metrics["cost"] = cost_info
    except Exception as e:
        log.error("Cost model training failed: %s", e, exc_info=True)
        cost_path = None

    try:
        speed_model, speed_info = _train_speed_model(df.copy())
        speed_path = paths.models_dir / SPEED_MODEL_NAME
        _save_model(speed_model, speed_path)
        metrics["speed"] = speed_info
    except Exception as e:
        log.error("Speed model training failed: %s", e, exc_info=True)
        speed_path = None

    try:
        station_model, station_info = _train_station_model(df.copy())
        station_path = paths.models_dir / STATION_MODEL_NAME
        _save_model(station_model, station_path)
        metrics["station"] = station_info
    except Exception as e:
        log.error("Station model training failed: %s", e, exc_info=True)
        station_path = None

    # Return RELATIVE filenames; the flow will prefix with models_dir
    models_index = {
        "cost": COST_MODEL_NAME if cost_path else None,
        "speed": SPEED_MODEL_NAME if speed_path else None,
        "station": STATION_MODEL_NAME if station_path else None,
    }
    try:
        _safe_write_json(paths.models_dir / MODELS_INDEX, models_index)
    except Exception as e:
        log.warning("Failed to write models index JSON: %s", e, exc_info=True)

    run_summary = {
        "features_used": str(feats_path),
        "models": models_index,
        "metrics": metrics,
    }
    try:
        _safe_write_json(paths.runs_dir / RUN_TRAIN_JSON, run_summary)
    except Exception as e:
        log.warning("Failed to write train summary JSON: %s", e, exc_info=True)

    log.info("Training complete. Models: %s", models_index)
    return {
        "models": models_index,
        "metrics": metrics,
        "notes": "Training finished (with graceful fallbacks).",
    }


__all__ = ["train_models"]
