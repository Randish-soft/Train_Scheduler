"""
Rolling-stock catalogue  (extend / edit Train-types.json to override).
"""
import json, warnings, pathlib as pl
from Model.dataload import ROOT

_DEFAULT = {
    "TER_4car": {
        "top_kph": 160,
        "accel_mps2": 0.6,
        "decel_mps2": 0.7,
        "seats": 600
    },
    "TGV_8car": {
        "top_kph": 300,
        "accel_mps2": 0.4,
        "decel_mps2": 0.45,
        "seats": 450
    }
}

CATALOG_PATH = ROOT / "input" / "Train-types.json"

def get_catalog():
    if CATALOG_PATH.exists():
        with open(CATALOG_PATH) as f:
            user = json.load(f)
        return _DEFAULT | user
    warnings.warn("Train-types.json not found – using default rolling stock")
    return _DEFAULT
