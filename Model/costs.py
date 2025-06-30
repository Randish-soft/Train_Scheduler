from .dataload import load_cost_params

COSTS = load_cost_params()

def track_capex(edge_attrs, double=False, high_speed=False):
    base = COSTS["track_cost_per_km"][edge_attrs["terrain"]]
    factor = 1
    if double:
        factor *= COSTS["double_track_multiplier"]
    if high_speed:
        factor *= COSTS["high_speed_multiplier"]
    return base * factor * edge_attrs["distance_km"]      # USD million

def train_capex(train_type):          # e.g. "TGV_8car"
    return COSTS["train_cost"][train_type]

def crew_opex_per_year():
    return COSTS["annual_crew_cost_per_train"]
