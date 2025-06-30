"""
Generate an origin–destination (OD) demand matrix.

Current model = simple gravity:   demand_ij ∝ population_i × population_j
Feel free to swap this out for survey data or a four-step model later.
"""
import pandas as pd
import networkx as nx
from Model.dataload import read_json


def build_od_matrix(G: nx.Graph,
                    gravity_k: float = 1.0,
                    demand_floor: float = 0.0
                   ) -> pd.DataFrame:
    """Return a square DataFrame indexed by city names."""
    pop = read_json("Population-per-city").set_index("city")["population"]
    cities = list(pop.index)

    D = pd.DataFrame(index=cities, columns=cities, dtype=float)
    for i in cities:
        for j in cities:
            if i == j:
                D.loc[i, j] = 0.0
            else:
                demand = gravity_k * pop[i] * pop[j] / 1e9      # scale factor
                D.loc[i, j] = max(demand, demand_floor)
    return D
