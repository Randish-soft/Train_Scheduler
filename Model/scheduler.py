"""
CAPEX-bounded build + timetable with realistic travel times.
Scheduler
"""
from ortools.linear_solver import pywraplp
import pandas as pd
import networkx as nx
import itertools, datetime as dt

from Model.dataload   import read_json
from Model.costs      import track_capex, train_capex
from Model.geom       import travel_time_km
from Model.rolling_stock import get_catalog


CAT = get_catalog()

def allocate_trains(G: nx.Graph,
                    demand_matrix: pd.DataFrame,
                    default_type: str = "TER_4car"
                   ) -> pd.DataFrame:
    solver = pywraplp.Solver.CreateSolver("SCIP")

    edges = list(G.edges(data=True))
    build_edge = {i: solver.IntVar(0, 1, f"b{i}") for i,_ in enumerate(edges)}
    buy_train  = {t: solver.IntVar(0, solver.infinity(), f"buy_{t}") for t in CAT}

    budget = float(read_json("Budget").iloc[0]["capex_million"])
    total_capex = (
        solver.Sum(track_capex(e) * build_edge[i] for i, (_,_,e) in enumerate(edges)) +
        solver.Sum(train_capex(t) * buy_train[t]  for t in CAT)
    )
    solver.Add(total_capex <= budget)
    solver.Minimize(total_capex)
    solver.Solve()

    # ----------------- Timetable ---------------------------------------
    rows, tid = [], itertools.count(1)
    for (o, d), demand in demand_matrix.stack().items():
        if demand < 1: continue
        seats = CAT[default_type]["seats"]
        trips = int((demand / seats) + 0.999)
        headway = int(18*60 / trips)         # minutes
        first = dt.time(5,0)

        # compute run-time along shortest geom path
        path = nx.shortest_path(G, o, d, weight="geom_km")
        km   = sum(G[a][b]["geom_km"] for a,b in nx.utils.pairwise(path))
        spec = CAT[default_type]
        runtime = travel_time_km(km, spec["top_kph"], spec["accel_mps2"], spec["decel_mps2"])

        for k in range(trips):
            dep_dt = (dt.datetime.combine(dt.date.today(), first) +
                      dt.timedelta(minutes=headway*k))
            arr_dt = dep_dt + dt.timedelta(minutes=runtime)
            rows.append({
                "train_id": next(tid),
                "dep_city": o,
                "arr_city": d,
                "dep_time": dep_dt.time().isoformat(timespec="minutes"),
                "arr_time": arr_dt.time().isoformat(timespec="minutes"),
                "train_type": default_type
            })
    return pd.DataFrame(rows)
