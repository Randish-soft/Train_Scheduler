"""
One-shot runner (no Dagster UI needed).
"""
from pathlib import Path
from Model import dataload, graph, demand, scheduler, yard_opt

# 1) refresh JSONs
dfs = dataload.parse_master_csv()
dataload.materialise_json(dfs)

# 2) build graph + demand
G = graph.build_graph()
D = demand.build_od_matrix(G)

# 3) optimisation
timetable = scheduler.allocate_trains(G, D)
yards     = yard_opt.optimise_yards(timetable)

# 4) output
out = Path("output"); out.mkdir(exist_ok=True)
timetable.to_csv(out / "timetable.csv", index=False)
yards.to_json(out / "railyards.json", orient="records", indent=2)

print("✅ Finished – outputs in ./output/")
