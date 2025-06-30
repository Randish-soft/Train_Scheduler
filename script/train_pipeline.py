from model import dataload, graph, demand, scheduler, yard_opt

G           = graph.build_graph()
demand_mat  = demand.build()
timetable   = scheduler.allocate_trains(G, demand_mat)
yard_final  = yard_opt.optimise_yards(timetable, yard_candidates=G.nodes, max_yards=5)

timetable.to_csv("output/timetable.csv", index=False)
yard_final.to_json("output/railyards.json", orient="records")
