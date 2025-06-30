from dagster import asset, Output, MetadataValue, Definitions
from model import dataload, graph, demand, scheduler, yard_opt

@asset(io_manager_key="json_io_manager")
def raw_csv() -> Output[str]:
    """Path to master-data.csv"""
    path = dataload.INPUT
    return Output(
        str(path),
        metadata={"rows": sum(1 for _ in open(path)) - 1}
    )

@asset(deps=[raw_csv])
def canonical_data(raw_csv) -> Output[dict]:
    dfs = dataload.parse_master_csv(pl.Path(raw_csv))
    dataload.materialise_json(dfs)
    return Output(
        {k: v.shape for k, v in dfs.items()},
        metadata={"files_written": list(dfs)}
    )

@asset(deps=[canonical_data])
def rail_graph() -> Output["nx.Graph"]:       # type: ignore
    G = graph.build_graph()
    return Output(
        G,
        metadata={"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}
    )

@asset(deps=[canonical_data])
def demand_matrix() -> Output["pd.DataFrame"]:  # type: ignore
    D = demand.build()
    return Output(
        D,
        metadata={"shape": D.shape}
    )

@asset(deps=[rail_graph, demand_matrix])
def timetable(rail_graph, demand_matrix):
    return scheduler.allocate_trains(rail_graph, demand_matrix)

@asset(deps=[timetable])
def optimised_yards(timetable):
    return yard_opt.optimise_yards(timetable, yard_candidates=None, max_yards=5)

defs = Definitions(
    assets=[
        raw_csv,
        canonical_data,
        rail_graph,
        demand_matrix,
        timetable,
        optimised_yards,
    ],
)
