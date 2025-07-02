"""
pipeline/assets.py
Updated pipeline with station optimization
"""
from dagster import asset, Output, MetadataValue, Definitions
import pathlib as pl
from Model import dataload, graph, demand, scheduler, yard_opt
from Model.station_optimizer import StationOptimizer
from Model.station_selector import select_stations_to_build

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

@asset(deps=["canonical_data"])
def rail_graph(canonical_data):
    return graph.build_graph()

@asset(deps=["rail_graph", "canonical_data"])
def candidate_stations(rail_graph, canonical_data):
    """Generate candidate station locations based on population"""
    optimizer = StationOptimizer(
        population_weight=0.7,
        accessibility_weight=0.3
    )
    
    # Find optimal station locations
    candidates = optimizer.optimize_network_coverage(
        rail_graph, 
        n_stations=20  # Generate more candidates than we'll build
    )
    
    # Evaluate the placement
    metrics = optimizer.evaluate_station_placement(candidates)
    
    return Output(
        candidates,
        metadata={
            "n_candidates": len(candidates),
            "coverage_ratio": f"{metrics['coverage_ratio']:.2%}",
            "avg_distance_km": f"{metrics['weighted_avg_distance']:.1f}",
            "equity_gini": f"{metrics['equity_gini']:.3f}"
        }
    )

@asset(deps=["candidate_stations", "rail_graph"])
def selected_stations(candidate_stations, rail_graph):
    """Select which stations to actually build"""
    # Create a simple demand matrix for station selection
    # (You might want to enhance this with actual OD data)
    import pandas as pd
    cities = list(candidate_stations['nearest_city'].unique())
    simple_demand = pd.DataFrame(
        index=cities, 
        columns=cities,
        data=1.0  # Uniform demand for now
    )
    
    selected = select_stations_to_build(
        candidate_stations,
        rail_graph,
        simple_demand,
        station_construction_cost=50.0,
        min_stations=8,
        max_stations=15
    )
    
    # Save to output
    selected.to_json("output/selected_stations.json", orient="records", indent=2)
    
    return Output(
        selected,
        metadata={
            "n_selected": len(selected),
            "total_cost": f"${selected['construction_cost'].sum():.1f}M",
            "major_hubs": len(selected[selected['station_type'] == 'major_hub']),
            "regional_hubs": len(selected[selected['station_type'] == 'regional_hub']),
            "total_coverage": f"{selected['catchment_population'].sum():,}"
        }
    )

@asset(deps=["rail_graph", "selected_stations"])
def demand_matrix(rail_graph, selected_stations):
    """Build demand matrix using selected stations"""
    return demand.build_od_matrix(rail_graph)

@asset(deps=["rail_graph", "demand_matrix", "selected_stations"])
def timetable(rail_graph, demand_matrix, selected_stations):
    """Create timetable considering selected stations"""
    # Modify the graph to only include selected stations
    import networkx as nx
    
    # Create subgraph with only selected station cities
    station_cities = set(selected_stations['nearest_city'])
    
    # Keep edges that connect selected stations
    edges_to_keep = [
        (u, v, d) for u, v, d in rail_graph.edges(data=True)
        if u in station_cities and v in station_cities
    ]
    
    G_stations = nx.Graph()
    G_stations.add_edges_from(edges_to_keep)
    
    return scheduler.allocate_trains(G_stations, demand_matrix)

@asset(deps=["timetable", "selected_stations"])
def optimised_yards(timetable, selected_stations):
    """Optimize yard locations considering major hubs"""
    # Prioritize yards at major hubs
    major_hubs = selected_stations[
        selected_stations['station_type'] == 'major_hub'
    ]['nearest_city'].tolist()
    
    # Pass hub information to yard optimizer
    # (You might need to modify yard_opt.optimise_yards to accept this)
    return yard_opt.optimise_yards(timetable, max_yards=4)

@asset(deps=["selected_stations"])
def station_specifications(selected_stations):
    """Generate detailed station specifications"""
    import pandas as pd
    
    specs = []
    for _, station in selected_stations.iterrows():
        spec = {
            'station_id': station['station_id'],
            'city': station['nearest_city'],
            'lat': station['optimal_lat'],
            'lon': station['optimal_lon'],
            'type': station['station_type'],
            'catchment_pop': station['catchment_population'],
            'n_platforms': 4 if station['station_type'] == 'major_hub' else 
                          2 if station['station_type'] == 'regional_hub' else 1,
            'amenities': 'full_service' if station['station_type'] == 'major_hub' else
                        'basic_service',
            'parking_spaces': int(station['catchment_population'] / 1000),
            'bus_connections': station['station_type'] != 'local_station'
        }
        specs.append(spec)
    
    specs_df = pd.DataFrame(specs)
    specs_df.to_json("output/station_specifications.json", orient="records", indent=2)
    
    return specs_df

# Define the pipeline
defs = Definitions(
    assets=[
        raw_csv,
        canonical_data,
        rail_graph,
        candidate_stations,
        selected_stations,
        demand_matrix,
        timetable,
        optimised_yards,
        station_specifications,
    ],
)