"""
Model/station_selector.py
Select which stations to build based on demand and budget constraints
"""
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
from .dataload import read_json
from .costs import track_capex

def select_stations_to_build(
    candidate_stations: pd.DataFrame,
    G,
    demand_matrix: pd.DataFrame,
    station_construction_cost: float = 50.0,  # Million USD per station
    min_stations: int = 5,
    max_stations: int = 15
):
    """
    Select which stations to actually build based on:
    - Population coverage
    - Demand served
    - Budget constraints
    - Network connectivity
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    
    # Decision variables
    n_candidates = len(candidate_stations)
    build_station = {}
    for i in range(n_candidates):
        build_station[i] = solver.IntVar(0, 1, f'build_station_{i}')
    
    # Get budget
    budget_df = read_json("Budget")
    total_budget = float(budget_df.iloc[0]["capex_million"])
    
    # Calculate station costs
    station_costs = station_construction_cost * np.ones(n_candidates)
    
    # Adjust costs based on expected traffic (larger stations cost more)
    for i, row in candidate_stations.iterrows():
        pop = row['catchment_population']
        if pop > 1_000_000:
            station_costs[i] *= 2.0  # Major hub
        elif pop > 500_000:
            station_costs[i] *= 1.5  # Regional hub
    
    # Objective: Maximize population coverage
    coverage_scores = candidate_stations['catchment_population'].values
    objective = solver.Sum(
        coverage_scores[i] * build_station[i] 
        for i in range(n_candidates)
    )
    solver.Maximize(objective)
    
    # Constraints
    # 1. Budget constraint
    total_cost = solver.Sum(
        station_costs[i] * build_station[i]
        for i in range(n_candidates)
    )
    solver.Add(total_cost <= total_budget * 0.4)  # Allocate 40% of budget to stations
    
    # 2. Min/max stations
    solver.Add(solver.Sum(build_station[i] for i in range(n_candidates)) >= min_stations)
    solver.Add(solver.Sum(build_station[i] for i in range(n_candidates)) <= max_stations)
    
    # 3. Coverage constraints - ensure minimum distance between stations
    min_distance_km = 30  # Minimum 30km between stations
    for i in range(n_candidates):
        for j in range(i + 1, n_candidates):
            lat1, lon1 = candidate_stations.iloc[i][['optimal_lat', 'optimal_lon']]
            lat2, lon2 = candidate_stations.iloc[j][['optimal_lat', 'optimal_lon']]
            
            from .geom import haversine
            dist = haversine(lat1, lon1, lat2, lon2)
            
            if dist < min_distance_km:
                # Can't build both stations if too close
                solver.Add(build_station[i] + build_station[j] <= 1)
    
    # Solve
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        selected_stations = []
        for i in range(n_candidates):
            if build_station[i].solution_value() > 0.5:
                station = candidate_stations.iloc[i].copy()
                station['construction_cost'] = station_costs[i]
                selected_stations.append(station)
        
        result_df = pd.DataFrame(selected_stations)
        result_df['station_type'] = result_df['catchment_population'].apply(
            lambda p: 'major_hub' if p > 1_000_000 else 
                     'regional_hub' if p > 500_000 else 'local_station'
        )
        
        return result_df
    else:
        raise ValueError("No feasible solution found for station selection")