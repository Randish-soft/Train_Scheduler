"""
script/train_pipeline.py
Fixed pipeline with adjusted parameters for Lebanon
"""
from pathlib import Path
import pandas as pd
import json
import networkx as nx
from Model import dataload
from Model.graph_builder import RailNetworkBuilder
from Model.station_optimizer import StationOptimizer
from Model.station_selector import select_stations_to_build
from Model.demand import build_od_matrix
from Model.enhanced_scheduler import TimetableGenerator
from Model.yard_opt import optimise_yards

print("🚂 Smart Train Scheduler Pipeline")
print("=" * 60)

# 1) Parse input data
print("\n📊 Step 1: Loading city data...")
dfs = dataload.parse_master_csv()
dataload.materialise_json(dfs)

# Get total population for scaling
total_pop = dfs['Population-per-city']['population'].sum()
n_cities = len(dfs['Population-per-city'])

# 2) Build rail network automatically
print("\n🗺️  Step 2: Building optimal rail network...")
network_builder = RailNetworkBuilder()
G = network_builder.build_network(max_direct_distance=120)
G = network_builder.determine_train_types(G)

print(f"   ✓ Created network with {G.number_of_nodes()} cities and {G.number_of_edges()} rail segments")

# Export generated tracks
tracks_df = network_builder.export_tracks(G)
tracks_df.to_json("data/Tracks.json", orient="records", indent=2)
print(f"   ✓ Exported track data")

# 3) Optimize station locations
print("\n📍 Step 3: Optimizing station placement...")
optimizer = StationOptimizer(population_weight=0.7, accessibility_weight=0.3)

# Adjust number of candidates based on actual cities (can't have more candidates than cities)
n_candidates = min(n_cities, max(8, n_cities // 2))

candidates = optimizer.optimize_network_coverage(G, n_stations=n_candidates)
metrics = optimizer.evaluate_station_placement(candidates)

print(f"   ✓ Generated {len(candidates)} candidate locations")
print(f"   → Population coverage: {metrics['coverage_ratio']:.1%}")

# 4) Select stations within budget
print("\n💰 Step 4: Selecting stations within budget...")

# Get budget from data
budget_info = dfs['Budget'].iloc[0] if len(dfs['Budget']) > 0 else {'capex_million': 1500}
total_budget = float(budget_info['capex_million'])

print(f"   → Total budget: ${total_budget}M")

# Adjust station costs and constraints for Lebanon's size
station_cost = 30.0  # Reduced from 40.0
min_stations = max(5, min(n_cities // 3, 8))  # At least 5, but not more than 1/3 of cities
max_stations = min(n_cities - 2, 12)  # Leave some cities without stations

print(f"   → Station cost: ${station_cost}M each")
print(f"   → Min stations: {min_stations}, Max stations: {max_stations}")

# Create demand matrix for selection
cities = list(candidates['nearest_city'].unique())
simple_demand = pd.DataFrame(index=cities, columns=cities, data=1.0)

try:
    selected_stations = select_stations_to_build(
        candidates, G, simple_demand,
        station_construction_cost=station_cost,
        min_stations=min_stations,
        max_stations=max_stations
    )
except ValueError as e:
    # If selection fails, use a simpler approach
    print(f"   ⚠️  Optimization failed: {e}")
    print("   → Using simplified selection based on population")
    
    # Simple selection: top cities by population
    selected_stations = candidates.nlargest(min_stations, 'catchment_population').copy()
    selected_stations['construction_cost'] = station_cost
    selected_stations['station_type'] = selected_stations['catchment_population'].apply(
        lambda p: 'major_hub' if p > 500000 else 
                 'regional_hub' if p > 200000 else 'local_station'
    )

print(f"   ✓ Selected {len(selected_stations)} stations")
print(f"   → Construction cost: ${selected_stations['construction_cost'].sum():.1f}M")

# Update stations data
stations_data = []
for _, station in selected_stations.iterrows():
    stations_data.append({
        'station_id': station['station_id'],
        'city': station['nearest_city'],
        'name': f"{station['nearest_city']} Station",
        'n_tracks': 4 if station['station_type'] == 'major_hub' else 2,
        'size_m2': 15000 if station['station_type'] == 'major_hub' else 8000,
        'amenities': 'full_service' if station['station_type'] == 'major_hub' else 'standard',
        'overhead_wires': 'yes'
    })

pd.DataFrame(stations_data).to_json("data/Stations.json", orient="records", indent=2)

# 5) Generate demand matrix
print("\n📈 Step 5: Calculating passenger demand...")
station_cities = set(selected_stations['nearest_city'])

# Ensure we only use cities that exist in the graph
station_cities = station_cities.intersection(set(G.nodes()))
G_stations = G.subgraph(station_cities)

# Check if graph is connected
if not nx.is_connected(G_stations):
    print("   ⚠️  Station network is not fully connected")
    components = list(nx.connected_components(G_stations))
    print(f"   → Found {len(components)} separate components")
    # Use largest component
    largest_component = max(components, key=len)
    G_stations = G_stations.subgraph(largest_component)
    station_cities = largest_component
    print(f"   → Using largest component with {len(station_cities)} stations")

demand_matrix = build_od_matrix(G_stations, gravity_k=1.0)
print(f"   ✓ Generated demand matrix for {len(station_cities)} stations")

# 6) Generate timetables
print("\n🚉 Step 6: Creating timetables...")
scheduler = TimetableGenerator()

# Import nx in enhanced_scheduler if needed
import sys
sys.modules['networkx'] = nx

timetable = scheduler.generate_timetable(G_stations, demand_matrix, selected_stations)
frequency_table = scheduler.generate_frequency_table(G_stations, demand_matrix, station_cities)

print(f"   ✓ Scheduled {len(timetable)} daily services")
print(f"   ✓ Generated frequency data for {len(frequency_table)} segments")

# Export frequency data
frequency_table.to_json("data/Frequency.json", orient="records", indent=2)

# 7) Optimize yard locations
print("\n🏗️  Step 7: Selecting railyard locations...")
if len(timetable) > 0:
    yards = optimise_yards(timetable, max_yards=3)
    print(f"   ✓ Selected {len(yards)} railyard locations")
else:
    yards = pd.DataFrame()
    print("   ⚠️  No services scheduled, skipping yard optimization")

# 8) Generate system summary
print("\n📊 Step 8: Generating system specifications...")

# Train fleet requirements
if len(timetable) > 0:
    fleet_requirements = timetable.groupby('train_type').size().to_dict()
    fleet_cost = sum(count * 15.0 for train_type, count in fleet_requirements.items())
else:
    fleet_requirements = {}
    fleet_cost = 0

# Generate cost parameters
terrain_distribution = tracks_df['terrain_class'].value_counts().to_dict()
avg_track_costs = {
    'coastal': 8.5,
    'rolling': 10.5,
    'mountain': 22.0
}

total_track_length = tracks_df['distance_km'].sum()
track_construction_cost = sum(
    tracks_df[tracks_df['terrain_class'] == terrain]['distance_km'].sum() * 
    avg_track_costs.get(terrain, 10.0)
    for terrain in terrain_distribution.keys()
)

# 9) Export all results
print("\n💾 Step 9: Saving all outputs...")
out = Path("output")
out.mkdir(exist_ok=True)

# Save main outputs
timetable.to_csv(out / "timetable.csv", index=False)
selected_stations.to_json(out / "selected_stations.json", orient="records", indent=2)
yards.to_json(out / "railyards.json", orient="records", indent=2)

# Save system summary
summary = {
    "network_statistics": {
        "total_cities": len(dfs['City-coords']),
        "stations_built": len(selected_stations),
        "rail_segments": len(tracks_df),
        "total_track_km": round(total_track_length, 1),
        "terrain_distribution": terrain_distribution
    },
    "station_breakdown": {
        "major_hubs": len(selected_stations[selected_stations['station_type'] == 'major_hub']),
        "regional_hubs": len(selected_stations[selected_stations['station_type'] == 'regional_hub']),
        "local_stations": len(selected_stations[selected_stations['station_type'] == 'local_station']),
        "population_covered": int(selected_stations['catchment_population'].sum()),
        "coverage_percentage": round(metrics['coverage_ratio'] * 100, 1)
    },
    "service_statistics": {
        "daily_services": len(timetable),
        "unique_routes": len(timetable[['origin', 'destination']].drop_duplicates()) if len(timetable) > 0 else 0,
        "fleet_requirements": fleet_requirements,
        "avg_journey_time_min": round(timetable['travel_time_min'].mean(), 1) if len(timetable) > 0 else 0
    },
    "cost_estimates_million_usd": {
        "track_construction": round(track_construction_cost, 1),
        "station_construction": round(selected_stations['construction_cost'].sum(), 1),
        "rolling_stock": round(fleet_cost, 1),
        "total_capex": round(track_construction_cost + selected_stations['construction_cost'].sum() + fleet_cost, 1),
        "budget_available": total_budget,
        "budget_utilization": round((track_construction_cost + selected_stations['construction_cost'].sum() + fleet_cost) / total_budget * 100, 1)
    }
}

with open(out / "system_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Print summary
print("\n" + "=" * 60)
print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\n📊 System Summary:")
print(f"   • Built {summary['station_breakdown']['stations_built']} stations covering {summary['station_breakdown']['coverage_percentage']}% of population")
print(f"   • Created {summary['network_statistics']['rail_segments']} rail segments totaling {summary['network_statistics']['total_track_km']}km")
print(f"   • Scheduled {summary['service_statistics']['daily_services']} daily services")
print(f"   • Total estimated cost: ${summary['cost_estimates_million_usd']['total_capex']}M")
print(f"   • Budget utilization: {summary['cost_estimates_million_usd']['budget_utilization']}%")
print(f"\n📂 All outputs saved to ./output/")
print("   • timetable.csv - Complete train schedule")
print("   • selected_stations.json - Station locations and specifications")  
print("   • system_summary.json - Full system statistics")
print("   • railyards.json - Maintenance facility locations")