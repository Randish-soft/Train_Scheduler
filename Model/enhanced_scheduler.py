"""
Model/enhanced_scheduler.py
Generate timetables based on demand patterns
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from collections import defaultdict
from .geom import travel_time_km
from .rolling_stock import get_catalog

class TimetableGenerator:
    def __init__(self):
        self.catalog = get_catalog()
        self.service_hours = {
            'start': time(5, 0),  # 5 AM
            'end': time(23, 0),   # 11 PM
            'peak_morning': (time(6, 30), time(9, 30)),
            'peak_evening': (time(17, 0), time(20, 0))
        }
    
    def is_peak_time(self, t):
        """Check if time is during peak hours"""
        morning_start, morning_end = self.service_hours['peak_morning']
        evening_start, evening_end = self.service_hours['peak_evening']
        
        return (morning_start <= t <= morning_end) or (evening_start <= t <= evening_end)
    
    def calculate_frequency(self, demand, capacity, min_headway=15):
        """Calculate train frequency based on demand"""
        # Daily demand to hourly
        hourly_demand_peak = demand * 0.15  # 15% of daily in peak hour
        hourly_demand_offpeak = demand * 0.05  # 5% in off-peak
        
        # Trains needed
        trains_peak = max(1, int(np.ceil(hourly_demand_peak / capacity)))
        trains_offpeak = max(1, int(np.ceil(hourly_demand_offpeak / capacity)))
        
        # Convert to headway (minutes between trains)
        headway_peak = max(min_headway, 60 // trains_peak)
        headway_offpeak = max(min_headway * 2, 60 // trains_offpeak)
        
        return {
            'peak_headway': headway_peak,
            'offpeak_headway': headway_offpeak,
            'peak_trains_per_hour': 60 // headway_peak,
            'offpeak_trains_per_hour': 60 // headway_offpeak
        }
    
    def select_train_type(self, route_distance, terrain_types, daily_demand):
        """Select appropriate train type for route"""
        # For high demand long coastal routes
        if daily_demand > 10000 and route_distance > 50 and 'coastal' in terrain_types:
            if 'TGV_8car' in self.catalog:
                return 'TGV_8car'
        
        # Default to regional trains
        return 'TER_4car'
    
    def generate_timetable(self, G, demand_matrix, selected_stations):
        """Generate complete timetable"""
        timetable_entries = []
        train_id_counter = 1
        
        # Get station cities
        station_cities = set(selected_stations['nearest_city'])
        
        # For each OD pair with demand
        for origin in station_cities:
            for destination in station_cities:
                if origin == destination:
                    continue
                
                if origin not in demand_matrix.index or destination not in demand_matrix.columns:
                    continue
                    
                daily_demand = demand_matrix.loc[origin, destination]
                if daily_demand < 100:  # Minimum demand threshold
                    continue
                
                try:
                    # Find shortest path
                    path = nx.shortest_path(G, origin, destination, weight='distance_km')
                    
                    # Calculate route distance and terrain
                    route_distance = sum(G[u][v]['distance_km'] 
                                       for u, v in zip(path[:-1], path[1:]))
                    terrain_types = set(G[u][v]['terrain_class'] 
                                      for u, v in zip(path[:-1], path[1:]))
                    
                    # Select train type
                    train_type = self.select_train_type(route_distance, terrain_types, daily_demand)
                    train_spec = self.catalog[train_type]
                    
                    # Calculate travel time
                    travel_minutes = travel_time_km(
                        route_distance,
                        train_spec['top_kph'],
                        train_spec['accel_mps2'],
                        train_spec['decel_mps2']
                    )
                    
                    # Add station dwell time
                    intermediate_stops = len(path) - 2
                    travel_minutes += intermediate_stops * 2  # 2 minutes per stop
                    
                    # Calculate frequency
                    freq = self.calculate_frequency(
                        daily_demand, 
                        train_spec['seats']
                    )
                    
                    # Generate departure times
                    current_time = datetime.combine(datetime.today(), self.service_hours['start'])
                    end_time = datetime.combine(datetime.today(), self.service_hours['end'])
                    
                    while current_time < end_time:
                        # Determine headway based on peak/offpeak
                        if self.is_peak_time(current_time.time()):
                            headway = freq['peak_headway']
                        else:
                            headway = freq['offpeak_headway']
                        
                        # Create timetable entry
                        departure_time = current_time.time()
                        arrival_time = (current_time + timedelta(minutes=travel_minutes)).time()
                        
                        timetable_entries.append({
                            'train_id': f"T{train_id_counter:04d}",
                            'service_name': f"{origin[:3].upper()}-{destination[:3].upper()}-{train_id_counter}",
                            'origin': origin,
                            'destination': destination,
                            'departure_time': departure_time.strftime('%H:%M'),
                            'arrival_time': arrival_time.strftime('%H:%M'),
                            'train_type': train_type,
                            'route': '->'.join(path),
                            'distance_km': round(route_distance, 1),
                            'travel_time_min': round(travel_minutes, 0),
                            'stops': intermediate_stops
                        })
                        
                        train_id_counter += 1
                        current_time += timedelta(minutes=headway)
                
                except nx.NetworkXNoPath:
                    print(f"No path found between {origin} and {destination}")
                    continue
        
        return pd.DataFrame(timetable_entries)
    
    def generate_frequency_table(self, G, demand_matrix, station_cities):
        """Generate frequency table for each segment"""
        segment_frequencies = defaultdict(lambda: {'peak': 0, 'offpeak': 0})
        
        for origin in station_cities:
            for destination in station_cities:
                if origin == destination:
                    continue
                
                if origin not in demand_matrix.index or destination not in demand_matrix.columns:
                    continue
                    
                daily_demand = demand_matrix.loc[origin, destination]
                if daily_demand < 100:
                    continue
                
                try:
                    path = nx.shortest_path(G, origin, destination, weight='distance_km')
                    
                    # Calculate frequency for this OD pair
                    train_type = 'TER_4car'  # Default
                    capacity = self.catalog[train_type]['seats']
                    freq = self.calculate_frequency(daily_demand, capacity)
                    
                    # Add frequency to each segment in path
                    for u, v in zip(path[:-1], path[1:]):
                        segment_id = G[u][v]['segment_id']
                        segment_frequencies[segment_id]['peak'] += freq['peak_trains_per_hour']
                        segment_frequencies[segment_id]['offpeak'] += freq['offpeak_trains_per_hour']
                
                except nx.NetworkXNoPath:
                    continue
        
        # Convert to DataFrame
        freq_data = []
        for segment_id, freqs in segment_frequencies.items():
            freq_data.append({
                'segment_id': segment_id,
                'peak_trains_per_hour': freqs['peak'],
                'offpeak_trains_per_hour': freqs['offpeak']
            })
        
        return pd.DataFrame(freq_data)