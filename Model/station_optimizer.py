"""
Model/station_optimizer.py
Enhanced station positioning optimizer based on population centers
"""
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from .dataload import read_json
from .geom import haversine

class StationOptimizer:
    def __init__(self, population_weight=0.7, accessibility_weight=0.3):
        self.pop_weight = population_weight
        self.acc_weight = accessibility_weight
        self.coords = read_json("City-coords").set_index("city")
        self.pop = read_json("Population-per-city").set_index("city")
        
    def find_population_centers(self, n_stations=10):
        """Find optimal station locations using population-weighted clustering"""
        # Prepare weighted data
        cities = self.coords.index
        lats = self.coords['lat'].values
        lons = self.coords['lon'].values
        pops = self.pop.loc[cities, 'population'].values
        
        # Create weighted samples (more samples for higher population)
        weighted_coords = []
        weights = []
        
        for i, city in enumerate(cities):
            # Normalize population for sampling
            weight = pops[i] / pops.sum()
            n_samples = max(1, int(weight * 1000))  # Scale to reasonable sample size
            
            for _ in range(n_samples):
                weighted_coords.append([lats[i], lons[i]])
                weights.append(weight)
        
        weighted_coords = np.array(weighted_coords)
        
        # K-means clustering to find station centers
        kmeans = KMeans(n_clusters=n_stations, random_state=42)
        kmeans.fit(weighted_coords)
        
        station_centers = kmeans.cluster_centers_
        
        # Find nearest city to each cluster center
        station_cities = []
        for center in station_centers:
            min_dist = float('inf')
            nearest_city = None
            
            for city in cities:
                lat, lon = self.coords.loc[city, ['lat', 'lon']]
                dist = haversine(center[0], center[1], lat, lon)
                if dist < min_dist:
                    min_dist = dist
                    nearest_city = city
            
            station_cities.append({
                'city': nearest_city,
                'optimal_lat': center[0],
                'optimal_lon': center[1],
                'cluster_population': self._calculate_catchment_population(center[0], center[1])
            })
        
        return pd.DataFrame(station_cities)
    
    def _calculate_catchment_population(self, lat, lon, radius_km=50):
        """Calculate population within catchment radius of a point"""
        total_pop = 0
        
        for city in self.coords.index:
            city_lat, city_lon = self.coords.loc[city, ['lat', 'lon']]
            dist = haversine(lat, lon, city_lat, city_lon)
            
            if dist <= radius_km:
                # Distance decay function
                decay = 1 - (dist / radius_km) ** 2
                total_pop += self.pop.loc[city, 'population'] * decay
        
        return int(total_pop)
    
    def optimize_network_coverage(self, G: nx.Graph, n_stations=10):
        """Optimize stations for both population coverage and network connectivity"""
        
        def objective(station_coords):
            """Minimize negative coverage (maximize positive coverage)"""
            coords = station_coords.reshape(-1, 2)
            
            # Population coverage score
            pop_coverage = 0
            for i, city in enumerate(self.coords.index):
                city_lat, city_lon = self.coords.loc[city, ['lat', 'lon']]
                city_pop = self.pop.loc[city, 'population']
                
                # Find nearest proposed station
                min_dist = float('inf')
                for station_lat, station_lon in coords:
                    dist = haversine(city_lat, city_lon, station_lat, station_lon)
                    min_dist = min(min_dist, dist)
                
                # Score based on distance (exponential decay)
                coverage = city_pop * np.exp(-min_dist / 30)  # 30km characteristic distance
                pop_coverage += coverage
            
            # Network connectivity score (stations should be on or near tracks)
            connectivity_score = 0
            for station_lat, station_lon in coords:
                # Find distance to nearest track
                min_track_dist = float('inf')
                for u, v in G.edges():
                    u_lat, u_lon = self.coords.loc[u, ['lat', 'lon']]
                    v_lat, v_lon = self.coords.loc[v, ['lat', 'lon']]
                    
                    # Simple approximation: distance to line segment
                    track_dist = self._point_to_line_distance(
                        station_lat, station_lon, u_lat, u_lon, v_lat, v_lon
                    )
                    min_track_dist = min(min_track_dist, track_dist)
                
                connectivity_score += np.exp(-min_track_dist / 10)  # 10km characteristic distance
            
            # Combined objective (negative because we minimize)
            return -(self.pop_weight * pop_coverage + self.acc_weight * connectivity_score)
        
        # Initial guess: use k-means centers
        initial_stations = self.find_population_centers(n_stations)
        x0 = np.array([[row['optimal_lat'], row['optimal_lon']] 
                      for _, row in initial_stations.iterrows()]).flatten()
        
        # Bounds: keep stations within reasonable geographic area
        lat_min, lat_max = self.coords['lat'].min() - 1, self.coords['lat'].max() + 1
        lon_min, lon_max = self.coords['lon'].min() - 1, self.coords['lon'].max() + 1
        bounds = [(lat_min, lat_max), (lon_min, lon_max)] * n_stations
        
        # Optimize
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 1000})
        
        # Extract optimized stations
        opt_coords = result.x.reshape(-1, 2)
        stations = []
        
        for i, (lat, lon) in enumerate(opt_coords):
            # Find nearest city
            min_dist = float('inf')
            nearest_city = None
            
            for city in self.coords.index:
                city_lat, city_lon = self.coords.loc[city, ['lat', 'lon']]
                dist = haversine(lat, lon, city_lat, city_lon)
                if dist < min_dist:
                    min_dist = dist
                    nearest_city = city
            
            stations.append({
                'station_id': f'S{i+1:03d}',
                'nearest_city': nearest_city,
                'optimal_lat': lat,
                'optimal_lon': lon,
                'catchment_population': self._calculate_catchment_population(lat, lon),
                'distance_to_city': min_dist
            })
        
        return pd.DataFrame(stations)
    
    def _point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """Calculate distance from point (px,py) to line segment (x1,y1)-(x2,y2)"""
        # Vector from p1 to p2
        dx = x2 - x1
        dy = y2 - y1
        
        # Vector from p1 to point
        dpx = px - x1
        dpy = py - y1
        
        # Project point onto line
        t = max(0, min(1, (dpx * dx + dpy * dy) / (dx * dx + dy * dy)))
        
        # Nearest point on segment
        nx = x1 + t * dx
        ny = y1 + t * dy
        
        # Distance
        return haversine(px, py, nx, ny)
    
    def evaluate_station_placement(self, stations_df):
        """Evaluate quality metrics for proposed station placement"""
        metrics = {}
        
        # Total population coverage
        total_coverage = stations_df['catchment_population'].sum()
        total_population = self.pop['population'].sum()
        metrics['coverage_ratio'] = total_coverage / total_population
        
        # Average distance to nearest station for each city
        avg_distances = []
        for city in self.coords.index:
            city_lat, city_lon = self.coords.loc[city, ['lat', 'lon']]
            city_pop = self.pop.loc[city, 'population']
            
            min_dist = float('inf')
            for _, station in stations_df.iterrows():
                dist = haversine(city_lat, city_lon, 
                               station['optimal_lat'], station['optimal_lon'])
                min_dist = min(min_dist, dist)
            
            avg_distances.append(min_dist * city_pop)
        
        metrics['weighted_avg_distance'] = sum(avg_distances) / total_population
        
        # Gini coefficient for equity
        sorted_pops = sorted(stations_df['catchment_population'])
        n = len(sorted_pops)
        index = np.arange(1, n + 1)
        metrics['equity_gini'] = (2 * index - n - 1).dot(sorted_pops) / (n * sum(sorted_pops))
        
        return metrics