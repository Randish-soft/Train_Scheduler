"""
BCPC (Bring the Cities back to the People and not the Cars)
Urban Rail Transportation Planning Pipeline

This pipeline reads city data and generates optimal public transportation solutions
including rail networks, stations, and supporting infrastructure.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
import folium
from geopy.distance import geodesic
import osmnx as ox
import networkx as nx
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Data Models
@dataclass
class CityData:
    name: str
    state: str
    population: int
    area: float
    center_lat: float
    center_lon: float
    budget: float
    tourism_index: float
    job_centers: List[Tuple[float, float]]
    
@dataclass
class TrainType:
    name: str
    capacity: int
    speed: int
    cost_per_km: float
    type: str  # 'heavy_rail', 'light_rail', 'metro', 'tram'
    
@dataclass
class Station:
    id: str
    name: str
    lat: float
    lon: float
    type: str
    tracks: int
    depth: float
    size: str  # 'small', 'medium', 'large', 'hub'
    
@dataclass
class RailLine:
    id: str
    name: str
    stations: List[Station]
    train_type: TrainType
    frequency: int  # trains per hour
    
class BCPCPipeline:
    def __init__(self):
        self.city_data = None
        self.constraints = None
        self.train_types = self._initialize_train_types()
        self.network = None
        self.stations = []
        self.lines = []
        self.railyards = []
        
    def _initialize_train_types(self):
        """Initialize available train types with their characteristics"""
        return {
            'heavy_rail': TrainType('Heavy Rail', 1500, 160, 5000000, 'heavy_rail'),
            'commuter': TrainType('Commuter Rail', 1000, 120, 3000000, 'heavy_rail'),
            'metro': TrainType('Metro', 800, 80, 4000000, 'metro'),
            'light_rail': TrainType('Light Rail', 400, 70, 2000000, 'light_rail'),
            'tram': TrainType('Tram', 200, 50, 1000000, 'tram'),
            's_bahn': TrainType('S-Bahn', 600, 100, 3500000, 'heavy_rail')
        }
        
    def read_csv_data(self, filepath: str) -> pd.DataFrame:
        """Read and parse CSV file with city data"""
        try:
            df = pd.read_csv(filepath)
            print(f"Successfully loaded data with {len(df)} rows")
            print(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return None
            
    def analyze_constraints(self, df: pd.DataFrame) -> Dict:
        """Extract constraints from the CSV data"""
        constraints = {
            'total_budget': df['budget'].sum() if 'budget' in df else 0,
            'population_ranges': {
                'small': df[df['population'] < 100000],
                'medium': df[(df['population'] >= 100000) & (df['population'] < 500000)],
                'large': df[df['population'] >= 500000]
            },
            'tourism_demand': df['tourism_index'].mean() if 'tourism_index' in df else 0.5,
            'job_centers': self._identify_job_centers(df)
        }
        return constraints
        
    def _identify_job_centers(self, df: pd.DataFrame) -> List[Dict]:
        """Identify major job centers from the data"""
        job_centers = []
        if 'job_center_lat' in df and 'job_center_lon' in df:
            for idx, row in df.iterrows():
                if pd.notna(row['job_center_lat']):
                    job_centers.append({
                        'city': row['city'],
                        'coords': (row['job_center_lat'], row['job_center_lon']),
                        'jobs': row.get('job_count', 10000)
                    })
        return job_centers
        
    def calculate_demand_patterns(self, city_data: CityData) -> np.ndarray:
        """Calculate real-time demand patterns based on tourism and jobs"""
        # Create hourly demand pattern (24 hours)
        base_pattern = np.array([
            0.3, 0.2, 0.15, 0.1, 0.15, 0.4,  # 0-5am
            0.7, 0.95, 1.0, 0.8, 0.6, 0.5,   # 6-11am
            0.6, 0.7, 0.6, 0.7, 0.8, 0.95,   # 12-5pm
            1.0, 0.8, 0.6, 0.5, 0.4, 0.35    # 6-11pm
        ])
        
        # Adjust for tourism
        tourism_factor = 1 + (city_data.tourism_index * 0.3)
        
        # Adjust for population
        pop_factor = min(city_data.population / 100000, 2.0)
        
        return base_pattern * tourism_factor * pop_factor
        
    def suggest_transport_modes(self, city_data: CityData) -> Dict[str, bool]:
        """Suggest appropriate transport modes based on city characteristics"""
        suggestions = {
            'heavy_rail': city_data.population > 1000000,
            'metro': city_data.population > 500000,
            'light_rail': city_data.population > 200000,
            'tram': city_data.population > 100000,
            's_bahn': city_data.population > 800000 and city_data.area > 500,
            'bus': True  # Always suggest bus as complementary
        }
        return suggestions
        
    def generate_rail_network(self, city_data: CityData, terrain_data: Optional[Dict] = None):
        """Generate optimal rail network considering NIMBY and terrain"""
        # Get city street network
        try:
            G = ox.graph_from_point(
                (city_data.center_lat, city_data.center_lon),
                dist=city_data.area * 500,  # Convert area to meters
                network_type='drive'
            )
        except:
            # Fallback to simple grid if OSM data unavailable
            G = self._create_grid_network(city_data)
            
        # Identify key destinations
        destinations = self._identify_key_destinations(city_data)
        
        # Apply NIMBY constraints
        G = self._apply_nimby_constraints(G, terrain_data)
        
        # Generate optimal routes
        routes = self._generate_optimal_routes(G, destinations, city_data)
        
        return routes
        
    def _create_grid_network(self, city_data: CityData) -> nx.Graph:
        """Create a simple grid network as fallback"""
        G = nx.grid_2d_graph(10, 10)
        # Convert to geographic coordinates
        for node in G.nodes():
            lat = city_data.center_lat + (node[0] - 5) * 0.01
            lon = city_data.center_lon + (node[1] - 5) * 0.01
            G.nodes[node]['y'] = lat
            G.nodes[node]['x'] = lon
        return G
        
    def _identify_key_destinations(self, city_data: CityData) -> List[Tuple[float, float]]:
        """Identify key destinations for rail network"""
        destinations = [(city_data.center_lat, city_data.center_lon)]  # City center
        destinations.extend(city_data.job_centers)
        
        # Add tourist areas if high tourism
        if city_data.tourism_index > 0.7:
            # Add points around city center
            for angle in [0, 90, 180, 270]:
                lat = city_data.center_lat + 0.02 * np.cos(np.radians(angle))
                lon = city_data.center_lon + 0.02 * np.sin(np.radians(angle))
                destinations.append((lat, lon))
                
        return destinations
        
    def _apply_nimby_constraints(self, G: nx.Graph, terrain_data: Optional[Dict]) -> nx.Graph:
        """Apply NIMBY constraints to avoid residential areas and preserve city aesthetics"""
        # This is a simplified version - in reality would use zoning data
        for node in list(G.nodes()):
            if terrain_data and node in terrain_data.get('protected_areas', []):
                G.remove_node(node)
        return G
        
    def suggest_stations(self, routes: List[Dict], city_data: CityData) -> List[Station]:
        """Suggest station locations, sizes, and characteristics"""
        stations = []
        station_id = 0
        
        for route in routes:
            # Place stations every 1-3 km depending on transport type
            if route['type'] == 'metro':
                spacing = 1.0
            elif route['type'] == 'tram':
                spacing = 0.5
            else:
                spacing = 2.0
                
            points = route['path']
            current_dist = 0
            
            for i in range(len(points) - 1):
                dist = geodesic(points[i], points[i+1]).km
                if current_dist + dist >= spacing:
                    # Place station
                    station = Station(
                        id=f"ST_{station_id}",
                        name=f"{city_data.name} Station {station_id}",
                        lat=points[i][0],
                        lon=points[i][1],
                        type=route['type'],
                        tracks=self._determine_tracks(route['type'], city_data.population),
                        depth=self._determine_depth(route['type'], terrain_data=None),
                        size=self._determine_station_size(city_data.population, i == 0)
                    )
                    stations.append(station)
                    station_id += 1
                    current_dist = 0
                else:
                    current_dist += dist
                    
        return stations
        
    def _determine_tracks(self, transport_type: str, population: int) -> int:
        """Determine number of tracks based on transport type and population"""
        base_tracks = {
            'heavy_rail': 4,
            'metro': 2,
            'light_rail': 2,
            'tram': 1,
            's_bahn': 3
        }
        
        # Add tracks for larger populations
        if population > 1000000:
            return base_tracks.get(transport_type, 2) + 2
        elif population > 500000:
            return base_tracks.get(transport_type, 2) + 1
        else:
            return base_tracks.get(transport_type, 2)
            
    def _determine_depth(self, transport_type: str, terrain_data: Optional[Dict]) -> float:
        """Determine station depth (0 for surface, negative for underground)"""
        if transport_type == 'metro':
            return -15.0  # 15 meters underground
        elif transport_type == 'tram':
            return 0.0  # Surface level
        else:
            return -5.0 if terrain_data and terrain_data.get('congested', False) else 0.0
            
    def _determine_station_size(self, population: int, is_hub: bool) -> str:
        """Determine station size based on population and hub status"""
        if is_hub:
            return 'hub'
        elif population > 1000000:
            return 'large'
        elif population > 500000:
            return 'medium'
        else:
            return 'small'
            
    def suggest_railyards(self, lines: List[RailLine], city_data: CityData) -> List[Dict]:
        """Suggest railyard positions and capacity"""
        railyards = []
        
        # Calculate total trains needed
        total_trains = sum(line.frequency * 2 for line in lines)  # *2 for both directions
        
        # Determine number of railyards (1 per 50 trains)
        num_yards = max(1, total_trains // 50)
        
        # Place railyards at city periphery
        angles = np.linspace(0, 360, num_yards, endpoint=False)
        
        for i, angle in enumerate(angles):
            # Place 5-10km from city center
            distance = 0.05 + (city_data.area / 10000)  # In degrees
            lat = city_data.center_lat + distance * np.cos(np.radians(angle))
            lon = city_data.center_lon + distance * np.sin(np.radians(angle))
            
            railyards.append({
                'id': f'RY_{i}',
                'name': f'{city_data.name} Railyard {i+1}',
                'lat': lat,
                'lon': lon,
                'capacity': total_trains // num_yards + 10,  # Add buffer
                'type': 'maintenance_depot'
            })
            
        return railyards
        
    def generate_results(self, output_path: str):
        """Generate results file with all suggestions"""
        results = {
            'city': self.city_data.name if self.city_data else 'Unknown',
            'total_budget': self.constraints.get('total_budget', 0) if self.constraints else 0,
            'suggested_modes': self.suggest_transport_modes(self.city_data) if self.city_data else {},
            'stations': [
                {
                    'id': s.id,
                    'name': s.name,
                    'coordinates': [s.lat, s.lon],
                    'type': s.type,
                    'tracks': s.tracks,
                    'depth': s.depth,
                    'size': s.size
                } for s in self.stations
            ],
            'lines': [
                {
                    'id': l.id,
                    'name': l.name,
                    'train_type': l.train_type.name,
                    'frequency': l.frequency,
                    'stations': [s.id for s in l.stations]
                } for l in self.lines
            ],
            'railyards': self.railyards
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {output_path}")
        
    def visualize_network(self, output_html: str = 'bcpc_network.html'):
        """Visualize the network on OpenStreetMap using Folium"""
        if not self.city_data:
            print("No city data available for visualization")
            return
            
        # Create base map
        m = folium.Map(
            location=[self.city_data.center_lat, self.city_data.center_lon],
            zoom_start=11
        )
        
        # Add stations
        for station in self.stations:
            color = {
                'heavy_rail': 'red',
                'metro': 'blue',
                'light_rail': 'green',
                'tram': 'orange',
                's_bahn': 'purple'
            }.get(station.type, 'gray')
            
            folium.CircleMarker(
                location=[station.lat, station.lon],
                radius=8 if station.size == 'hub' else 5,
                popup=f"{station.name}<br>Type: {station.type}<br>Tracks: {station.tracks}",
                color=color,
                fill=True,
                fillColor=color
            ).add_to(m)
            
        # Add rail lines
        for line in self.lines:
            coords = [[s.lat, s.lon] for s in line.stations]
            folium.PolyLine(
                coords,
                color={
                    'Heavy Rail': 'red',
                    'Metro': 'blue',
                    'Light Rail': 'green',
                    'Tram': 'orange',
                    'S-Bahn': 'purple'
                }.get(line.train_type.name, 'gray'),
                weight=3,
                opacity=0.8,
                popup=f"{line.name}<br>Type: {line.train_type.name}<br>Frequency: {line.frequency} trains/hour"
            ).add_to(m)
            
        # Add railyards
        for yard in self.railyards:
            folium.Marker(
                location=[yard['lat'], yard['lon']],
                popup=f"{yard['name']}<br>Capacity: {yard['capacity']} trains",
                icon=folium.Icon(color='darkred', icon='train')
            ).add_to(m)
            
        # Save map
        m.save(output_html)
        print(f"Visualization saved to {output_html}")
        
    def _generate_optimal_routes(self, G: nx.Graph, destinations: List[Tuple[float, float]], 
                                city_data: CityData) -> List[Dict]:
        """Generate optimal routes between destinations"""
        routes = []
        
        # Convert destinations to nearest nodes in graph
        dest_nodes = []
        for dest in destinations:
            nearest = ox.nearest_nodes(G, dest[1], dest[0]) if hasattr(ox, 'nearest_nodes') else None
            if nearest:
                dest_nodes.append(nearest)
                
        # Generate routes between major destinations
        transport_modes = self.suggest_transport_modes(city_data)
        
        # Create hub-and-spoke network for metros/heavy rail
        if transport_modes.get('metro') or transport_modes.get('heavy_rail'):
            # Main line through city center
            if len(dest_nodes) >= 2:
                route = {
                    'type': 'metro' if transport_modes.get('metro') else 'heavy_rail',
                    'path': destinations[:min(5, len(destinations))],  # Main destinations
                    'name': f"{city_data.name} Main Line"
                }
                routes.append(route)
                
        # Add tram/light rail for local coverage
        if transport_modes.get('tram') or transport_modes.get('light_rail'):
            # Create circular route
            angles = np.linspace(0, 360, 8, endpoint=False)
            circular_path = []
            for angle in angles:
                lat = city_data.center_lat + 0.02 * np.cos(np.radians(angle))
                lon = city_data.center_lon + 0.02 * np.sin(np.radians(angle))
                circular_path.append((lat, lon))
            circular_path.append(circular_path[0])  # Close the loop
            
            route = {
                'type': 'tram' if transport_modes.get('tram') else 'light_rail',
                'path': circular_path,
                'name': f"{city_data.name} Circle Line"
            }
            routes.append(route)
            
        return routes
        
    def run_pipeline(self, csv_path: str):
        """Run the complete BCPC pipeline"""
        print("Starting BCPC Pipeline...")
        
        # Step 1: Read CSV data
        df = self.read_csv_data(csv_path)
        if df is None:
            return
            
        # Step 2: Analyze constraints
        self.constraints = self.analyze_constraints(df)
        print(f"Budget constraint: ${self.constraints['total_budget']:,.2f}")
        
        # Process each city
        for idx, row in df.iterrows():
            print(f"\nProcessing {row.get('city', 'Unknown City')}...")
            
            # Create city data object
            self.city_data = CityData(
                name=row.get('city', 'Unknown'),
                state=row.get('state', 'Unknown'),
                population=row.get('population', 100000),
                area=row.get('area', 100),
                center_lat=row.get('lat', 40.7128),
                center_lon=row.get('lon', -74.0060),
                budget=row.get('budget', 1000000000),
                tourism_index=row.get('tourism_index', 0.5),
                job_centers=[]  # Would be populated from additional data
            )
            
            # Step 3: Calculate demand patterns
            demand = self.calculate_demand_patterns(self.city_data)
            print(f"Peak demand factor: {demand.max():.2f}")
            
            # Step 4: Generate rail network
            routes = self.generate_rail_network(self.city_data)
            print(f"Generated {len(routes)} rail routes")
            
            # Step 5: Suggest stations
            self.stations = self.suggest_stations(routes, self.city_data)
            print(f"Suggested {len(self.stations)} stations")
            
            # Step 6: Create rail lines
            self.lines = []
            for i, route in enumerate(routes):
                # Get stations for this route
                route_stations = [s for s in self.stations if s.type == route['type']][:10]
                
                if route_stations:
                    line = RailLine(
                        id=f"L{i+1}",
                        name=route['name'],
                        stations=route_stations,
                        train_type=self.train_types.get(route['type'], self.train_types['metro']),
                        frequency=int(10 * demand.max())  # Trains per hour at peak
                    )
                    self.lines.append(line)
                    
            # Step 7: Suggest railyards
            self.railyards = self.suggest_railyards(self.lines, self.city_data)
            print(f"Suggested {len(self.railyards)} railyards")
            
            # Step 8: Generate results
            self.generate_results(f"bcpc_results_{self.city_data.name.replace(' ', '_')}.json")
            
            # Step 9: Visualize
            self.visualize_network(f"bcpc_network_{self.city_data.name.replace(' ', '_')}.html")
            
            print(f"\nCompleted processing for {self.city_data.name}")

# Example usage
if __name__ == "__main__":
    # Create sample CSV data for testing
    sample_data = pd.DataFrame({
        'city': ['Metro City', 'Coastal Town', 'Mountain Valley'],
        'state': ['CA', 'CA', 'CO'],
        'population': [2500000, 750000, 350000],
        'area': [1500, 500, 300],  # km²
        'lat': [37.7749, 36.9741, 39.7392],
        'lon': [-122.4194, -122.0308, -104.9903],
        'budget': [5000000000, 1500000000, 800000000],
        'tourism_index': [0.8, 0.9, 0.6],
        'job_center_lat': [37.7849, 36.9841, 39.7492],
        'job_center_lon': [-122.4094, -122.0208, -104.9803],
        'job_count': [500000, 150000, 75000]
    })
    
    # Save sample data
    sample_data.to_csv('sample_city_data.csv', index=False)
    
    # Run pipeline
    pipeline = BCPCPipeline()
    pipeline.run_pipeline('sample_city_data.csv')