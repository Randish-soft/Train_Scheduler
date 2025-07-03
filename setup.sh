#!/bin/bash

# BCPC Setup Script - Creates all files and directories in root
echo "Setting up BCPC project..."

# Create directories
mkdir -p src data outputs logs notebooks tests
mkdir -p dagster/home dagster/storage dagster/compute_logs
mkdir -p airflow/dags airflow/logs airflow/plugins airflow/config
mkdir -p .devcontainer docker

# Create src/bcpc_pipeline.py
cat > src/bcpc_pipeline.py << 'EOF'
"""
BCPC (Bring the Cities back to the People and not the Cars)
Urban Rail Transportation Planning Pipeline
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
    type: str

@dataclass
class Station:
    id: str
    name: str
    lat: float
    lon: float
    type: str
    tracks: int
    depth: float
    size: str

@dataclass
class RailLine:
    id: str
    name: str
    stations: List[Station]
    train_type: TrainType
    frequency: int

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
        return {
            'heavy_rail': TrainType('Heavy Rail', 1500, 160, 5000000, 'heavy_rail'),
            'commuter': TrainType('Commuter Rail', 1000, 120, 3000000, 'heavy_rail'),
            'metro': TrainType('Metro', 800, 80, 4000000, 'metro'),
            'light_rail': TrainType('Light Rail', 400, 70, 2000000, 'light_rail'),
            'tram': TrainType('Tram', 200, 50, 1000000, 'tram'),
            's_bahn': TrainType('S-Bahn', 600, 100, 3500000, 'heavy_rail')
        }

    def read_csv_data(self, filepath: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(filepath)
            print(f"Successfully loaded data with {len(df)} rows")
            print(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return None

    def analyze_constraints(self, df: pd.DataFrame) -> Dict:
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
        base_pattern = np.array([
            0.3, 0.2, 0.15, 0.1, 0.15, 0.4,
            0.7, 0.95, 1.0, 0.8, 0.6, 0.5,
            0.6, 0.7, 0.6, 0.7, 0.8, 0.95,
            1.0, 0.8, 0.6, 0.5, 0.4, 0.35
        ])
        tourism_factor = 1 + (city_data.tourism_index * 0.3)
        pop_factor = min(city_data.population / 100000, 2.0)
        return base_pattern * tourism_factor * pop_factor

    def suggest_transport_modes(self, city_data: CityData) -> Dict[str, bool]:
        suggestions = {
            'heavy_rail': city_data.population > 1000000,
            'metro': city_data.population > 500000,
            'light_rail': city_data.population > 200000,
            'tram': city_data.population > 100000,
            's_bahn': city_data.population > 800000 and city_data.area > 500,
            'bus': True
        }
        return suggestions

    def generate_rail_network(self, city_data: CityData, terrain_data: Optional[Dict] = None):
        try:
            G = ox.graph_from_point(
                (city_data.center_lat, city_data.center_lon),
                dist=city_data.area * 500,
                network_type='drive'
            )
        except:
            G = self._create_grid_network(city_data)
        destinations = self._identify_key_destinations(city_data)
        G = self._apply_nimby_constraints(G, terrain_data)
        routes = self._generate_optimal_routes(G, destinations, city_data)
        return routes

    def _create_grid_network(self, city_data: CityData) -> nx.Graph:
        G = nx.grid_2d_graph(10, 10)
        for node in G.nodes():
            lat = city_data.center_lat + (node[0] - 5) * 0.01
            lon = city_data.center_lon + (node[1] - 5) * 0.01
            G.nodes[node]['y'] = lat
            G.nodes[node]['x'] = lon
        return G

    def _identify_key_destinations(self, city_data: CityData) -> List[Tuple[float, float]]:
        destinations = [(city_data.center_lat, city_data.center_lon)]
        destinations.extend(city_data.job_centers)
        if city_data.tourism_index > 0.7:
            for angle in [0, 90, 180, 270]:
                lat = city_data.center_lat + 0.02 * np.cos(np.radians(angle))
                lon = city_data.center_lon + 0.02 * np.sin(np.radians(angle))
                destinations.append((lat, lon))
        return destinations

    def _apply_nimby_constraints(self, G: nx.Graph, terrain_data: Optional[Dict]) -> nx.Graph:
        for node in list(G.nodes()):
            if terrain_data and node in terrain_data.get('protected_areas', []):
                G.remove_node(node)
        return G

    def suggest_stations(self, routes: List[Dict], city_data: CityData) -> List[Station]:
        stations = []
        station_id = 0
        for route in routes:
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
        base_tracks = {
            'heavy_rail': 4,
            'metro': 2,
            'light_rail': 2,
            'tram': 1,
            's_bahn': 3
        }
        if population > 1000000:
            return base_tracks.get(transport_type, 2) + 2
        elif population > 500000:
            return base_tracks.get(transport_type, 2) + 1
        else:
            return base_tracks.get(transport_type, 2)

    def _determine_depth(self, transport_type: str, terrain_data: Optional[Dict]) -> float:
        if transport_type == 'metro':
            return -15.0
        elif transport_type == 'tram':
            return 0.0
        else:
            return -5.0 if terrain_data and terrain_data.get('congested', False) else 0.0

    def _determine_station_size(self, population: int, is_hub: bool) -> str:
        if is_hub:
            return 'hub'
        elif population > 1000000:
            return 'large'
        elif population > 500000:
            return 'medium'
        else:
            return 'small'

    def suggest_railyards(self, lines: List[RailLine], city_data: CityData) -> List[Dict]:
        railyards = []
        total_trains = sum(line.frequency * 2 for line in lines)
        num_yards = max(1, total_trains // 50)
        angles = np.linspace(0, 360, num_yards, endpoint=False)
        for i, angle in enumerate(angles):
            distance = 0.05 + (city_data.area / 10000)
            lat = city_data.center_lat + distance * np.cos(np.radians(angle))
            lon = city_data.center_lon + distance * np.sin(np.radians(angle))
            railyards.append({
                'id': f'RY_{i}',
                'name': f'{city_data.name} Railyard {i+1}',
                'lat': lat,
                'lon': lon,
                'capacity': total_trains // num_yards + 10,
                'type': 'maintenance_depot'
            })
        return railyards

    def generate_results(self, output_path: str):
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
        if not self.city_data:
            print("No city data available for visualization")
            return
        m = folium.Map(
            location=[self.city_data.center_lat, self.city_data.center_lon],
            zoom_start=11
        )
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
        for yard in self.railyards:
            folium.Marker(
                location=[yard['lat'], yard['lon']],
                popup=f"{yard['name']}<br>Capacity: {yard['capacity']} trains",
                icon=folium.Icon(color='darkred', icon='train')
            ).add_to(m)
        m.save(output_html)
        print(f"Visualization saved to {output_html}")

    def _generate_optimal_routes(self, G: nx.Graph, destinations: List[Tuple[float, float]],
                                city_data: CityData) -> List[Dict]:
        routes = []
        dest_nodes = []
        for dest in destinations:
            nearest = ox.nearest_nodes(G, dest[1], dest[0]) if hasattr(ox, 'nearest_nodes') else None
            if nearest:
                dest_nodes.append(nearest)
        transport_modes = self.suggest_transport_modes(city_data)
        if transport_modes.get('metro') or transport_modes.get('heavy_rail'):
            if len(dest_nodes) >= 2:
                route = {
                    'type': 'metro' if transport_modes.get('metro') else 'heavy_rail',
                    'path': destinations[:min(5, len(destinations))],
                    'name': f"{city_data.name} Main Line"
                }
                routes.append(route)
        if transport_modes.get('tram') or transport_modes.get('light_rail'):
            angles = np.linspace(0, 360, 8, endpoint=False)
            circular_path = []
            for angle in angles:
                lat = city_data.center_lat + 0.02 * np.cos(np.radians(angle))
                lon = city_data.center_lon + 0.02 * np.sin(np.radians(angle))
                circular_path.append((lat, lon))
            circular_path.append(circular_path[0])
            route = {
                'type': 'tram' if transport_modes.get('tram') else 'light_rail',
                'path': circular_path,
                'name': f"{city_data.name} Circle Line"
            }
            routes.append(route)
        return routes

    def run_pipeline(self, csv_path: str):
        print("Starting BCPC Pipeline...")
        df = self.read_csv_data(csv_path)
        if df is None:
            return
        self.constraints = self.analyze_constraints(df)
        print(f"Budget constraint: ${self.constraints['total_budget']:,.2f}")
        for idx, row in df.iterrows():
            print(f"\nProcessing {row.get('city', 'Unknown City')}...")
            self.city_data = CityData(
                name=row.get('city', 'Unknown'),
                state=row.get('state', 'Unknown'),
                population=row.get('population', 100000),
                area=row.get('area', 100),
                center_lat=row.get('lat', 40.7128),
                center_lon=row.get('lon', -74.0060),
                budget=row.get('budget', 1000000000),
                tourism_index=row.get('tourism_index', 0.5),
                job_centers=[]
            )
            demand = self.calculate_demand_patterns(self.city_data)
            print(f"Peak demand factor: {demand.max():.2f}")
            routes = self.generate_rail_network(self.city_data)
            print(f"Generated {len(routes)} rail routes")
            self.stations = self.suggest_stations(routes, self.city_data)
            print(f"Suggested {len(self.stations)} stations")
            self.lines = []
            for i, route in enumerate(routes):
                route_stations = [s for s in self.stations if s.type == route['type']][:10]
                if route_stations:
                    line = RailLine(
                        id=f"L{i+1}",
                        name=route['name'],
                        stations=route_stations,
                        train_type=self.train_types.get(route['type'], self.train_types['metro']),
                        frequency=int(10 * demand.max())
                    )
                    self.lines.append(line)
            self.railyards = self.suggest_railyards(self.lines, self.city_data)
            print(f"Suggested {len(self.railyards)} railyards")
            self.generate_results(f"bcpc_results_{self.city_data.name.replace(' ', '_')}.json")
            self.visualize_network(f"bcpc_network_{self.city_data.name.replace(' ', '_')}.html")
            print(f"\nCompleted processing for {self.city_data.name}")

if __name__ == "__main__":
    import pandas as pd
    sample_data = pd.DataFrame({
        'city': ['Metro City', 'Coastal Town', 'Mountain Valley'],
        'state': ['CA', 'CA', 'CO'],
        'population': [2500000, 750000, 350000],
        'area': [1500, 500, 300],
        'lat': [37.7749, 36.9741, 39.7392],
        'lon': [-122.4194, -122.0308, -104.9903],
        'budget': [5000000000, 1500000000, 800000000],
        'tourism_index': [0.8, 0.9, 0.6],
        'job_center_lat': [37.7849, 36.9841, 39.7492],
        'job_center_lon': [-122.4094, -122.0208, -104.9803],
        'job_count': [500000, 150000, 75000]
    })
    sample_data.to_csv('data/sample_city_data.csv', index=False)
    pipeline = BCPCPipeline()
    pipeline.run_pipeline('data/sample_city_data.csv')
EOF

# Create src/__init__.py
touch src/__init__.py

# Create requirements.txt
cat > requirements.txt << 'EOF'
pandas==2.0.3
numpy==1.24.3
scipy==1.10.1
scikit-learn==1.3.0
geopandas==0.13.2
folium==0.14.0
geopy==2.3.0
osmnx==1.6.0
networkx==3.1
shapely==2.0.1
dagster==1.5.0
dagster-webserver==1.5.0
dagster-postgres==0.21.0
apache-airflow==2.7.3
apache-airflow-providers-postgres==5.6.1
apache-airflow-providers-celery==3.4.1
psycopg2-binary==2.9.7
sqlalchemy==2.0.21
fastapi==0.103.1
uvicorn==0.23.2
pydantic==2.4.2
pytest==7.4.2
black==23.9.1
flake8==6.1.0
jupyter==1.0.0
jupyterlab==4.0.6
EOF

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.10-slim
RUN apt-get update && apt-get install -y \
    gcc g++ git libgeos-dev libgdal-dev libspatialindex-dev curl make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/data /app/outputs /app/logs
ENV PYTHONPATH=/app
EXPOSE 3000 8080 8000 8888
CMD ["python", "-m", "src.bcpc_pipeline"]
EOF

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_USER: bcpc_user
      POSTGRES_PASSWORD: bcpc_password
      POSTGRES_DB: bcpc_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  bcpc-app:
    build: .
    volumes:
      - ./src:/app/src
      - ./data:/app/data
      - ./outputs:/app/outputs
    environment:
      - PYTHONPATH=/app
    depends_on:
      - postgres
    ports:
      - "8000:8000"

volumes:
  postgres_data:
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.env
venv/
.vscode/
.idea/
*.egg-info/
dist/
build/
.coverage
.pytest_cache/
outputs/
logs/
*.log
.DS_Store
EOF

# Create Makefile
cat > Makefile << 'EOF'
.PHONY: help build up down run clean

help:
	@echo "Commands:"
	@echo "  make build  - Build Docker images"
	@echo "  make up     - Start services"
	@echo "  make down   - Stop services"
	@echo "  make run    - Run pipeline"
	@echo "  make clean  - Clean outputs"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

run:
	docker-compose run --rm bcpc-app python -m src.bcpc_pipeline

clean:
	rm -rf outputs/* logs/* __pycache__
EOF

# Create README.md
cat > README.md << 'EOF'
# BCPC - Bring the Cities back to the People and not the Cars

Urban rail transportation planning pipeline that prioritizes public transit.

## Quick Start

1. Build: `make build`
2. Start: `make up`
3. Run: `make run`

## Features

- Analyzes city data and constraints
- Suggests optimal public transportation modes
- Designs rail networks with NIMBY considerations
- Plans station locations and characteristics
- Generates interactive visualizations

## Data Format

CSV file with columns:
- city, state, population, area, lat, lon
- budget, tourism_index
- job_center_lat, job_center_lon, job_count
EOF

# Create sample data
mkdir -p data
cat > data/sample_city_data.csv << 'EOF'
city,state,population,area,lat,lon,budget,tourism_index,job_center_lat,job_center_lon,job_count
Metro City,CA,2500000,1500,37.7749,-122.4194,5000000000,0.8,37.7849,-122.4094,500000
Coastal Town,CA,750000,500,36.9741,-122.0308,1500000000,0.9,36.9841,-122.0208,150000
Mountain Valley,CO,350000,300,39.7392,-104.9903,800000000,0.6,39.7492,-104.9803,75000
EOF

# Make script executable
chmod +x setup-bcpc.sh

echo "Setup complete! Now run:"
echo "1. make build"
echo "2. make up"
echo "3. make run"
EOF