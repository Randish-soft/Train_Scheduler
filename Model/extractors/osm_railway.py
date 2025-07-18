# File: Model/extractors/osm_railway.py
import requests
import json
from typing import Dict, List, Tuple
import time

class OSMRailwayExtractor:
    def __init__(self):
        self.overpass_url = "http://overpass-api.de/api/interpreter"
        
    def extract_country_railways(self, country: str) -> Dict:
        """Extract all railway infrastructure for a country"""
        query = f"""
        [out:json][timeout:300];
        area["ISO3166-1"="{country.upper()}"][admin_level=2];
        (
          way["railway"~"^(rail|light_rail|subway|tram)$"](area);
          node["railway"="station"](area);
          node["railway"="halt"](area);
          way["railway"="platform"](area);
          node["railway"="junction"](area);
        );
        out geom;
        """
        
        response = requests.post(self.overpass_url, data={'data': query})
        return response.json()
    
    def extract_train_stations(self, country: str) -> List[Dict]:
        """Extract train stations with details"""
        query = f"""
        [out:json][timeout:180];
        area["ISO3166-1"="{country.upper()}"][admin_level=2];
        (
          node["railway"="station"]["public_transport"="station"](area);
          node["railway"="station"]["train"="yes"](area);
        );
        out;
        """
        
        response = requests.post(self.overpass_url, data={'data': query})
        data = response.json()
        
        stations = []
        for element in data['elements']:
            if element['type'] == 'node':
                station = {
                    'id': element['id'],
                    'lat': element['lat'],
                    'lon': element['lon'],
                    'name': element.get('tags', {}).get('name', 'Unknown'),
                    'operator': element.get('tags', {}).get('operator', ''),
                    'platforms': element.get('tags', {}).get('platforms', ''),
                    'level': element.get('tags', {}).get('railway:station_category', 'regional')
                }
                stations.append(station)
        
        return stations
    
    def extract_railway_tracks(self, country: str) -> List[Dict]:
        """Extract railway tracks with technical details"""
        query = f"""
        [out:json][timeout:300];
        area["ISO3166-1"="{country.upper()}"][admin_level=2];
        way["railway"="rail"]["usage"~"main|branch"](area);
        out geom;
        """
        
        response = requests.post(self.overpass_url, data={'data': query})
        data = response.json()
        
        tracks = []
        for element in data['elements']:
            if element['type'] == 'way':
                track = {
                    'id': element['id'],
                    'geometry': element['geometry'],
                    'maxspeed': element.get('tags', {}).get('maxspeed', ''),
                    'electrified': element.get('tags', {}).get('electrified', 'no'),
                    'gauge': element.get('tags', {}).get('gauge', '1435'),
                    'usage': element.get('tags', {}).get('usage', 'main'),
                    'service': element.get('tags', {}).get('service', 'passenger')
                }
                tracks.append(track)
                
        return tracks


# File: Model/extractors/terrain_analysis.py
import numpy as np
import requests
from typing import Tuple, List
import io
from PIL import Image

class TerrainAnalyzer:
    def __init__(self):
        self.srtm_base_url = "https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1"
        
    def get_elevation_data(self, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Get elevation data for bounding box (west, south, east, north)"""
        west, south, east, north = bbox
        
        # Simple elevation model - in production you'd use proper SRTM tiles
        lat_points = np.linspace(south, north, 100)
        lon_points = np.linspace(west, east, 100)
        
        # Mock elevation data with realistic patterns
        elevation = np.zeros((100, 100))
        for i, lat in enumerate(lat_points):
            for j, lon in enumerate(lon_points):
                # Simulate elevation based on coordinate patterns
                elevation[i, j] = abs(lat * 10) + abs(lon * 5) + np.random.normal(0, 50)
        
        return elevation
    
    def calculate_slope(self, elevation: np.ndarray) -> np.ndarray:
        """Calculate slope gradients from elevation data"""
        dy, dx = np.gradient(elevation)
        slope = np.sqrt(dx**2 + dy**2)
        return np.degrees(np.arctan(slope))
    
    def identify_obstacles(self, bbox: Tuple[float, float, float, float]) -> List[Dict]:
        """Identify terrain obstacles (rivers, mountains, protected areas)"""
        # This would integrate with OSM for water bodies, protected areas
        query = f"""
        [out:json][timeout:120];
        (
          way["natural"="water"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
          way["waterway"="river"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
          way["leisure"="nature_reserve"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
          way["boundary"="protected_area"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        );
        out geom;
        """
        
        response = requests.post("http://overpass-api.de/api/interpreter", 
                               data={'data': query})
        return response.json()
    
    def get_terrain_cost_matrix(self, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Generate terrain cost matrix for pathfinding"""
        elevation = self.get_elevation_data(bbox)
        slope = self.calculate_slope(elevation)
        
        # Cost increases exponentially with slope
        cost_matrix = 1.0 + (slope / 10.0) ** 2
        
        # Add penalties for extreme elevations
        cost_matrix += np.where(elevation > 1000, elevation / 1000, 0)
        
        return cost_matrix


# File: Model/extractors/network_parser.py
import networkx as nx
from typing import Dict, List, Tuple
import math

class RailwayNetworkParser:
    def __init__(self):
        self.graph = nx.Graph()
        
    def parse_osm_to_network(self, osm_data: Dict) -> nx.Graph:
        """Convert OSM railway data to network graph"""
        stations = {}
        tracks = []
        
        # Extract stations as nodes
        for element in osm_data['elements']:
            if element['type'] == 'node' and 'railway' in element.get('tags', {}):
                if element['tags']['railway'] in ['station', 'halt']:
                    stations[element['id']] = {
                        'lat': element['lat'],
                        'lon': element['lon'],
                        'name': element.get('tags', {}).get('name', 'Unknown'),
                        'type': element['tags']['railway']
                    }
            
            elif element['type'] == 'way' and 'railway' in element.get('tags', {}):
                if element['tags']['railway'] == 'rail':
                    tracks.append({
                        'id': element['id'],
                        'nodes': element['nodes'],
                        'geometry': element.get('geometry', []),
                        'properties': element.get('tags', {})
                    })
        
        # Build graph
        for station_id, station_data in stations.items():
            self.graph.add_node(station_id, **station_data)
        
        # Add edges based on track connections
        for track in tracks:
            if len(track['geometry']) > 1:
                self._connect_track_segments(track, stations)
        
        return self.graph
    
    def _connect_track_segments(self, track: Dict, stations: Dict):
        """Connect stations along track segments"""
        geometry = track['geometry']
        
        for i in range(len(geometry) - 1):
            start_point = geometry[i]
            end_point = geometry[i + 1]
            
            # Find nearest stations to track endpoints
            start_station = self._find_nearest_station(start_point, stations)
            end_station = self._find_nearest_station(end_point, stations)
            
            if start_station and end_station and start_station != end_station:
                distance = self._haversine_distance(
                    stations[start_station]['lat'], stations[start_station]['lon'],
                    stations[end_station]['lat'], stations[end_station]['lon']
                )
                
                self.graph.add_edge(start_station, end_station, 
                                  weight=distance,
                                  track_id=track['id'],
                                  properties=track['properties'])
    
    def _find_nearest_station(self, point: Dict, stations: Dict, threshold_km: float = 1.0) -> int:
        """Find nearest station within threshold"""
        min_distance = float('inf')
        nearest_station = None
        
        for station_id, station in stations.items():
            distance = self._haversine_distance(
                point['lat'], point['lon'],
                station['lat'], station['lon']
            )
            
            if distance < min_distance and distance < threshold_km:
                min_distance = distance
                nearest_station = station_id
        
        return nearest_station
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in km"""
        R = 6371  # Earth's radius in km
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2) * math.sin(dlon/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def analyze_network_patterns(self) -> Dict:
        """Analyze network topology patterns"""
        return {
            'node_count': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges(),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            'clustering_coefficient': nx.average_clustering(self.graph),
            'connected_components': nx.number_connected_components(self.graph),
            'diameter': nx.diameter(self.graph) if nx.is_connected(self.graph) else None
        }