# File: Model/extractors/network_parser.py
import networkx as nx
from typing import Dict, List, Tuple, Optional
import math
import json
from collections import defaultdict

class RailwayNetworkParser:
    def __init__(self):
        self.graph = nx.Graph()
        self.stations = {}
        self.tracks = []
        
    def parse_osm_to_network(self, osm_data: Dict) -> nx.Graph:
        """Convert OSM railway data to network graph"""
        self.stations = {}
        self.tracks = []
        
        # First pass: collect all stations and track segments
        for element in osm_data.get('elements', []):
            if element['type'] == 'node':
                self._process_station_node(element)
            elif element['type'] == 'way':
                self._process_track_way(element)
        
        # Build the network graph
        self._build_network_graph()
        
        return self.graph
    
    def _process_station_node(self, element: Dict):
        """Process station nodes from OSM data"""
        tags = element.get('tags', {})
        railway_type = tags.get('railway', '')
        
        if railway_type in ['station', 'halt', 'stop']:
            # Determine station importance
            station_category = self._classify_station(tags)
            
            self.stations[element['id']] = {
                'lat': element['lat'],
                'lon': element['lon'],
                'name': tags.get('name', f"Station_{element['id']}"),
                'type': railway_type,
                'category': station_category,
                'operator': tags.get('operator', 'unknown'),
                'platforms': self._parse_platforms(tags.get('platforms', '1')),
                'services': self._extract_services(tags),
                'electrified': tags.get('electrified', 'no') == 'yes',
                'wheelchair': tags.get('wheelchair', 'unknown')
            }
    
    def _process_track_way(self, element: Dict):
        """Process track ways from OSM data"""
        tags = element.get('tags', {})
        railway_type = tags.get('railway', '')
        
        if railway_type == 'rail' and tags.get('usage') in ['main', 'branch', None]:
            track_data = {
                'id': element['id'],
                'nodes': element.get('nodes', []),
                'geometry': element.get('geometry', []),
                'maxspeed': self._parse_speed(tags.get('maxspeed', '100')),
                'electrified': tags.get('electrified', 'no') == 'yes',
                'gauge': tags.get('gauge', '1435'),
                'tracks': self._parse_tracks(tags.get('railway:tracks', '1')),
                'usage': tags.get('usage', 'main'),
                'service': tags.get('service', 'passenger'),
                'operator': tags.get('operator', 'unknown')
            }
            self.tracks.append(track_data)
    
    def _build_network_graph(self):
        """Build NetworkX graph from stations and tracks"""
        # Add all stations as nodes
        for station_id, station_data in self.stations.items():
            self.graph.add_node(station_id, **station_data)
        
        # Connect stations based on track proximity
        for track in self.tracks:
            connected_stations = self._find_stations_on_track(track)
            self._connect_consecutive_stations(connected_stations, track)
    
    def _find_stations_on_track(self, track: Dict) -> List[Tuple[int, float]]:
        """Find stations along a track and their positions"""
        stations_on_track = []
        
        for station_id, station in self.stations.items():
            min_distance, position = self._point_to_line_distance(
                station['lat'], station['lon'], track['geometry']
            )
            
            # Station is considered on track if within 500m
            if min_distance < 0.5:  # km
                stations_on_track.append((station_id, position))
        
        # Sort by position along track
        stations_on_track.sort(key=lambda x: x[1])
        return stations_on_track
    
    def _connect_consecutive_stations(self, stations_on_track: List[Tuple[int, float]], track: Dict):
        """Connect consecutive stations along a track"""
        for i in range(len(stations_on_track) - 1):
            station1_id = stations_on_track[i][0]
            station2_id = stations_on_track[i + 1][0]
            
            # Calculate distance between stations
            station1 = self.stations[station1_id]
            station2 = self.stations[station2_id]
            
            distance = self._haversine_distance(
                station1['lat'], station1['lon'],
                station2['lat'], station2['lon']
            )
            
            # Add edge with track properties
            self.graph.add_edge(
                station1_id, station2_id,
                distance=distance,
                track_id=track['id'],
                maxspeed=track['maxspeed'],
                electrified=track['electrified'],
                tracks=track['tracks'],
                travel_time=distance / track['maxspeed']  # hours
            )
    
    def _point_to_line_distance(self, lat: float, lon: float, geometry: List[Dict]) -> Tuple[float, float]:
        """Calculate minimum distance from point to line and position along line"""
        if len(geometry) < 2:
            return float('inf'), 0
        
        min_distance = float('inf')
        best_position = 0
        total_length = 0
        
        for i in range(len(geometry) - 1):
            seg_start = geometry[i]
            seg_end = geometry[i + 1]
            
            # Distance to line segment
            dist, pos_on_seg = self._point_to_segment_distance(
                lat, lon, seg_start['lat'], seg_start['lon'], 
                seg_end['lat'], seg_end['lon']
            )
            
            if dist < min_distance:
                min_distance = dist
                best_position = total_length + pos_on_seg
            
            total_length += self._haversine_distance(
                seg_start['lat'], seg_start['lon'],
                seg_end['lat'], seg_end['lon']
            )
        
        return min_distance, best_position
    
    def _point_to_segment_distance(self, px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
        """Distance from point to line segment with position along segment"""
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return self._haversine_distance(px, py, x1, y1), 0
        
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        distance = self._haversine_distance(px, py, closest_x, closest_y)
        position = t * self._haversine_distance(x1, y1, x2, y2)
        
        return distance, position
    
    def _classify_station(self, tags: Dict) -> str:
        """Classify station importance based on tags"""
        if 'railway:station_category' in tags:
            return tags['railway:station_category']
        
        # Infer from other attributes
        if tags.get('railway') == 'station':
            if 'intercity' in tags.get('name', '').lower():
                return 'intercity'
            elif int(tags.get('platforms', '1')) >= 4:
                return 'major'
            else:
                return 'regional'
        
        return 'local'
    
    def _parse_platforms(self, platforms_str: str) -> int:
        """Parse platform count"""
        try:
            return int(platforms_str.split(';')[0])
        except:
            return 1
    
    def _parse_speed(self, speed_str: str) -> int:
        """Parse maximum speed"""
        try:
            return int(speed_str.replace('km/h', '').replace(' ', ''))
        except:
            return 100
    
    def _parse_tracks(self, tracks_str: str) -> int:
        """Parse track count"""
        try:
            return int(tracks_str)
        except:
            return 1
    
    def _extract_services(self, tags: Dict) -> List[str]:
        """Extract available services"""
        services = []
        if tags.get('toilets') == 'yes':
            services.append('toilets')
        if tags.get('shop'):
            services.append('shop')
        if tags.get('restaurant'):
            services.append('restaurant')
        if tags.get('wifi') == 'yes':
            services.append('wifi')
        return services
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in km"""
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    def analyze_network_patterns(self) -> Dict:
        """Analyze network topology and operational patterns"""
        if self.graph.number_of_nodes() == 0:
            return {}
        
        # Basic network metrics
        degree_sequence = [d for n, d in self.graph.degree()]
        
        analysis = {
            'topology': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'avg_degree': sum(degree_sequence) / len(degree_sequence) if degree_sequence else 0,
                'max_degree': max(degree_sequence) if degree_sequence else 0,
                'clustering': nx.average_clustering(self.graph),
                'components': nx.number_connected_components(self.graph)
            },
            'operational': self._analyze_operational_patterns(),
            'station_categories': self._analyze_station_distribution(),
            'connectivity': self._analyze_connectivity_patterns()
        }
        
        return analysis
    
    def _analyze_operational_patterns(self) -> Dict:
        """Analyze operational characteristics"""
        speeds = []
        distances = []
        electrified_ratio = 0
        
        for u, v, data in self.graph.edges(data=True):
            speeds.append(data.get('maxspeed', 100))
            distances.append(data.get('distance', 0))
            if data.get('electrified', False):
                electrified_ratio += 1
        
        total_edges = self.graph.number_of_edges()
        
        return {
            'avg_speed': sum(speeds) / len(speeds) if speeds else 0,
            'avg_segment_length': sum(distances) / len(distances) if distances else 0,
            'electrification_ratio': electrified_ratio / total_edges if total_edges > 0 else 0,
            'total_network_length': sum(distances)
        }
    
    def _analyze_station_distribution(self) -> Dict:
        """Analyze distribution of station types"""
        categories = defaultdict(int)
        for node_id, data in self.graph.nodes(data=True):
            category = data.get('category', 'unknown')
            categories[category] += 1
        
        return dict(categories)
    
    def _analyze_connectivity_patterns(self) -> Dict:
        """Analyze connectivity patterns"""
        hub_threshold = 3  # stations with 3+ connections
        hubs = [n for n, d in self.graph.degree() if d >= hub_threshold]
        
        if nx.is_connected(self.graph):
            diameter = nx.diameter(self.graph)
            avg_path_length = nx.average_shortest_path_length(self.graph)
        else:
            diameter = None
            avg_path_length = None
        
        return {
            'hubs': len(hubs),
            'hub_ratio': len(hubs) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            'diameter': diameter,
            'avg_path_length': avg_path_length
        }