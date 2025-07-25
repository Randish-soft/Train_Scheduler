# File: Model/extractors/network_parser.py
import networkx as nx
from typing import Dict, List, Tuple
import math
import logging

class RailwayNetworkParser:
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for railway network
        self.logger = logging.getLogger(__name__)
        
    def parse_osm_to_network(self, osm_data: Dict) -> nx.DiGraph:
        """Convert OSM railway data to network graph"""
        # Clear existing graph
        self.graph.clear()
        
        # Process elements
        nodes_added = set()
        ways_processed = 0
        
        for element in osm_data.get('elements', []):
            if element.get('type') == 'node':
                # Add node to graph
                node_id = str(element.get('id', ''))
                if node_id and node_id not in nodes_added:
                    self.graph.add_node(
                        node_id,
                        lat=element.get('lat', 0),
                        lon=element.get('lon', 0),
                        tags=element.get('tags', {}),
                        element_type='station' if 'railway' in element.get('tags', {}) else 'node'
                    )
                    nodes_added.add(node_id)
            
            elif element.get('type') == 'way':
                # Process way elements
                way_nodes = element.get('nodes', [])
                if len(way_nodes) >= 2:
                    # Add edges between consecutive nodes
                    for i in range(len(way_nodes) - 1):
                        from_node = str(way_nodes[i])
                        to_node = str(way_nodes[i + 1])
                        
                        # Only add edge if both nodes exist in graph
                        if from_node in self.graph and to_node in self.graph:
                            # Calculate edge weight (distance)
                            from_data = self.graph.nodes[from_node]
                            to_data = self.graph.nodes[to_node]
                            distance = self._haversine_distance(
                                from_data['lat'], from_data['lon'],
                                to_data['lat'], to_data['lon']
                            )
                            
                            # Add bidirectional edges for railways
                            self.graph.add_edge(
                                from_node, to_node,
                                weight=distance,
                                way_id=element.get('id', ''),
                                tags=element.get('tags', {})
                            )
                            self.graph.add_edge(
                                to_node, from_node,
                                weight=distance,
                                way_id=element.get('id', ''),
                                tags=element.get('tags', {})
                            )
                    ways_processed += 1
        
        self.logger.info(f"Built network with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges from {ways_processed} ways")
        return self.graph
    
    def analyze_network_patterns(self) -> Dict:
        """Analyze network topology patterns"""
        if self.graph.number_of_nodes() == 0:
            return {
                'topology': {
                    'nodes': 0,
                    'edges': 0,
                    'connected_components': 0,
                    'density': 0
                }
            }
        
        # Basic metrics
        patterns = {
            'topology': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph)
            }
        }
        
        # Connected components (for directed graph, use weak connectivity)
        if self.graph.is_directed():
            patterns['topology']['connected_components'] = nx.number_weakly_connected_components(self.graph)
            patterns['topology']['is_connected'] = nx.is_weakly_connected(self.graph)
        else:
            patterns['topology']['connected_components'] = nx.number_connected_components(self.graph)
            patterns['topology']['is_connected'] = nx.is_connected(self.graph)
        
        # Degree analysis
        degrees = [d for n, d in self.graph.degree()]
        if degrees:
            patterns['degree_distribution'] = {
                'avg_degree': sum(degrees) / len(degrees),
                'max_degree': max(degrees),
                'min_degree': min(degrees),
                'hub_nodes': sum(1 for d in degrees if d >= 4)
            }
        
        # Additional analysis for non-empty graphs
        if self.graph.number_of_edges() > 0:
            try:
                # Find important nodes (high degree centrality)
                degree_centrality = nx.degree_centrality(self.graph)
                top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                patterns['important_nodes'] = [
                    {'node': node, 'centrality': cent} for node, cent in top_nodes
                ]
            except Exception as e:
                self.logger.warning(f"Failed to calculate centrality: {e}")
        
        return patterns
    
    def _connect_track_segments(self, track: Dict, stations: Dict):
        """Connect stations along track segments"""
        geometry = track.get('geometry', [])
        
        if not geometry or len(geometry) < 2:
            return
        
        # Find stations near track endpoints
        connected_stations = []
        
        # Check start and end points
        for point in [geometry[0], geometry[-1]]:
            nearest_station = self._find_nearest_station(point, stations)
            if nearest_station and nearest_station not in connected_stations:
                connected_stations.append(nearest_station)
        
        # Connect consecutive stations
        if len(connected_stations) >= 2:
            for i in range(len(connected_stations) - 1):
                start_station = connected_stations[i]
                end_station = connected_stations[i + 1]
                
                if start_station in stations and end_station in stations:
                    distance = self._haversine_distance(
                        stations[start_station]['lat'], stations[start_station]['lon'],
                        stations[end_station]['lat'], stations[end_station]['lon']
                    )
                    
                    self.graph.add_edge(
                        start_station, end_station,
                        weight=distance,
                        track_id=track['id'],
                        properties=track.get('properties', {})
                    )
    
    def _find_nearest_station(self, point: Dict, stations: Dict, threshold_km: float = 1.0) -> str:
        """Find nearest station within threshold"""
        min_distance = float('inf')
        nearest_station = None
        
        point_lat = point.get('lat', 0)
        point_lon = point.get('lon', 0)
        
        for station_id, station in stations.items():
            distance = self._haversine_distance(
                point_lat, point_lon,
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