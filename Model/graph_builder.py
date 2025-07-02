"""
Model/graph_builder.py
Automatically build rail network graph based on geography and demand
"""
import networkx as nx
import pandas as pd
import numpy as np
from itertools import combinations
from .dataload import read_json
from .geom import haversine

class RailNetworkBuilder:
    def __init__(self, terrain_classifier=None):
        self.coords = read_json("City-coords").set_index("city")
        self.pop = read_json("Population-per-city").set_index("city")
        self.terrain_classifier = terrain_classifier or self.default_terrain_classifier
        
    def default_terrain_classifier(self, city1, city2, lat1, lon1, lat2, lon2):
        """Simple terrain classification based on coordinates"""
        # Coastal if both cities are near the coast (western Lebanon)
        if lon1 < 35.7 and lon2 < 35.7:
            return "coastal"
        # Mountain if elevation change is significant or eastern regions
        elif abs(lat1 - lat2) < 0.5 and (lon1 > 35.9 or lon2 > 35.9):
            return "mountain"
        else:
            return "rolling"
    
    def should_connect_cities(self, city1, city2, distance_km):
        """Determine if two cities should have a direct rail connection"""
        pop1 = self.pop.loc[city1, 'population']
        pop2 = self.pop.loc[city2, 'population']
        
        # Gravity model score
        gravity_score = (pop1 * pop2) / (distance_km ** 2)
        
        # Connection rules:
        # 1. Always connect if distance < 30km and both have pop > 50k
        if distance_km < 30 and min(pop1, pop2) > 50000:
            return True
        
        # 2. Connect major cities (>200k) if distance < 100km
        if min(pop1, pop2) > 200000 and distance_km < 100:
            return True
        
        # 3. Use gravity score threshold
        if gravity_score > 1e8:  # Tune this threshold
            return True
        
        # 4. Ensure network connectivity - connect isolated cities
        # (This will be handled in post-processing)
        
        return False
    
    def build_network(self, max_direct_distance=150):
        """Build rail network graph automatically"""
        G = nx.Graph()
        
        # Add all cities as nodes
        for city in self.coords.index:
            lat, lon = self.coords.loc[city, ['lat', 'lon']]
            pop = self.pop.loc[city, 'population']
            G.add_node(city, lat=lat, lon=lon, population=pop)
        
        # Consider all possible connections
        connections = []
        for city1, city2 in combinations(self.coords.index, 2):
            lat1, lon1 = self.coords.loc[city1, ['lat', 'lon']]
            lat2, lon2 = self.coords.loc[city2, ['lat', 'lon']]
            
            distance = haversine(lat1, lon1, lat2, lon2)
            
            if distance <= max_direct_distance:
                if self.should_connect_cities(city1, city2, distance):
                    terrain = self.terrain_classifier(city1, city2, lat1, lon1, lat2, lon2)
                    connections.append({
                        'city_a': city1,
                        'city_b': city2,
                        'distance': distance,
                        'terrain': terrain,
                        'gravity_score': (self.pop.loc[city1, 'population'] * 
                                        self.pop.loc[city2, 'population']) / (distance ** 2)
                    })
        
        # Sort by gravity score and add connections
        connections.sort(key=lambda x: x['gravity_score'], reverse=True)
        
        segment_id = 1
        for conn in connections:
            G.add_edge(
                conn['city_a'], 
                conn['city_b'],
                segment_id=f"SEG{segment_id:03d}",
                distance_km=conn['distance'],
                terrain_class=conn['terrain'],
                gravity_score=conn['gravity_score']
            )
            segment_id += 1
        
        # Ensure connectivity - add minimum spanning tree edges if needed
        if not nx.is_connected(G):
            # Create complete graph with distances
            complete_G = nx.Graph()
            for city in self.coords.index:
                complete_G.add_node(city)
            
            for city1, city2 in combinations(self.coords.index, 2):
                if not G.has_edge(city1, city2):
                    lat1, lon1 = self.coords.loc[city1, ['lat', 'lon']]
                    lat2, lon2 = self.coords.loc[city2, ['lat', 'lon']]
                    distance = haversine(lat1, lon1, lat2, lon2)
                    complete_G.add_edge(city1, city2, weight=distance)
            
            # Find minimum spanning tree
            mst = nx.minimum_spanning_tree(complete_G)
            
            # Add MST edges to ensure connectivity
            for u, v in mst.edges():
                if not G.has_edge(u, v):
                    lat1, lon1 = self.coords.loc[u, ['lat', 'lon']]
                    lat2, lon2 = self.coords.loc[v, ['lat', 'lon']]
                    distance = haversine(lat1, lon1, lat2, lon2)
                    terrain = self.terrain_classifier(u, v, lat1, lon1, lat2, lon2)
                    
                    G.add_edge(
                        u, v,
                        segment_id=f"SEG{segment_id:03d}",
                        distance_km=distance,
                        terrain_class=terrain,
                        gravity_score=0  # MST edge, not demand-based
                    )
                    segment_id += 1
        
        return G
    
    def determine_train_types(self, G):
        """Determine allowed train types for each segment"""
        for u, v, data in G.edges(data=True):
            terrain = data['terrain_class']
            distance = data['distance_km']
            
            # High-speed trains only on coastal routes > 40km
            if terrain == 'coastal' and distance > 40:
                train_types = ['TER_4car', 'TGV_8car']
            # Mountain routes limited to regional trains
            elif terrain == 'mountain':
                train_types = ['TER_4car']
            # Standard routes
            else:
                train_types = ['TER_4car']
            
            G[u][v]['allowed_train_types'] = ';'.join(train_types)
        
        return G
    
    def export_tracks(self, G):
        """Export tracks data for the model"""
        tracks = []
        for u, v, data in G.edges(data=True):
            tracks.append({
                'segment_id': data['segment_id'],
                'city_a': u,
                'city_b': v,
                'distance_km': data['distance_km'],
                'terrain_class': data['terrain_class'],
                'allowed_train_types': data['allowed_train_types']
            })
        
        return pd.DataFrame(tracks)