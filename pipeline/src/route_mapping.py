"""
route_mapping.py - Terrain-aware routing module for BCPC Pipeline

This module implements the terrain-aware A* search algorithm for finding
optimal rail routes between cities using OpenStreetMap data and elevation
information from terrain_analysis.py.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import nearest_points, linemerge, transform
import networkx as nx
import osmnx as ox
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, field
import heapq
from collections import defaultdict
import math
import json
from datetime import datetime
import warnings
from functools import partial
import pyproj

# Configure logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class RouteSegment:
    """Represents a segment of a rail route"""
    start_point: Point
    end_point: Point
    distance: float  # in meters
    elevation_change: float  # in meters
    gradient: float  # as decimal (0.035 = 3.5%)
    osm_way_id: Optional[int] = None
    track_type: str = 'conventional'
    infrastructure_type: str = 'existing_rail'  # or 'new_construction'
    
    def to_dict(self) -> Dict:
        """Convert segment to dictionary for serialization"""
        return {
            'start': [self.start_point.x, self.start_point.y],
            'end': [self.end_point.x, self.end_point.y],
            'distance': self.distance,
            'elevation_change': self.elevation_change,
            'gradient': self.gradient,
            'track_type': self.track_type,
            'infrastructure_type': self.infrastructure_type
        }


@dataclass
class RouteResult:
    """Result of routing calculation"""
    route_geometry: LineString
    segments: List[RouteSegment]
    total_distance: float  # in meters
    total_elevation_gain: float  # in meters
    total_elevation_loss: float  # in meters
    max_gradient: float
    avg_gradient: float
    waypoints: List[Point]
    cost: float
    existing_rail_percentage: float = 0.0
    tunnels_required: int = 0
    bridges_required: int = 0
    
    def to_dict(self) -> Dict:
        """Convert route result to dictionary for serialization"""
        return {
            'geometry': {
                'type': 'LineString',
                'coordinates': [[p.x, p.y] for p in self.waypoints]
            },
            'properties': {
                'total_distance_km': self.total_distance / 1000,
                'total_elevation_gain_m': self.total_elevation_gain,
                'total_elevation_loss_m': self.total_elevation_loss,
                'max_gradient_percent': self.max_gradient * 100,
                'avg_gradient_percent': self.avg_gradient * 100,
                'cost': self.cost,
                'existing_rail_percentage': self.existing_rail_percentage,
                'tunnels_required': self.tunnels_required,
                'bridges_required': self.bridges_required
            }
        }


class TerrainAwareRouter:
    """
    Implements terrain-aware A* routing for rail corridors.
    
    This router considers:
    - Distance
    - Elevation changes and gradients
    - Existing rail infrastructure
    - Terrain penalties for steep slopes
    - Curvature constraints for high-speed rail
    """
    
    def __init__(self, 
                 terrain_data: Optional[np.ndarray] = None,
                 bounds: Optional[Tuple[float, float, float, float]] = None,
                 resolution: float = 30.0,
                 max_gradient: float = 0.035,  # 3.5% max gradient for conventional rail
                 gradient_penalty: float = 100.0,
                 elevation_penalty: float = 10.0,
                 new_construction_penalty: float = 2.0):
        """
        Initialize the terrain-aware router.
        
        Args:
            terrain_data: 2D elevation array from terrain_analysis.py
            bounds: (min_lon, min_lat, max_lon, max_lat)
            resolution: DEM resolution in meters
            max_gradient: Maximum allowed gradient (0.035 = 3.5%)
            gradient_penalty: Cost multiplier for steep gradients
            elevation_penalty: Cost multiplier for elevation changes
            new_construction_penalty: Cost multiplier for new track construction
        """
        self.terrain_data = terrain_data
        self.bounds = bounds
        self.resolution = resolution
        self.max_gradient = max_gradient
        self.gradient_penalty = gradient_penalty
        self.elevation_penalty = elevation_penalty
        self.new_construction_penalty = new_construction_penalty
        
        # Calculate terrain dimensions if data provided
        if terrain_data is not None and bounds is not None:
            self.min_lon, self.min_lat, self.max_lon, self.max_lat = bounds
            self.height, self.width = terrain_data.shape
        else:
            self.min_lon = self.min_lat = self.max_lon = self.max_lat = None
            self.height = self.width = None
        
        # Initialize graph components
        self.rail_network = None
        self.road_network = None
        self.combined_graph = None
        
        # Cache for elevation queries
        self._elevation_cache = {}
        
    def load_osm_networks(self, place_name: Optional[str] = None,
                         polygon: Optional[gpd.GeoDataFrame] = None,
                         bbox: Optional[Tuple[float, float, float, float]] = None) -> None:
        """
        Load rail and road networks from OpenStreetMap.
        
        Args:
            place_name: Name of place to download
            polygon: GeoDataFrame with boundary polygon
            bbox: Bounding box (north, south, east, west)
        """
        logger.info("Loading OSM networks...")
        
        # Configure osmnx
        ox.config(use_cache=True, log_console=False)
        
        try:
            # Download rail network
            if polygon is not None and not polygon.empty:
                logger.info("Loading rail network from polygon...")
                self.rail_network = ox.graph_from_polygon(
                    polygon.geometry.iloc[0],
                    network_type='all',
                    custom_filter='["railway"~"rail|light_rail|subway|narrow_gauge"]'
                )
            elif bbox is not None:
                logger.info("Loading rail network from bbox...")
                north, south, east, west = bbox
                self.rail_network = ox.graph_from_bbox(
                    north, south, east, west,
                    network_type='all',
                    custom_filter='["railway"~"rail|light_rail|subway|narrow_gauge"]'
                )
            elif place_name is not None:
                logger.info(f"Loading rail network for {place_name}...")
                self.rail_network = ox.graph_from_place(
                    place_name,
                    network_type='all',
                    custom_filter='["railway"~"rail|light_rail|subway|narrow_gauge"]'
                )
            else:
                raise ValueError("Must provide either place_name, polygon, or bbox")
                
            logger.info(f"Loaded {len(self.rail_network.nodes)} rail nodes, "
                       f"{len(self.rail_network.edges)} rail edges")
            
        except Exception as e:
            logger.warning(f"Could not load rail network: {e}")
            self.rail_network = nx.MultiDiGraph()
            
        # Load road network for areas without rail
        try:
            if polygon is not None and not polygon.empty:
                logger.info("Loading road network from polygon...")
                self.road_network = ox.graph_from_polygon(
                    polygon.geometry.iloc[0],
                    network_type='drive',
                    simplify=True
                )
            elif bbox is not None:
                logger.info("Loading road network from bbox...")
                north, south, east, west = bbox
                self.road_network = ox.graph_from_bbox(
                    north, south, east, west,
                    network_type='drive',
                    simplify=True
                )
            elif place_name is not None:
                logger.info(f"Loading road network for {place_name}...")
                self.road_network = ox.graph_from_place(
                    place_name,
                    network_type='drive',
                    simplify=True
                )
                
            logger.info(f"Loaded {len(self.road_network.nodes)} road nodes, "
                       f"{len(self.road_network.edges)} road edges")
            
        except Exception as e:
            logger.warning(f"Could not load road network: {e}")
            self.road_network = nx.MultiDiGraph()
            
        # Create combined graph
        self._create_combined_graph()
        
    def _create_combined_graph(self) -> None:
        """Create a combined graph with terrain-aware edge weights."""
        logger.info("Creating combined routing graph...")
        self.combined_graph = nx.Graph()
        
        # Track node coordinates for both networks
        all_nodes = {}
        
        # Add rail edges with lower cost
        if self.rail_network and len(self.rail_network.edges) > 0:
            for u, v, key, data in self.rail_network.edges(keys=True, data=True):
                length = data.get('length', 0)
                
                # Get node coordinates
                u_data = self.rail_network.nodes[u]
                v_data = self.rail_network.nodes[v]
                
                # Store node data
                all_nodes[u] = {'x': u_data['x'], 'y': u_data['y']}
                all_nodes[v] = {'x': v_data['x'], 'y': v_data['y']}
                
                u_point = Point(u_data['x'], u_data['y'])
                v_point = Point(v_data['x'], v_data['y'])
                
                # Calculate terrain cost if terrain data available
                if self.terrain_data is not None:
                    terrain_cost = self._calculate_terrain_cost(u_point, v_point)
                else:
                    terrain_cost = 0
                
                # Rail has base cost multiplier of 1.0
                total_cost = length + terrain_cost
                
                self.combined_graph.add_edge(
                    u, v,
                    length=length,
                    terrain_cost=terrain_cost,
                    total_cost=total_cost,
                    edge_type='rail',
                    infrastructure_type='existing_rail',
                    geometry=data.get('geometry', LineString([u_point, v_point]))
                )
                
        # Add road edges with higher cost (for areas without rail)
        if self.road_network and len(self.road_network.edges) > 0:
            for u, v, key, data in self.road_network.edges(keys=True, data=True):
                # Skip if rail already exists between these nodes
                if self.combined_graph.has_edge(u, v):
                    continue
                    
                length = data.get('length', 0)
                
                # Get node coordinates
                if u in self.road_network.nodes and v in self.road_network.nodes:
                    u_data = self.road_network.nodes[u]
                    v_data = self.road_network.nodes[v]
                    
                    # Store node data
                    all_nodes[u] = {'x': u_data['x'], 'y': u_data['y']}
                    all_nodes[v] = {'x': v_data['x'], 'y': v_data['y']}
                    
                    u_point = Point(u_data['x'], u_data['y'])
                    v_point = Point(v_data['x'], v_data['y'])
                    
                    # Calculate terrain cost if terrain data available
                    if self.terrain_data is not None:
                        terrain_cost = self._calculate_terrain_cost(u_point, v_point)
                    else:
                        terrain_cost = 0
                    
                    # Roads have higher base cost (simulating new track construction)
                    total_cost = (length * self.new_construction_penalty) + (terrain_cost * 1.5)
                    
                    self.combined_graph.add_edge(
                        u, v,
                        length=length,
                        terrain_cost=terrain_cost,
                        total_cost=total_cost,
                        edge_type='road',
                        infrastructure_type='new_construction',
                        geometry=data.get('geometry', LineString([u_point, v_point]))
                    )
                    
        # Add node attributes to combined graph
        for node, data in all_nodes.items():
            self.combined_graph.add_node(node, **data)
            
        logger.info(f"Combined graph has {len(self.combined_graph.nodes)} nodes, "
                   f"{len(self.combined_graph.edges)} edges")
                
    def _calculate_terrain_cost(self, start: Point, end: Point) -> float:
        """
        Calculate terrain-based routing cost between two points.
        
        Args:
            start: Starting point
            end: Ending point
            
        Returns:
            Terrain cost based on elevation change and gradient
        """
        # Get elevation at both points
        start_elev = self._get_elevation_at_point(start.x, start.y)
        end_elev = self._get_elevation_at_point(end.x, end.y)
        
        # Calculate distance and elevation change
        # Use geodesic distance for accuracy
        distance = self._geodesic_distance(start, end)
        elev_change = end_elev - start_elev
        
        # Calculate gradient
        if distance > 0:
            gradient = abs(elev_change / distance)
        else:
            gradient = 0
            
        # Apply penalties
        terrain_cost = 0
        
        # Gradient penalty (exponential for steep slopes)
        if gradient > self.max_gradient:
            # Heavily penalize impossible gradients
            terrain_cost += self.gradient_penalty * math.exp(gradient / self.max_gradient)
        else:
            # Linear penalty for acceptable gradients
            terrain_cost += self.gradient_penalty * gradient
            
        # Elevation change penalty (going up costs more than going down)
        if elev_change > 0:
            terrain_cost += self.elevation_penalty * elev_change * 1.5
        else:
            terrain_cost += self.elevation_penalty * abs(elev_change) * 0.5
        
        return terrain_cost
        
    def _get_elevation_at_point(self, lon: float, lat: float) -> float:
        """
        Get elevation at a specific coordinate with caching.
        
        Args:
            lon: Longitude
            lat: Latitude
            
        Returns:
            Elevation in meters
        """
        # Check cache first
        cache_key = (round(lon, 6), round(lat, 6))
        if cache_key in self._elevation_cache:
            return self._elevation_cache[cache_key]
            
        # Default elevation if no terrain data
        if self.terrain_data is None:
            return 0.0
            
        # Convert coordinates to array indices
        x_idx = int((lon - self.min_lon) / (self.max_lon - self.min_lon) * self.width)
        y_idx = int((self.max_lat - lat) / (self.max_lat - self.min_lat) * self.height)
        
        # Clamp indices to array bounds
        x_idx = max(0, min(x_idx, self.width - 1))
        y_idx = max(0, min(y_idx, self.height - 1))
        
        elevation = float(self.terrain_data[y_idx, x_idx])
        
        # Cache the result
        self._elevation_cache[cache_key] = elevation
        
        return elevation
        
    def _geodesic_distance(self, start: Point, end: Point) -> float:
        """
        Calculate geodesic distance between two points.
        
        Args:
            start: Starting point
            end: Ending point
            
        Returns:
            Distance in meters
        """
        # Use pyproj for accurate geodesic calculations
        geod = pyproj.Geod(ellps='WGS84')
        _, _, distance = geod.inv(start.x, start.y, end.x, end.y)
        return distance
        
    def find_route(self, start_city: Dict[str, Any], 
                  end_city: Dict[str, Any],
                  train_type: str = 'conventional',
                  prefer_existing_rail: bool = True) -> Optional[RouteResult]:
        """
        Find optimal route between two cities.
        
        Args:
            start_city: Dictionary with 'name', 'lat', 'lon' keys
            end_city: Dictionary with 'name', 'lat', 'lon' keys
            train_type: 'conventional' or 'high_speed'
            prefer_existing_rail: Whether to prefer existing rail infrastructure
            
        Returns:
            RouteResult object or None if no route found
        """
        logger.info(f"Finding route from {start_city['name']} to {end_city['name']}")
        
        # Adjust parameters based on train type
        if train_type == 'high_speed':
            self.max_gradient = 0.025  # 2.5% for high-speed rail
        else:
            self.max_gradient = 0.035  # 3.5% for conventional rail
        
        # Create points for start and end
        start_point = Point(start_city['lon'], start_city['lat'])
        end_point = Point(end_city['lon'], end_city['lat'])
        
        # Check if graph is empty
        if not self.combined_graph or len(self.combined_graph.nodes) == 0:
            logger.error("Combined graph is empty. No route can be found.")
            return None
            
        # Find nearest nodes in the graph
        try:
            start_node = ox.nearest_nodes(self.combined_graph, start_point.x, start_point.y)
            end_node = ox.nearest_nodes(self.combined_graph, end_point.x, end_point.y)
        except Exception as e:
            logger.error(f"Could not find nearest nodes: {e}")
            return None
        
        if start_node == end_node:
            logger.warning("Start and end nodes are the same")
            return None
            
        try:
            # Adjust edge weights based on preference
            if prefer_existing_rail:
                self._adjust_weights_for_existing_rail()
                
            # Use A* algorithm with terrain-aware heuristic
            path = nx.astar_path(
                self.combined_graph,
                start_node,
                end_node,
                heuristic=lambda u, v: self._heuristic(u, v, end_point),
                weight='total_cost'
            )
            
            # Process path into route result
            route_result = self._process_path_to_route(path, start_point, end_point)
            
            logger.info(f"Route found: {route_result.total_distance/1000:.2f} km, "
                       f"max gradient: {route_result.max_gradient:.2%}, "
                       f"existing rail: {route_result.existing_rail_percentage:.1f}%")
            
            return route_result
            
        except nx.NetworkXNoPath:
            logger.warning(f"No path found between {start_city['name']} and {end_city['name']}")
            return None
        except Exception as e:
            logger.error(f"Error finding route: {e}")
            return None
            
    def _adjust_weights_for_existing_rail(self) -> None:
        """Temporarily adjust weights to prefer existing rail infrastructure."""
        for u, v, data in self.combined_graph.edges(data=True):
            if data.get('infrastructure_type') == 'existing_rail':
                # Reduce cost for existing rail
                data['total_cost'] = data['total_cost'] * 0.5
                
    def _heuristic(self, node: int, target_node: int, target_point: Point) -> float:
        """
        A* heuristic function considering terrain.
        
        Args:
            node: Current node ID
            target_node: Target node ID (unused but required by nx)
            target_point: Target point coordinates
            
        Returns:
            Estimated cost to target
        """
        if node not in self.combined_graph.nodes:
            return float('inf')
            
        node_data = self.combined_graph.nodes[node]
        node_point = Point(node_data['x'], node_data['y'])
        
        # Geodesic distance
        distance = self._geodesic_distance(node_point, target_point)
        
        # Add terrain estimate if available
        if self.terrain_data is not None:
            terrain_estimate = self._calculate_terrain_cost(node_point, target_point)
            return distance + terrain_estimate * 0.3  # Weight terrain less in heuristic
        else:
            return distance
        
    def _process_path_to_route(self, path: List[int], 
                              start_point: Point, 
                              end_point: Point) -> RouteResult:
        """
        Process graph path into detailed route result.
        
        Args:
            path: List of node IDs
            start_point: Original start point
            end_point: Original end point
            
        Returns:
            RouteResult with full route details
        """
        segments = []
        waypoints = []
        total_distance = 0
        total_elev_gain = 0
        total_elev_loss = 0
        max_gradient = 0
        gradient_sum = 0
        total_cost = 0
        existing_rail_distance = 0
        
        # Add start point
        waypoints.append(start_point)
        
        # Add connection from start point to first node if needed
        if len(path) > 0:
            first_node = self.combined_graph.nodes[path[0]]
            first_point = Point(first_node['x'], first_node['y'])
            
            # Only add if not the same as start point
            if start_point.distance(first_point) > 0.0001:  # ~10m threshold
                waypoints.append(first_point)
            
        # Process each edge in path
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = self.combined_graph[u][v]
            
            # Get node points
            u_data = self.combined_graph.nodes[u]
            v_data = self.combined_graph.nodes[v]
            u_point = Point(u_data['x'], u_data['y'])
            v_point = Point(v_data['x'], v_data['y'])
            
            # Calculate segment details
            distance = edge_data['length']
            
            if self.terrain_data is not None:
                u_elev = self._get_elevation_at_point(u_point.x, u_point.y)
                v_elev = self._get_elevation_at_point(v_point.x, v_point.y)
                elev_change = v_elev - u_elev
            else:
                u_elev = v_elev = elev_change = 0
            
            if distance > 0:
                gradient = elev_change / distance
            else:
                gradient = 0
                
            # Create segment
            segment = RouteSegment(
                start_point=u_point,
                end_point=v_point,
                distance=distance,
                elevation_change=elev_change,
                gradient=gradient,
                track_type='high_speed' if abs(gradient) <= 0.025 else 'conventional',
                infrastructure_type=edge_data.get('infrastructure_type', 'new_construction')
            )
            segments.append(segment)
            waypoints.append(v_point)
            
            # Update totals
            total_distance += distance
            if elev_change > 0:
                total_elev_gain += elev_change
            else:
                total_elev_loss += abs(elev_change)
            max_gradient = max(max_gradient, abs(gradient))
            gradient_sum += abs(gradient)
            total_cost += edge_data['total_cost']
            
            if edge_data.get('infrastructure_type') == 'existing_rail':
                existing_rail_distance += distance
            
        # Add connection from last node to end point if needed
        if len(path) > 0:
            last_node = self.combined_graph.nodes[path[-1]]
            last_point = Point(last_node['x'], last_node['y'])
            
            # Only add if not the same as end point
            if end_point.distance(last_point) > 0.0001:  # ~10m threshold
                waypoints.append(end_point)
            
        # Create route geometry
        route_geometry = LineString(waypoints)
        
        # Calculate statistics
        avg_gradient = gradient_sum / len(segments) if segments else 0
        existing_rail_percentage = (existing_rail_distance / total_distance * 100) if total_distance > 0 else 0
        
        # Estimate infrastructure requirements
        tunnels_required = sum(1 for s in segments if s.gradient > self.max_gradient)
        bridges_required = sum(1 for s in segments if s.elevation_change < -10)  # Major drops
        
        return RouteResult(
            route_geometry=route_geometry,
            segments=segments,
            total_distance=total_distance,
            total_elevation_gain=total_elev_gain,
            total_elevation_loss=total_elev_loss,
            max_gradient=max_gradient,
            avg_gradient=avg_gradient,
            waypoints=waypoints,
            cost=total_cost,
            existing_rail_percentage=existing_rail_percentage,
            tunnels_required=tunnels_required,
            bridges_required=bridges_required
        )
        
    def optimize_route_for_speed(self, route: RouteResult, 
                               train_type: str = 'conventional') -> RouteResult:
        """
        Optimize route for different train types (conventional vs high-speed).
        
        Args:
            route: Original route result
            train_type: 'conventional' or 'high_speed'
            
        Returns:
            Optimized route result
        """
        if train_type == 'high_speed':
            # High-speed rail requires gentler curves and gradients
            max_gradient = 0.025  # 2.5% for high-speed
            max_curve_radius = 7000  # meters
        else:
            max_gradient = 0.035  # 3.5% for conventional
            max_curve_radius = 300  # meters
            
        # Mark segments that need modification
        modified_segments = []
        for segment in route.segments:
            if abs(segment.gradient) <= max_gradient:
                modified_segments.append(segment)
            else:
                # Mark for tunnel/bridge/cutting
                logger.info(f"Segment requires engineering: gradient={segment.gradient:.2%}")
                segment.track_type = f"{train_type}_engineered"
                modified_segments.append(segment)
                
        route.segments = modified_segments
        
        # Recalculate route statistics
        route.max_gradient = max(abs(s.gradient) for s in route.segments) if route.segments else 0
        route.avg_gradient = sum(abs(s.gradient) for s in route.segments) / len(route.segments) if route.segments else 0
        
        return route
        
    def export_to_geojson(self, route: RouteResult, 
                         output_path: str,
                         include_elevation_profile: bool = True,
                         include_segments: bool = False) -> None:
        """
        Export route to GeoJSON format.
        
        Args:
            route: Route result to export
            output_path: Path to save GeoJSON file
            include_elevation_profile: Whether to include elevation data
            include_segments: Whether to include individual segments
        """
        # Create feature collection
        features = []
        
        # Main route feature
        route_feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[p.x, p.y] for p in route.waypoints]
            },
            "properties": {
                "type": "rail_route",
                "total_distance_km": route.total_distance / 1000,
                "total_elevation_gain_m": route.total_elevation_gain,
                "total_elevation_loss_m": route.total_elevation_loss,
                "max_gradient_percent": route.max_gradient * 100,
                "avg_gradient_percent": route.avg_gradient * 100,
                "cost": route.cost,
                "existing_rail_percentage": route.existing_rail_percentage,
                "tunnels_required": route.tunnels_required,
                "bridges_required": route.bridges_required,
                "created_at": datetime.now().isoformat()
            }
        }
        features.append(route_feature)
        
        # Add elevation profile if requested
        if include_elevation_profile and self.terrain_data is not None:
            elevation_points = []
            cumulative_distance = 0
            
            for segment in route.segments:
                start_elev = self._get_elevation_at_point(
                    segment.start_point.x, 
                    segment.start_point.y
                )
                elevation_points.append({
                    "distance_km": cumulative_distance / 1000,
                    "elevation_m": start_elev,
                    "gradient_percent": segment.gradient * 100,
                    "lon": segment.start_point.x,
                    "lat": segment.start_point.y
                })
                cumulative_distance += segment.distance
                
            route_feature["properties"]["elevation_profile"] = elevation_points
            
        # Add individual segments if requested
        if include_segments:
            for i, segment in enumerate(route.segments):
                segment_feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [segment.start_point.x, segment.start_point.y],
                            [segment.end_point.x, segment.end_point.y]
                        ]
                    },
                    "properties": {
                        "segment_id": i,
                        "distance_m": segment.distance,
                        "gradient_percent": segment.gradient * 100,
                        "elevation_change_m": segment.elevation_change,
                        "track_type": segment.track_type,
                        "infrastructure_type": segment.infrastructure_type
                    }
                }
                features.append(segment_feature)
            
        # Create feature collection
        feature_collection = {
            "type": "FeatureCollection",
            "features": features,
            "crs": {
                "type": "name",
                "properties": {
                    "name": "EPSG:4326"
                }
            }
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(feature_collection, f, indent=2)
            
        logger.info(f"Route exported to {output_path}")


class RouteMapper:
    """
    Main interface for route mapping in the BCPC pipeline.
    Handles integration with other pipeline components.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the route mapper with configuration.
        
        Args:
            config: Configuration dictionary with routing parameters
        """
        self.config = config or {}
        self.router = None
        self.routes_cache = {}
        
    def initialize_router(self, terrain_data: np.ndarray,
                         bounds: Tuple[float, float, float, float],
                         train_type: str = 'conventional') -> None:
        """
        Initialize the terrain-aware router.
        
        Args:
            terrain_data: Elevation data from terrain_analysis
            bounds: Geographic bounds
            train_type: Type of train service
        """
        # Set gradient limits based on train type
        if train_type == 'high_speed':
            max_gradient = self.config.get('max_gradient_high_speed', 0.025)
        else:
            max_gradient = self.config.get('max_gradient_conventional', 0.035)
            
        self.router = TerrainAwareRouter(
            terrain_data=terrain_data,
            bounds=bounds,
            max_gradient=max_gradient,
            gradient_penalty=self.config.get('gradient_penalty', 100.0),
            elevation_penalty=self.config.get('elevation_penalty', 10.0),
            new_construction_penalty=self.config.get('new_construction_penalty', 2.0)
        )
        
    def load_networks_for_region(self, region_name: Optional[str] = None,
                                boundary_gdf: Optional[gpd.GeoDataFrame] = None,
                                bbox: Optional[Tuple[float, float, float, float]] = None) -> None:
        """
        Load OSM networks for the study region.
        
        Args:
            region_name: Name of the region
            boundary_gdf: Boundary GeoDataFrame
            bbox: Bounding box
        """
        if self.router is None:
            raise ValueError("Router not initialized. Call initialize_router first.")
            
        self.router.load_osm_networks(
            place_name=region_name,
            polygon=boundary_gdf,
            bbox=bbox
        )
        
    def find_route_between_cities(self, city1: Dict[str, Any],
                                 city2: Dict[str, Any],
                                 train_type: str = 'conventional',
                                 use_cache: bool = True) -> Optional[RouteResult]:
        """
        Find optimal route between two cities.
        
        Args:
            city1: First city dictionary
            city2: Second city dictionary
            train_type: Type of train service
            use_cache: Whether to use cached routes
            
        Returns:
            RouteResult or None
        """
        # Create cache key
        cache_key = f"{city1['name']}_{city2['name']}_{train_type}"
        
        # Check cache
        if use_cache and cache_key in self.routes_cache:
            logger.info(f"Using cached route for {cache_key}")
            return self.routes_cache[cache_key]
            
        # Find route
        route = self.router.find_route(city1, city2, train_type)
        
        # Optimize for train type
        if route:
            route = self.router.optimize_route_for_speed(route, train_type)
            
            # Cache the result
            if use_cache:
                self.routes_cache[cache_key] = route
                
        return route
        
    def find_all_routes(self, cities_df: pd.DataFrame,
                       train_type: str = 'conventional',
                       export_individual: bool = False,
                       output_dir: str = 'output/routes') -> pd.DataFrame:
        """
        Find routes between all city pairs in the dataframe.
        
        Args:
            cities_df: DataFrame with city information
            train_type: Type of train service
            export_individual: Whether to export each route to GeoJSON
            output_dir: Directory for individual route exports
            
        Returns:
            DataFrame with route information
        """
        import os
        from itertools import combinations
        
        # Create output directory if needed
        if export_individual:
            os.makedirs(output_dir, exist_ok=True)
            
        routes_data = []
        
        # Generate all city pairs
        city_pairs = list(combinations(cities_df.index, 2))
        
        logger.info(f"Finding routes for {len(city_pairs)} city pairs...")
        
        for i, (idx1, idx2) in enumerate(city_pairs):
            city1 = cities_df.loc[idx1].to_dict()
            city2 = cities_df.loc[idx2].to_dict()
            
            # Ensure required fields
            city1['name'] = city1.get('city', f'City_{idx1}')
            city2['name'] = city2.get('city', f'City_{idx2}')
            
            logger.info(f"Processing {i+1}/{len(city_pairs)}: "
                       f"{city1['name']} to {city2['name']}")
            
            # Find route
            route = self.find_route_between_cities(city1, city2, train_type)
            
            if route:
                # Create route record
                route_record = {
                    'origin_city': city1['name'],
                    'destination_city': city2['name'],
                    'origin_lat': city1['lat'],
                    'origin_lon': city1['lon'],
                    'destination_lat': city2['lat'],
                    'destination_lon': city2['lon'],
                    'distance_km': route.total_distance / 1000,
                    'elevation_gain_m': route.total_elevation_gain,
                    'elevation_loss_m': route.total_elevation_loss,
                    'max_gradient_percent': route.max_gradient * 100,
                    'avg_gradient_percent': route.avg_gradient * 100,
                    'existing_rail_percent': route.existing_rail_percentage,
                    'tunnels_required': route.tunnels_required,
                    'bridges_required': route.bridges_required,
                    'route_cost': route.cost,
                    'geometry': route.route_geometry.wkt
                }
                routes_data.append(route_record)
                
                # Export individual route if requested
                if export_individual:
                    filename = f"{city1['name']}_{city2['name']}_{train_type}.geojson"
                    filepath = os.path.join(output_dir, filename)
                    self.router.export_to_geojson(route, filepath)
            else:
                logger.warning(f"No route found between {city1['name']} and {city2['name']}")
                
        # Create routes dataframe
        routes_df = pd.DataFrame(routes_data)
        
        logger.info(f"Found {len(routes_df)} valid routes out of {len(city_pairs)} pairs")
        
        return routes_df
        
    def export_network_to_geojson(self, output_path: str) -> None:
        """
        Export the entire rail/road network to GeoJSON.
        
        Args:
            output_path: Path to save the network GeoJSON
        """
        if self.router is None or self.router.combined_graph is None:
            raise ValueError("No network loaded")
            
        features = []
        
        # Export edges
        for u, v, data in self.router.combined_graph.edges(data=True):
            # Get node coordinates
            u_data = self.router.combined_graph.nodes[u]
            v_data = self.router.combined_graph.nodes[v]
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [u_data['x'], u_data['y']],
                        [v_data['x'], v_data['y']]
                    ]
                },
                "properties": {
                    "edge_type": data.get('edge_type', 'unknown'),
                    "infrastructure_type": data.get('infrastructure_type', 'unknown'),
                    "length_m": data.get('length', 0),
                    "cost": data.get('total_cost', 0)
                }
            }
            features.append(feature)
            
        # Create feature collection
        feature_collection = {
            "type": "FeatureCollection",
            "features": features
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(feature_collection, f, indent=2)
            
        logger.info(f"Network exported to {output_path}")


# Utility functions for pipeline integration

def create_route_mapper(config: Optional[Dict] = None) -> RouteMapper:
    """
    Factory function to create a RouteMapper instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured RouteMapper instance
    """
    return RouteMapper(config)


def route_cities_from_csv(csv_path: str,
                         terrain_data: np.ndarray,
                         bounds: Tuple[float, float, float, float],
                         output_path: str,
                         train_type: str = 'conventional',
                         region_name: Optional[str] = None) -> pd.DataFrame:
    """
    Main pipeline function to route cities from a CSV file.
    
    Args:
        csv_path: Path to cities CSV file
        terrain_data: Elevation data array
        bounds: Geographic bounds
        output_path: Path for output routes CSV
        train_type: Type of train service
        region_name: Name of the region for OSM data
        
    Returns:
        DataFrame with all routes
    """
    logger.info(f"Loading cities from {csv_path}")
    
    # Load cities
    cities_df = pd.read_csv(csv_path)
    
    # Ensure required columns
    required_cols = ['lat', 'lon']
    if not all(col in cities_df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
        
    # Create route mapper
    mapper = create_route_mapper()
    
    # Initialize router
    mapper.initialize_router(terrain_data, bounds, train_type)
    
    # Load networks
    logger.info(f"Loading OSM networks for region...")
    mapper.load_networks_for_region(region_name=region_name, bbox=bounds)
    
    # Find all routes
    routes_df = mapper.find_all_routes(
        cities_df,
        train_type=train_type,
        export_individual=True
    )
    
    # Save results
    routes_df.to_csv(output_path, index=False)
    logger.info(f"Routes saved to {output_path}")
    
    return routes_df


def visualize_route(route: RouteResult, 
                   terrain_data: Optional[np.ndarray] = None,
                   output_path: Optional[str] = None) -> None:
    """
    Create a simple visualization of a route.
    
    Args:
        route: Route result to visualize
        terrain_data: Optional elevation data for background
        output_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Route map
    ax1.set_title('Route Map')
    
    # Plot route segments with gradient coloring
    norm = Normalize(vmin=0, vmax=0.035)  # 0-3.5% gradient
    cmap = cm.RdYlGn_r  # Red for steep, green for flat
    
    for segment in route.segments:
        color = cmap(norm(abs(segment.gradient)))
        ax1.plot([segment.start_point.x, segment.end_point.x],
                [segment.start_point.y, segment.end_point.y],
                color=color, linewidth=2, alpha=0.8)
    
    # Add start and end markers
    if route.waypoints:
        start = route.waypoints[0]
        end = route.waypoints[-1]
        ax1.plot(start.x, start.y, 'go', markersize=10, label='Start')
        ax1.plot(end.x, end.y, 'ro', markersize=10, label='End')
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Gradient (%)')
    
    # Plot 2: Elevation profile
    ax2.set_title('Elevation Profile')
    
    if terrain_data is not None and route.segments:
        distances = []
        elevations = []
        cumulative_dist = 0
        
        for segment in route.segments:
            distances.append(cumulative_dist / 1000)  # km
            elev = terrain_data[int(segment.start_point.y), int(segment.start_point.x)]
            elevations.append(elev)
            cumulative_dist += segment.distance
            
        ax2.plot(distances, elevations, 'b-', linewidth=2)
        ax2.fill_between(distances, elevations, alpha=0.3)
        ax2.set_xlabel('Distance (km)')
        ax2.set_ylabel('Elevation (m)')
        ax2.grid(True, alpha=0.3)
        
        # Add gradient zones
        ax2.axhline(y=max(elevations) * 0.965, color='r', linestyle='--', 
                   alpha=0.5, label='3.5% gradient limit')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Route visualization saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Route mapping module loaded successfully")
    
    # Example: Create a simple test route
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Test with sample cities
        mapper = create_route_mapper()
        
        # Would need actual terrain data and city data to run
        logger.info("Run with actual data to test routing")