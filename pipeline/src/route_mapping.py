"""
BCPC Pipeline: Route Mapping Module

This module maps potential rail routes between cities using OpenStreetMap data,
terrain-aware pathfinding, and infrastructure planning. It identifies optimal
alignments considering existing transportation networks and geographic constraints.
"""

import logging
import numpy as np
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from math import radians, cos, sin, asin, sqrt, atan2, degrees
import heapq

logger = logging.getLogger(__name__)

@dataclass
class RoutePoint:
    """Single point along a route."""
    latitude: float
    longitude: float
    elevation_m: float
    distance_from_start_km: float = 0.0
    infrastructure_type: str = "surface"  # surface, tunnel, bridge, viaduct
    speed_limit_kmh: int = 160
    curve_radius_m: Optional[int] = None

@dataclass
class RouteSegment:
    """Segment of a route between two points."""
    start_point: RoutePoint
    end_point: RoutePoint
    distance_km: float
    bearing_degrees: float
    grade_percent: float
    curve_radius_m: Optional[int]
    infrastructure_type: str
    construction_difficulty: str
    estimated_cost_per_km: float

@dataclass
class RouteOption:
    """Complete route option between two cities."""
    origin_city: str
    destination_city: str
    route_name: str
    total_distance_km: float
    route_points: List[RoutePoint]
    route_segments: List[RouteSegment]
    travel_time_hours: float
    max_speed_kmh: int
    average_speed_kmh: float
    total_construction_cost: float
    terrain_difficulty_score: float
    infrastructure_summary: Dict[str, float]
    environmental_impact_score: float
    existing_rail_utilization_percent: float
    feasibility_score: float

class RouteMapper:
    """Maps potential rail routes using OpenStreetMap and terrain data."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize the route mapper."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # OSM Overpass API endpoints
        self.overpass_url = "https://overpass-api.de/api/interpreter"
        
        # Route planning parameters
        self.max_detour_factor = 1.5  # Allow up to 50% detour from straight line
        self.min_curve_radius = 300   # Minimum curve radius in meters
        self.max_grade = 3.5          # Maximum grade percentage
        self.preferred_grade = 2.0    # Preferred maximum grade
        
        # Infrastructure cost estimates (per km)
        self.cost_estimates = {
            'surface_flat': 25_000_000,      # $25M per km on flat terrain
            'surface_rolling': 35_000_000,   # $35M per km on rolling terrain
            'surface_hilly': 50_000_000,     # $50M per km on hilly terrain
            'surface_mountainous': 80_000_000, # $80M per km on mountainous terrain
            'tunnel': 200_000_000,           # $200M per km for tunnels
            'bridge_major': 100_000_000,     # $100M per km for major bridges
            'viaduct': 75_000_000,           # $75M per km for viaducts
            'station': 50_000_000            # $50M per station
        }
        
        # Speed limits by infrastructure type
        self.speed_limits = {
            'surface': 300,     # Up to 300 km/h on surface
            'tunnel': 250,      # 250 km/h in tunnels
            'bridge': 200,      # 200 km/h on bridges
            'viaduct': 250,     # 250 km/h on viaducts
            'curve_tight': 80,  # 80 km/h on tight curves
            'curve_medium': 160, # 160 km/h on medium curves
            'curve_wide': 250    # 250 km/h on wide curves
        }
    
    def map_routes(self, cities: List, terrain_data: Dict[str, Any]) -> Dict[str, List[RouteOption]]:
        """
        Map potential routes between all city pairs.
        
        Args:
            cities: List of CityData objects
            terrain_data: Terrain analysis results from TerrainAnalyzer
            
        Returns:
            Dictionary mapping city pairs to list of route options
        """
        try:
            logger.info(f"Starting route mapping for {len(cities)} cities")
            
            all_routes = {}
            
            # Map routes for all city pairs
            for i, origin in enumerate(cities):
                for j, destination in enumerate(cities):
                    if i >= j:  # Avoid duplicates and self-pairs
                        continue
                    
                    try:
                        route_key = f"{origin.name}-{destination.name}"
                        logger.info(f"Mapping routes for: {route_key}")
                        
                        # Get terrain data for this route
                        terrain_analysis = terrain_data.get(route_key)
                        
                        # Generate multiple route options
                        route_options = self._generate_route_options(
                            origin, destination, terrain_analysis
                        )
                        
                        all_routes[route_key] = route_options
                        
                        # Add delay to respect API rate limits
                        time.sleep(0.5)
                        
                    except Exception as e:
                        logger.warning(f"Error mapping routes for {origin.name} to {destination.name}: {e}")
                        continue
            
            logger.info(f"Route mapping complete for {len(all_routes)} city pairs")
            return all_routes
            
        except Exception as e:
            logger.error(f"Route mapping failed: {e}")
            raise
    
    def _generate_route_options(self, origin, destination, terrain_analysis) -> List[RouteOption]:
        """Generate multiple route options between two cities."""
        try:
            route_options = []
            
            # Option 1: Direct route (straight line with terrain adjustments)
            direct_route = self._create_direct_route(origin, destination, terrain_analysis)
            if direct_route:
                route_options.append(direct_route)
            
            # Option 2: Existing rail network route (if available)
            rail_route = self._create_rail_network_route(origin, destination, terrain_analysis)
            if rail_route and rail_route.route_name != direct_route.route_name:
                route_options.append(rail_route)
            
            # Option 3: Valley route (following river valleys and low terrain)
            valley_route = self._create_valley_route(origin, destination, terrain_analysis)
            if valley_route and self._is_significantly_different(valley_route, route_options):
                route_options.append(valley_route)
            
            # Option 4: Highway corridor route (following existing highways)
            highway_route = self._create_highway_corridor_route(origin, destination, terrain_analysis)
            if highway_route and self._is_significantly_different(highway_route, route_options):
                route_options.append(highway_route)
            
            # Sort by feasibility score
            route_options.sort(key=lambda r: r.feasibility_score, reverse=True)
            
            logger.info(f"Generated {len(route_options)} route options")
            return route_options
            
        except Exception as e:
            logger.error(f"Error generating route options: {e}")
            return []
    
    def _create_direct_route(self, origin, destination, terrain_analysis) -> Optional[RouteOption]:
        """Create a direct route option with minimal deviation."""
        try:
            route_points = []
            
            # Use terrain analysis elevation profile if available
            if terrain_analysis and terrain_analysis.elevation_profile:
                for elev_point in terrain_analysis.elevation_profile:
                    route_point = RoutePoint(
                        latitude=elev_point.latitude,
                        longitude=elev_point.longitude,
                        elevation_m=elev_point.elevation_m,
                        distance_from_start_km=elev_point.distance_from_start_km
                    )
                    route_points.append(route_point)
            else:
                # Create simple two-point route
                route_points = [
                    RoutePoint(origin.latitude, origin.longitude, 100.0, 0.0),
                    RoutePoint(destination.latitude, destination.longitude, 120.0, 
                             self._haversine_distance(origin.latitude, origin.longitude,
                                                    destination.latitude, destination.longitude))
                ]
            
            # Optimize route for rail constraints
            optimized_points = self._optimize_route_for_rail(route_points, terrain_analysis)
            
            # Create route segments
            route_segments = self._create_route_segments(optimized_points, terrain_analysis)
            
            # Calculate route metrics
            total_distance = optimized_points[-1].distance_from_start_km
            travel_time, avg_speed, max_speed = self._calculate_travel_metrics(route_segments)
            construction_cost = self._calculate_construction_cost(route_segments)
            terrain_difficulty = self._calculate_route_terrain_difficulty(route_segments, terrain_analysis)
            infrastructure_summary = self._summarize_infrastructure(route_segments)
            env_impact = self._calculate_route_environmental_impact(route_segments, terrain_analysis)
            feasibility = self._calculate_route_feasibility(route_segments, terrain_analysis)
            
            return RouteOption(
                origin_city=origin.name,
                destination_city=destination.name,
                route_name="Valley Route",
                total_distance_km=total_distance,
                route_points=optimized_points,
                route_segments=route_segments,
                travel_time_hours=travel_time,
                max_speed_kmh=max_speed,
                average_speed_kmh=avg_speed,
                total_construction_cost=construction_cost,
                terrain_difficulty_score=terrain_difficulty,
                infrastructure_summary=infrastructure_summary,
                environmental_impact_score=env_impact,
                existing_rail_utilization_percent=0.0,
                feasibility_score=feasibility
            )
            
        except Exception as e:
            logger.error(f"Error creating valley route: {e}")
            return None
    
    def _create_highway_corridor_route(self, origin, destination, terrain_analysis) -> Optional[RouteOption]:
        """Create route following existing highway corridors."""
        try:
            # Query highway network
            highway_data = self._query_osm_highway_network(origin, destination)
            
            if not highway_data:
                return None
            
            # Create route following highway corridor
            route_points = self._create_highway_parallel_route(highway_data, origin, destination)
            
            if not route_points or len(route_points) < 2:
                return None
            
            # Optimize for rail constraints
            optimized_points = self._optimize_route_for_rail(route_points, terrain_analysis)
            
            # Create segments
            route_segments = self._create_route_segments(optimized_points, terrain_analysis)
            
            # Calculate metrics
            total_distance = optimized_points[-1].distance_from_start_km
            travel_time, avg_speed, max_speed = self._calculate_travel_metrics(route_segments)
            construction_cost = self._calculate_construction_cost(route_segments)
            terrain_difficulty = self._calculate_route_terrain_difficulty(route_segments, terrain_analysis)
            infrastructure_summary = self._summarize_infrastructure(route_segments)
            env_impact = self._calculate_route_environmental_impact(route_segments, terrain_analysis)
            feasibility = self._calculate_route_feasibility(route_segments, terrain_analysis)
            
            return RouteOption(
                origin_city=origin.name,
                destination_city=destination.name,
                route_name="Highway Corridor Route",
                total_distance_km=total_distance,
                route_points=optimized_points,
                route_segments=route_segments,
                travel_time_hours=travel_time,
                max_speed_kmh=max_speed,
                average_speed_kmh=avg_speed,
                total_construction_cost=construction_cost,
                terrain_difficulty_score=terrain_difficulty,
                infrastructure_summary=infrastructure_summary,
                environmental_impact_score=env_impact,
                existing_rail_utilization_percent=0.0,
                feasibility_score=feasibility
            )
            
        except Exception as e:
            logger.error(f"Error creating highway corridor route: {e}")
            return None
    
    def _query_osm_rail_network(self, origin, destination) -> List[Dict[str, Any]]:
        """Query OpenStreetMap for existing rail infrastructure."""
        try:
            # Create bounding box around the route
            lat_min = min(origin.latitude, destination.latitude) - 0.1
            lat_max = max(origin.latitude, destination.latitude) + 0.1
            lon_min = min(origin.longitude, destination.longitude) - 0.1
            lon_max = max(origin.longitude, destination.longitude) + 0.1
            
            # Overpass query for railway infrastructure
            overpass_query = f"""
            [out:json][timeout:25];
            (
              way["railway"~"rail|light_rail|subway|tram"]({lat_min},{lon_min},{lat_max},{lon_max});
              relation["railway"~"rail|light_rail|subway|tram"]({lat_min},{lon_min},{lat_max},{lon_max});
            );
            out geom;
            """
            
            response = requests.post(
                self.overpass_url,
                data=overpass_query,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            rail_elements = []
            
            for element in data.get('elements', []):
                if element.get('type') == 'way' and 'geometry' in element:
                    rail_elements.append({
                        'id': element['id'],
                        'tags': element.get('tags', {}),
                        'geometry': element['geometry'],
                        'railway_type': element.get('tags', {}).get('railway', 'rail')
                    })
            
            logger.info(f"Found {len(rail_elements)} railway elements")
            return rail_elements
            
        except Exception as e:
            logger.warning(f"Error querying OSM rail network: {e}")
            return []
    
    def _query_osm_highway_network(self, origin, destination) -> List[Dict[str, Any]]:
        """Query OpenStreetMap for highway network."""
        try:
            # Create bounding box
            lat_min = min(origin.latitude, destination.latitude) - 0.1
            lat_max = max(origin.latitude, destination.latitude) + 0.1
            lon_min = min(origin.longitude, destination.longitude) - 0.1
            lon_max = max(origin.longitude, destination.longitude) + 0.1
            
            # Query for major highways
            overpass_query = f"""
            [out:json][timeout:25];
            (
              way["highway"~"motorway|trunk|primary"]({lat_min},{lon_min},{lat_max},{lon_max});
            );
            out geom;
            """
            
            response = requests.post(
                self.overpass_url,
                data=overpass_query,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            highway_elements = []
            
            for element in data.get('elements', []):
                if element.get('type') == 'way' and 'geometry' in element:
                    highway_elements.append({
                        'id': element['id'],
                        'tags': element.get('tags', {}),
                        'geometry': element['geometry'],
                        'highway_type': element.get('tags', {}).get('highway', 'primary')
                    })
            
            logger.info(f"Found {len(highway_elements)} highway elements")
            return highway_elements
            
        except Exception as e:
            logger.warning(f"Error querying OSM highway network: {e}")
            return []
    
    def _optimize_route_for_rail(self, route_points: List[RoutePoint], terrain_analysis) -> List[RoutePoint]:
        """Optimize route points for rail construction constraints."""
        try:
            if len(route_points) < 3:
                return route_points
            
            optimized_points = [route_points[0]]  # Always keep start point
            
            for i in range(1, len(route_points) - 1):
                current_point = route_points[i]
                prev_point = optimized_points[-1]
                next_point = route_points[i + 1]
                
                # Calculate curve radius
                curve_radius = self._calculate_curve_radius(prev_point, current_point, next_point)
                
                # Check if curve is too tight
                if curve_radius and curve_radius < self.min_curve_radius:
                    # Smooth the curve by adjusting the point
                    adjusted_point = self._smooth_curve_point(prev_point, current_point, next_point)
                    optimized_points.append(adjusted_point)
                else:
                    # Keep the original point
                    current_point.curve_radius_m = curve_radius
                    optimized_points.append(current_point)
            
            optimized_points.append(route_points[-1])  # Always keep end point
            
            # Recalculate distances
            cumulative_distance = 0.0
            optimized_points[0].distance_from_start_km = 0.0
            
            for i in range(1, len(optimized_points)):
                segment_distance = self._haversine_distance(
                    optimized_points[i-1].latitude, optimized_points[i-1].longitude,
                    optimized_points[i].latitude, optimized_points[i].longitude
                )
                cumulative_distance += segment_distance
                optimized_points[i].distance_from_start_km = cumulative_distance
            
            return optimized_points
            
        except Exception as e:
            logger.error(f"Error optimizing route for rail: {e}")
            return route_points
    
    def _calculate_curve_radius(self, p1: RoutePoint, p2: RoutePoint, p3: RoutePoint) -> Optional[int]:
        """Calculate the radius of curvature at point p2."""
        try:
            # Convert to radians
            lat1, lon1 = radians(p1.latitude), radians(p1.longitude)
            lat2, lon2 = radians(p2.latitude), radians(p2.longitude)
            lat3, lon3 = radians(p3.latitude), radians(p3.longitude)
            
            # Calculate distances
            d12 = self._haversine_distance(p1.latitude, p1.longitude, p2.latitude, p2.longitude) * 1000  # meters
            d23 = self._haversine_distance(p2.latitude, p2.longitude, p3.latitude, p3.longitude) * 1000  # meters
            d13 = self._haversine_distance(p1.latitude, p1.longitude, p3.latitude, p3.longitude) * 1000  # meters
            
            if d12 == 0 or d23 == 0 or d13 == 0:
                return None
            
            # Calculate angle at p2
            angle = np.arccos(max(-1, min(1, (d12**2 + d23**2 - d13**2) / (2 * d12 * d23))))
            
            if angle < 0.01:  # Very small angle, essentially straight
                return 999999  # Very large radius
            
            # Calculate radius using geometry
            radius = (d12 * d23 * d13) / (4 * self._triangle_area(d12, d23, d13))
            
            return int(radius) if radius > 0 else None
            
        except Exception as e:
            logger.warning(f"Error calculating curve radius: {e}")
            return None
    
    def _triangle_area(self, a: float, b: float, c: float) -> float:
        """Calculate triangle area using Heron's formula."""
        s = (a + b + c) / 2
        return sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
    
    def _smooth_curve_point(self, p1: RoutePoint, p2: RoutePoint, p3: RoutePoint) -> RoutePoint:
        """Smooth a curve point to meet minimum radius requirements."""
        try:
            # Simple approach: move p2 towards the line between p1 and p3
            factor = 0.3  # Move 30% towards the straight line
            
            new_lat = p2.latitude + factor * ((p1.latitude + p3.latitude) / 2 - p2.latitude)
            new_lon = p2.longitude + factor * ((p1.longitude + p3.longitude) / 2 - p2.longitude)
            new_elev = p2.elevation_m + factor * ((p1.elevation_m + p3.elevation_m) / 2 - p2.elevation_m)
            
            smoothed_point = RoutePoint(
                latitude=new_lat,
                longitude=new_lon,
                elevation_m=new_elev,
                distance_from_start_km=p2.distance_from_start_km,
                infrastructure_type=p2.infrastructure_type,
                speed_limit_kmh=p2.speed_limit_kmh
            )
            
            return smoothed_point
            
        except Exception as e:
            logger.warning(f"Error smoothing curve point: {e}")
            return p2
    
    def _create_route_segments(self, route_points: List[RoutePoint], terrain_analysis) -> List[RouteSegment]:
        """Create route segments from route points."""
        try:
            segments = []
            
            for i in range(len(route_points) - 1):
                start_point = route_points[i]
                end_point = route_points[i + 1]
                
                # Calculate segment properties
                distance_km = self._haversine_distance(
                    start_point.latitude, start_point.longitude,
                    end_point.latitude, end_point.longitude
                )
                
                bearing = self._calculate_bearing(
                    start_point.latitude, start_point.longitude,
                    end_point.latitude, end_point.longitude
                )
                
                elevation_change = end_point.elevation_m - start_point.elevation_m
                grade_percent = (elevation_change / (distance_km * 1000)) * 100 if distance_km > 0 else 0
                
                # Determine infrastructure type based on terrain and grade
                infrastructure_type = self._determine_infrastructure_type(
                    grade_percent, elevation_change, distance_km, terrain_analysis
                )
                
                # Update route points with infrastructure type
                start_point.infrastructure_type = infrastructure_type
                end_point.infrastructure_type = infrastructure_type
                
                # Determine construction difficulty
                difficulty = self._determine_construction_difficulty(grade_percent, infrastructure_type)
                
                # Calculate cost per km
                cost_per_km = self._calculate_segment_cost_per_km(infrastructure_type, grade_percent)
                
                segment = RouteSegment(
                    start_point=start_point,
                    end_point=end_point,
                    distance_km=distance_km,
                    bearing_degrees=bearing,
                    grade_percent=grade_percent,
                    curve_radius_m=start_point.curve_radius_m,
                    infrastructure_type=infrastructure_type,
                    construction_difficulty=difficulty,
                    estimated_cost_per_km=cost_per_km
                )
                
                segments.append(segment)
            
            return segments
            
        except Exception as e:
            logger.error(f"Error creating route segments: {e}")
            return []
    
    def _determine_infrastructure_type(self, grade_percent: float, elevation_change: float, 
                                     distance_km: float, terrain_analysis) -> str:
        """Determine required infrastructure type for a segment."""
        try:
            # Check if tunnel is needed for steep grades
            if abs(grade_percent) > 8.0:
                return "tunnel"
            
            # Check if bridge is needed for valleys
            if elevation_change < -30 and distance_km > 0.5:
                return "bridge"
            
            # Check if viaduct is needed for elevated sections
            if elevation_change > 50 and distance_km > 1.0:
                return "viaduct"
            
            # Default to surface
            return "surface"
            
        except Exception as e:
            logger.warning(f"Error determining infrastructure type: {e}")
            return "surface"
    
    def _determine_construction_difficulty(self, grade_percent: float, infrastructure_type: str) -> str:
        """Determine construction difficulty level."""
        if infrastructure_type == "tunnel":
            return "extreme"
        elif infrastructure_type in ["bridge", "viaduct"]:
            return "difficult"
        elif abs(grade_percent) > 4.0:
            return "difficult"
        elif abs(grade_percent) > 2.0:
            return "moderate"
        else:
            return "easy"
    
    def _calculate_segment_cost_per_km(self, infrastructure_type: str, grade_percent: float) -> float:
        """Calculate estimated cost per km for a segment."""
        try:
            if infrastructure_type == "tunnel":
                return self.cost_estimates['tunnel']
            elif infrastructure_type == "bridge":
                return self.cost_estimates['bridge_major']
            elif infrastructure_type == "viaduct":
                return self.cost_estimates['viaduct']
            else:
                # Surface construction cost based on terrain difficulty
                if abs(grade_percent) > 6.0:
                    return self.cost_estimates['surface_mountainous']
                elif abs(grade_percent) > 3.0:
                    return self.cost_estimates['surface_hilly']
                elif abs(grade_percent) > 1.0:
                    return self.cost_estimates['surface_rolling']
                else:
                    return self.cost_estimates['surface_flat']
                    
        except Exception as e:
            logger.warning(f"Error calculating segment cost: {e}")
            return self.cost_estimates['surface_rolling']
    
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing between two points."""
        try:
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            
            dlon = lon2 - lon1
            y = sin(dlon) * cos(lat2)
            x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
            
            bearing = atan2(y, x)
            bearing = degrees(bearing)
            bearing = (bearing + 360) % 360
            
            return bearing
            
        except Exception as e:
            logger.warning(f"Error calculating bearing: {e}")
            return 0.0
    
    def _calculate_travel_metrics(self, route_segments: List[RouteSegment]) -> Tuple[float, float, int]:
        """Calculate travel time and speed metrics."""
        try:
            if not route_segments:
                return 0.0, 0.0, 0
            
            total_time = 0.0
            total_distance = 0.0
            max_speed = 0
            
            for segment in route_segments:
                # Determine speed limit for this segment
                speed_limit = self._get_segment_speed_limit(segment)
                max_speed = max(max_speed, speed_limit)
                
                # Calculate travel time for this segment
                segment_time = segment.distance_km / speed_limit  # hours
                total_time += segment_time
                total_distance += segment.distance_km
            
            avg_speed = total_distance / total_time if total_time > 0 else 0
            
            return total_time, avg_speed, max_speed
            
        except Exception as e:
            logger.error(f"Error calculating travel metrics: {e}")
            return 0.0, 0.0, 0
    
    def _get_segment_speed_limit(self, segment: RouteSegment) -> int:
        """Get speed limit for a route segment."""
        try:
            base_speed = self.speed_limits.get(segment.infrastructure_type, 160)
            
            # Reduce speed for tight curves
            if segment.curve_radius_m:
                if segment.curve_radius_m < 500:
                    base_speed = min(base_speed, self.speed_limits['curve_tight'])
                elif segment.curve_radius_m < 1500:
                    base_speed = min(base_speed, self.speed_limits['curve_medium'])
                else:
                    base_speed = min(base_speed, self.speed_limits['curve_wide'])
            
            # Reduce speed for steep grades
            if abs(segment.grade_percent) > 3.0:
                base_speed = int(base_speed * 0.8)
            elif abs(segment.grade_percent) > 1.5:
                base_speed = int(base_speed * 0.9)
            
            return max(base_speed, 40)  # Minimum 40 km/h
            
        except Exception as e:
            logger.warning(f"Error getting segment speed limit: {e}")
            return 80
    
    def _calculate_construction_cost(self, route_segments: List[RouteSegment], 
                                   existing_utilization: float = 0.0) -> float:
        """Calculate total construction cost for the route."""
        try:
            total_cost = 0.0
            
            for segment in route_segments:
                segment_cost = segment.distance_km * segment.estimated_cost_per_km
                total_cost += segment_cost
            
            # Apply discount for existing rail utilization
            if existing_utilization > 0:
                discount_factor = 1.0 - (existing_utilization * 0.6)  # Up to 60% discount
                total_cost *= discount_factor
            
            return total_cost
            
        except Exception as e:
            logger.error(f"Error calculating construction cost: {e}")
            return 0.0
    
    def _calculate_route_terrain_difficulty(self, route_segments: List[RouteSegment], terrain_analysis) -> float:
        """Calculate overall terrain difficulty score for the route."""
        try:
            if terrain_analysis and hasattr(terrain_analysis, 'terrain_difficulty_score'):
                return terrain_analysis.terrain_difficulty_score
            
            # Calculate from segments
            difficulty_scores = {
                'easy': 1.0,
                'moderate': 3.0,
                'difficult': 7.0,
                'extreme': 10.0
            }
            
            total_distance = sum(segment.distance_km for segment in route_segments)
            if total_distance == 0:
                return 5.0
            
            weighted_difficulty = sum(
                segment.distance_km * difficulty_scores.get(segment.construction_difficulty, 5.0)
                for segment in route_segments
            )
            
            return weighted_difficulty / total_distance
            
        except Exception as e:
            logger.error(f"Error calculating route terrain difficulty: {e}")
            return 5.0
    
    def _summarize_infrastructure(self, route_segments: List[RouteSegment]) -> Dict[str, float]:
        """Summarize infrastructure requirements for the route."""
        try:
            infrastructure_summary = {
                'surface_km': 0.0,
                'tunnel_km': 0.0,
                'bridge_km': 0.0,
                'viaduct_km': 0.0
            }
            
            for segment in route_segments:
                key = f"{segment.infrastructure_type}_km"
                if key in infrastructure_summary:
                    infrastructure_summary[key] += segment.distance_km
                else:
                    infrastructure_summary['surface_km'] += segment.distance_km
            
            return infrastructure_summary
            
        except Exception as e:
            logger.error(f"Error summarizing infrastructure: {e}")
            return {'surface_km': 0.0, 'tunnel_km': 0.0, 'bridge_km': 0.0, 'viaduct_km': 0.0}
    
    def _calculate_route_environmental_impact(self, route_segments: List[RouteSegment], terrain_analysis) -> float:
        """Calculate environmental impact score for the route."""
        try:
            if terrain_analysis and hasattr(terrain_analysis, 'environmental_impact_score'):
                return terrain_analysis.environmental_impact_score
            
            # Calculate from infrastructure types
            impact_scores = {
                'surface': 3.0,
                'tunnel': 5.0,
                'bridge': 7.0,
                'viaduct': 6.0
            }
            
            total_distance = sum(segment.distance_km for segment in route_segments)
            if total_distance == 0:
                return 5.0
            
            weighted_impact = sum(
                segment.distance_km * impact_scores.get(segment.infrastructure_type, 5.0)
                for segment in route_segments
            )
            
            return weighted_impact / total_distance
            
        except Exception as e:
            logger.error(f"Error calculating environmental impact: {e}")
            return 5.0
    
    def _calculate_route_feasibility(self, route_segments: List[RouteSegment], terrain_analysis) -> float:
        """Calculate overall feasibility score (0-10, higher is better)."""
        try:
            feasibility_score = 10.0
            
            # Penalize extreme grades
            max_grade = max(abs(segment.grade_percent) for segment in route_segments) if route_segments else 0
            if max_grade > 8.0:
                feasibility_score -= 5.0
            elif max_grade > 4.0:
                feasibility_score -= 2.0
            elif max_grade > 2.0:
                feasibility_score -= 1.0
            
            # Penalize excessive infrastructure
            infrastructure_summary = self._summarize_infrastructure(route_segments)
            total_distance = sum(infrastructure_summary.values())
            
            if total_distance > 0:
                special_infrastructure_ratio = (
                    infrastructure_summary['tunnel_km'] +
                    infrastructure_summary['bridge_km'] +
                    infrastructure_summary['viaduct_km']
                ) / total_distance
                
                feasibility_score -= special_infrastructure_ratio * 3.0
            
            # Penalize very high costs
            total_cost = self._calculate_construction_cost(route_segments)
            total_distance = sum(segment.distance_km for segment in route_segments)
            cost_per_km = total_cost / total_distance if total_distance > 0 else 0
            
            if cost_per_km > 100_000_000:  # > $100M per km
                feasibility_score -= 3.0
            elif cost_per_km > 60_000_000:  # > $60M per km
                feasibility_score -= 1.5
            
            return max(0.0, min(10.0, feasibility_score))
            
        except Exception as e:
            logger.error(f"Error calculating route feasibility: {e}")
            return 5.0
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points in kilometers."""
        try:
            # Convert to radians
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            
            # Earth radius in kilometers
            r = 6371
            
            return c * r
            
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return 0.0
    
    def _is_significantly_different(self, new_route: RouteOption, existing_routes: List[RouteOption], 
                                  threshold: float = 0.15) -> bool:
        """Check if a new route is significantly different from existing routes."""
        try:
            for existing_route in existing_routes:
                # Compare distance
                distance_diff = abs(new_route.total_distance_km - existing_route.total_distance_km)
                distance_ratio = distance_diff / max(new_route.total_distance_km, existing_route.total_distance_km)
                
                # Compare cost
                cost_diff = abs(new_route.total_construction_cost - existing_route.total_construction_cost)
                cost_ratio = cost_diff / max(new_route.total_construction_cost, existing_route.total_construction_cost)
                
                # If both distance and cost are very similar, routes are not significantly different
                if distance_ratio < threshold and cost_ratio < threshold:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking route difference: {e}")
            return True
    
    # Placeholder methods for complex implementations
    def _blend_existing_rail_with_new(self, rail_data: List[Dict], origin, destination) -> List[RoutePoint]:
        """Blend existing rail infrastructure with new construction needs."""
        # Simplified implementation - in reality would involve complex pathfinding
        return [
            RoutePoint(origin.latitude, origin.longitude, 100.0, 0.0),
            RoutePoint(destination.latitude, destination.longitude, 120.0, 
                     self._haversine_distance(origin.latitude, origin.longitude,
                                            destination.latitude, destination.longitude))
        ]
    
    def _calculate_existing_rail_utilization(self, route_segments: List[RouteSegment], 
                                           rail_data: List[Dict]) -> float:
        """Calculate percentage of route using existing rail."""
        # Simplified implementation
        return 25.0 if rail_data else 0.0
    
    def _find_valley_path(self, elevation_profile: List, origin, destination) -> List[RoutePoint]:
        """Find path following valleys through elevation profile."""
        # Simplified implementation - would involve elevation analysis
        if not elevation_profile or len(elevation_profile) < 2:
            return []
        
        valley_points = []
        for point in elevation_profile:
            route_point = RoutePoint(
                latitude=point.latitude,
                longitude=point.longitude,
                elevation_m=point.elevation_m,
                distance_from_start_km=point.distance_from_start_km
            )
            valley_points.append(route_point)
        
        return valley_points
    
    def _create_highway_parallel_route(self, highway_data: List[Dict], origin, destination) -> List[RoutePoint]:
        """Create route parallel to existing highways."""
        # Simplified implementation
        return [
            RoutePoint(origin.latitude, origin.longitude, 100.0, 0.0),
            RoutePoint(destination.latitude, destination.longitude, 120.0,
                     self._haversine_distance(origin.latitude, origin.longitude,
                                            destination.latitude, destination.longitude))
        ]
                origin_city=origin.name,
                destination_city=destination.name,
                route_name="Direct Route",
                total_distance_km=total_distance,
                route_points=optimized_points,
                route_segments=route_segments,
                travel_time_hours=travel_time,
                max_speed_kmh=max_speed,
                average_speed_kmh=avg_speed,
                total_construction_cost=construction_cost,
                terrain_difficulty_score=terrain_difficulty,
                infrastructure_summary=infrastructure_summary,
                environmental_impact_score=env_impact,
                existing_rail_utilization_percent=0.0,
                feasibility_score=feasibility
            )
            
        except Exception as e:
            logger.error(f"Error creating direct route: {e}")
            return None
    
    def _create_rail_network_route(self, origin, destination, terrain_analysis) -> Optional[RouteOption]:
        """Create route utilizing existing rail network where possible."""
        try:
            # Query existing rail infrastructure
            rail_data = self._query_osm_rail_network(origin, destination)
            
            if not rail_data or len(rail_data) < 2:
                # No significant existing rail network found
                return self._create_direct_route(origin, destination, terrain_analysis)
            
            # Create route following existing rail where possible
            route_points = self._blend_existing_rail_with_new(rail_data, origin, destination)
            
            # Optimize for rail constraints
            optimized_points = self._optimize_route_for_rail(route_points, terrain_analysis)
            
            # Create segments
            route_segments = self._create_route_segments(optimized_points, terrain_analysis)
            
            # Calculate utilization of existing rail
            existing_utilization = self._calculate_existing_rail_utilization(route_segments, rail_data)
            
            # Calculate metrics
            total_distance = optimized_points[-1].distance_from_start_km
            travel_time, avg_speed, max_speed = self._calculate_travel_metrics(route_segments)
            construction_cost = self._calculate_construction_cost(route_segments, existing_utilization)
            terrain_difficulty = self._calculate_route_terrain_difficulty(route_segments, terrain_analysis)
            infrastructure_summary = self._summarize_infrastructure(route_segments)
            env_impact = self._calculate_route_environmental_impact(route_segments, terrain_analysis)
            feasibility = self._calculate_route_feasibility(route_segments, terrain_analysis)
            
            return RouteOption(
                origin_city=origin.name,
                destination_city=destination.name,
                route_name="Rail Network Route",
                total_distance_km=total_distance,
                route_points=optimized_points,
                route_segments=route_segments,
                travel_time_hours=travel_time,
                max_speed_kmh=max_speed,
                average_speed_kmh=avg_speed,
                total_construction_cost=construction_cost,
                terrain_difficulty_score=terrain_difficulty,
                infrastructure_summary=infrastructure_summary,
                environmental_impact_score=env_impact,
                existing_rail_utilization_percent=existing_utilization,
                feasibility_score=feasibility
            )
            
        except Exception as e:
            logger.error(f"Error creating rail network route: {e}")
            return None
    
    def _create_valley_route(self, origin, destination, terrain_analysis) -> Optional[RouteOption]:
        """Create route following valleys and low-elevation paths."""
        try:
            if not terrain_analysis or not terrain_analysis.elevation_profile:
                return None
            
            # Find valley path through elevation profile
            valley_points = self._find_valley_path(terrain_analysis.elevation_profile, origin, destination)
            
            if not valley_points or len(valley_points) < 2:
                return None
            
            # Optimize for rail constraints
            optimized_points = self._optimize_route_for_rail(valley_points, terrain_analysis)
            
            # Create segments
            route_segments = self._create_route_segments(optimized_points, terrain_analysis)
            
            # Calculate metrics
            total_distance = optimized_points[-1].distance_from_start_km
            travel_time, avg_speed, max_speed = self._calculate_travel_metrics(route_segments)
            construction_cost = self._calculate_construction_cost(route_segments)
            terrain_difficulty = self._calculate_route_terrain_difficulty(route_segments, terrain_analysis)
            infrastructure_summary = self._summarize_infrastructure(route_segments)
            env_impact = self._calculate_route_environmental_impact(route_segments, terrain_analysis)
            feasibility = self._calculate_route_feasibility(route_segments, terrain_analysis)