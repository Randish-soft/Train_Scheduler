"""
Route Plotting Module - Main Coordinator
========================================

Main route plotting module that coordinates with other specialized modules.
This module focuses purely on route generation and basic visualization,
delegating complex analysis to appropriate specialized modules.

Author: Miguel Ibrahim E
"""

import logging
import math
import json
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path


class RoutePlotter:
    """Main route plotting coordinator - generates routes and basic visualizations."""
    
    def __init__(self, route_data: Dict[str, Any], output_dir: Path):
        self.route_data = route_data
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Create maps directory
        self.maps_dir = output_dir / "route_maps"
        self.maps_dir.mkdir(exist_ok=True)
    
    def plot(self) -> Dict[str, Any]:
        """
        Main plotting method - generates route options and basic visualizations.
        
        Returns:
            Dictionary containing route options and visualization data
        """
        self.logger.info("🗺️ Plotting railway routes...")
        
        # Get cities and demand data
        cities = self.route_data.get('cities', [])
        demand_data = self.route_data.get('demand', {})
        
        if not cities:
            self.logger.warning("No cities data available for route plotting")
            return {}
        
        # Generate route options for high-demand corridors
        route_options = self._generate_route_options(cities, demand_data)
        
        # Create basic route visualizations
        visualizations = self._generate_basic_visualizations(route_options)
        
        # Save route data for other modules to use
        self._save_route_data(route_options)
        
        results = {
            'route_options': route_options,
            'visualizations': visualizations,
            'total_routes_generated': len(route_options),
            'maps_directory': str(self.maps_dir)
        }
        
        self.logger.info(f"✅ Generated {len(route_options)} route options")
        return results
    
    def _generate_route_options(self, cities: List[Dict], demand_data: Dict) -> List[Dict[str, Any]]:
        """Generate multiple route options for each high-demand city pair."""
        self.logger.info("🛤️ Generating route options...")
        
        route_options = []
        demand_matrix = demand_data.get('demand_matrix', {})
        
        # Focus on high-demand routes
        high_demand_routes = demand_data.get('high_demand_routes', [])[:10]  # Top 10
        
        for route_info in high_demand_routes:
            route_name = route_info.get('route', '')
            if ' - ' not in route_name:
                continue
                
            origin_name, destination_name = route_name.split(' - ')
            
            # Find city coordinates
            origin_city = self._find_city_by_name(cities, origin_name)
            destination_city = self._find_city_by_name(cities, destination_name)
            
            if not origin_city or not destination_city:
                continue
            
            # Generate multiple route variants
            route_variants = self._generate_route_variants(
                origin_city, destination_city, cities, route_info
            )
            
            route_options.extend(route_variants)
        
        return route_options
    
    def _find_city_by_name(self, cities: List[Dict], city_name: str) -> Optional[Dict]:
        """Find city data by name."""
        for city in cities:
            if city.get('city_name', '').strip() == city_name.strip():
                return city
        return None
    
    def _generate_route_variants(self, origin: Dict, destination: Dict, 
                               cities: List[Dict], route_info: Dict) -> List[Dict[str, Any]]:
        """Generate multiple routing variants between two cities."""
        variants = []
        
        # Variant 1: Direct route (shortest path)
        direct_route = self._create_direct_route(origin, destination, route_info)
        variants.append(direct_route)
        
        # Variant 2: Via intermediate major city (if beneficial)
        intermediate_route = self._create_intermediate_route(origin, destination, cities, route_info)
        if intermediate_route:
            variants.append(intermediate_route)
        
        # Variant 3: Terrain-aware route (basic terrain consideration)
        terrain_route = self._create_terrain_aware_route(origin, destination, route_info)
        variants.append(terrain_route)
        
        return variants
    
    def _create_direct_route(self, origin: Dict, destination: Dict, route_info: Dict) -> Dict[str, Any]:
        """Create direct route between two cities."""
        distance = self._calculate_distance(
            origin['latitude'], origin['longitude'],
            destination['latitude'], destination['longitude']
        )
        
        # Generate intermediate points for analysis
        route_points = self._generate_route_points(
            origin['latitude'], origin['longitude'],
            destination['latitude'], destination['longitude'],
            num_points=10
        )
        
        return {
            'route_id': f"direct_{origin['city_name']}_{destination['city_name']}",
            'route_name': f"{origin['city_name']} - {destination['city_name']} (Direct)",
            'route_type': 'direct',
            'origin': origin,
            'destination': destination,
            'intermediate_cities': [],
            'route_points': route_points,
            'total_distance_km': distance,
            'demand_data': route_info,
            'routing_strategy': 'shortest_path',
            'basic_metrics': {
                'straight_line_distance': distance,
                'elevation_change': abs((origin.get('elevation', 0) or 0) - (destination.get('elevation', 0) or 0)),
                'average_grade_estimate': self._estimate_average_grade(origin, destination, distance)
            }
        }
    
    def _create_intermediate_route(self, origin: Dict, destination: Dict, 
                                 cities: List[Dict], route_info: Dict) -> Optional[Dict[str, Any]]:
        """Create route via intermediate city if beneficial."""
        # Find potential intermediate cities
        intermediate_candidates = []
        
        for city in cities:
            if city['city_name'] in [origin['city_name'], destination['city_name']]:
                continue
                
            # Check if city is roughly between origin and destination
            if self._is_city_between(origin, destination, city):
                # Calculate total distance via this city
                dist1 = self._calculate_distance(
                    origin['latitude'], origin['longitude'],
                    city['latitude'], city['longitude']
                )
                dist2 = self._calculate_distance(
                    city['latitude'], city['longitude'],
                    destination['latitude'], destination['longitude']
                )
                total_dist = dist1 + dist2
                
                # Calculate direct distance
                direct_dist = self._calculate_distance(
                    origin['latitude'], origin['longitude'],
                    destination['latitude'], destination['longitude']
                )
                
                # Only consider if detour is reasonable (less than 150% of direct distance)
                if total_dist < direct_dist * 1.5 and city['population'] > 100000:
                    intermediate_candidates.append({
                        'city': city,
                        'total_distance': total_dist,
                        'detour_factor': total_dist / direct_dist
                    })
        
        if not intermediate_candidates:
            return None
        
        # Select best intermediate city (shortest total distance)
        best_intermediate = min(intermediate_candidates, key=lambda x: x['total_distance'])
        intermediate_city = best_intermediate['city']
        
        # Generate route points via intermediate city
        points1 = self._generate_route_points(
            origin['latitude'], origin['longitude'],
            intermediate_city['latitude'], intermediate_city['longitude'],
            num_points=5
        )
        points2 = self._generate_route_points(
            intermediate_city['latitude'], intermediate_city['longitude'],
            destination['latitude'], destination['longitude'],
            num_points=5
        )
        
        all_points = points1 + points2[1:]  # Avoid duplicate intermediate point
        
        return {
            'route_id': f"via_{origin['city_name']}_{intermediate_city['city_name']}_{destination['city_name']}",
            'route_name': f"{origin['city_name']} - {destination['city_name']} (via {intermediate_city['city_name']})",
            'route_type': 'via_intermediate',
            'origin': origin,
            'destination': destination,
            'intermediate_cities': [intermediate_city],
            'route_points': all_points,
            'total_distance_km': best_intermediate['total_distance'],
            'demand_data': route_info,
            'routing_strategy': 'via_major_city',
            'detour_factor': best_intermediate['detour_factor'],
            'basic_metrics': {
                'additional_connectivity': f"Serves {intermediate_city['city_name']} ({intermediate_city['population']:,} people)",
                'detour_percentage': (best_intermediate['detour_factor'] - 1) * 100
            }
        }
    
    def _create_terrain_aware_route(self, origin: Dict, destination: Dict, 
                                  route_info: Dict) -> Dict[str, Any]:
        """Create route with basic terrain awareness."""
        # Basic terrain consideration - add curvature for difficult terrain
        bearing = self._calculate_bearing(
            origin['latitude'], origin['longitude'],
            destination['latitude'], destination['longitude']
        )
        
        distance = self._calculate_distance(
            origin['latitude'], origin['longitude'],
            destination['latitude'], destination['longitude']
        )
        
        # Estimate terrain difficulty
        terrain_difficulty = self._estimate_terrain_difficulty(origin, destination)
        
        if terrain_difficulty > 0.6:  # High terrain difficulty
            # Add waypoints to create a more curved, gradual route
            waypoints = self._generate_terrain_waypoints(origin, destination, bearing, distance)
            
            route_points = []
            current_point = (origin['latitude'], origin['longitude'])
            
            for waypoint in waypoints:
                segment_points = self._generate_route_points(
                    current_point[0], current_point[1],
                    waypoint[0], waypoint[1],
                    num_points=3
                )
                route_points.extend(segment_points[:-1])  # Avoid duplicates
                current_point = waypoint
            
            # Add final segment to destination
            final_points = self._generate_route_points(
                current_point[0], current_point[1],
                destination['latitude'], destination['longitude'],
                num_points=3
            )
            route_points.extend(final_points)
            
            # Calculate total distance
            total_distance = sum(
                self._calculate_distance(route_points[i][0], route_points[i][1],
                                       route_points[i+1][0], route_points[i+1][1])
                for i in range(len(route_points)-1)
            )
            
        else:
            # Low terrain difficulty, use mostly direct route
            route_points = self._generate_route_points(
                origin['latitude'], origin['longitude'],
                destination['latitude'], destination['longitude'],
                num_points=8
            )
            total_distance = distance
        
        return {
            'route_id': f"terrain_{origin['city_name']}_{destination['city_name']}",
            'route_name': f"{origin['city_name']} - {destination['city_name']} (Terrain Aware)",
            'route_type': 'terrain_aware',
            'origin': origin,
            'destination': destination,
            'intermediate_cities': [],
            'route_points': route_points,
            'total_distance_km': total_distance,
            'demand_data': route_info,
            'routing_strategy': 'terrain_consideration',
            'terrain_difficulty': terrain_difficulty,
            'basic_metrics': {
                'terrain_complexity': 'High' if terrain_difficulty > 0.6 else 'Medium' if terrain_difficulty > 0.3 else 'Low',
                'route_lengthening': ((total_distance / distance) - 1) * 100 if distance > 0 else 0
            }
        }
    
    def _is_city_between(self, origin: Dict, destination: Dict, intermediate: Dict) -> bool:
        """Check if intermediate city is roughly between origin and destination."""
        min_lat = min(origin['latitude'], destination['latitude'])
        max_lat = max(origin['latitude'], destination['latitude'])
        min_lon = min(origin['longitude'], destination['longitude'])
        max_lon = max(origin['longitude'], destination['longitude'])
        
        # Add buffer for non-straight routes
        lat_buffer = (max_lat - min_lat) * 0.3
        lon_buffer = (max_lon - min_lon) * 0.3
        
        return (min_lat - lat_buffer <= intermediate['latitude'] <= max_lat + lat_buffer and
                min_lon - lon_buffer <= intermediate['longitude'] <= max_lon + lon_buffer)
    
    def _generate_route_points(self, start_lat: float, start_lon: float,
                             end_lat: float, end_lon: float, num_points: int) -> List[Tuple[float, float]]:
        """Generate intermediate points along a route."""
        points = []
        
        for i in range(num_points + 1):
            fraction = i / num_points
            
            # Linear interpolation (great circle would be more accurate)
            lat = start_lat + (end_lat - start_lat) * fraction
            lon = start_lon + (end_lon - start_lon) * fraction
            
            points.append((lat, lon))
        
        return points
    
    def _generate_terrain_waypoints(self, origin: Dict, destination: Dict,
                                  bearing: float, distance: float) -> List[Tuple[float, float]]:
        """Generate waypoints for terrain-aware routing."""
        waypoints = []
        
        # Add waypoints to create a more gradual route
        num_waypoints = max(2, int(distance / 100))  # One waypoint per 100km
        
        for i in range(1, num_waypoints + 1):
            fraction = i / (num_waypoints + 1)
            
            # Add lateral offset to avoid direct steep routes
            lateral_offset = math.sin(fraction * math.pi) * 0.2  # Max 0.2 degree offset
            
            # Calculate intermediate position with offset
            intermediate_bearing = bearing + lateral_offset
            intermediate_distance = distance * fraction
            
            # Convert to lat/lon (simplified calculation)
            lat_offset = intermediate_distance * math.cos(math.radians(intermediate_bearing)) / 111.0
            lon_offset = intermediate_distance * math.sin(math.radians(intermediate_bearing)) / (111.0 * math.cos(math.radians(origin['latitude'])))
            
            waypoint_lat = origin['latitude'] + lat_offset
            waypoint_lon = origin['longitude'] + lon_offset
            
            waypoints.append((waypoint_lat, waypoint_lon))
        
        return waypoints
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula."""
        R = 6371  # Earth radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing between two points."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)
        
        y = math.sin(delta_lon) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
        
        bearing = math.atan2(y, x)
        return math.degrees(bearing)
    
    def _estimate_terrain_difficulty(self, origin: Dict, destination: Dict) -> float:
        """Estimate terrain difficulty between two cities."""
        origin_elev = origin.get('elevation', 0) or 0
        destination_elev = destination.get('elevation', 0) or 0
        
        elevation_diff = abs(destination_elev - origin_elev)
        distance = self._calculate_distance(
            origin['latitude'], origin['longitude'],
            destination['latitude'], destination['longitude']
        )
        
        if distance == 0:
            return 0.0
        
        # Calculate average grade
        avg_grade = (elevation_diff / 1000) / distance  # Convert to decimal grade
        
        # Normalize to 0-1 scale (0.08 = 8% grade is very difficult for railways)
        difficulty = min(avg_grade / 0.08, 1.0)
        
        return difficulty
    
    def _estimate_average_grade(self, origin: Dict, destination: Dict, distance: float) -> float:
        """Estimate average grade percentage."""
        if distance == 0:
            return 0.0
            
        origin_elev = origin.get('elevation', 0) or 0
        destination_elev = destination.get('elevation', 0) or 0
        elevation_diff = abs(destination_elev - origin_elev)
        
        # Convert to percentage grade
        return (elevation_diff / 1000) / distance * 100
    
    def _generate_basic_visualizations(self, route_options: List[Dict]) -> Dict[str, Any]:
        """Generate basic route visualizations."""
        self.logger.info("📊 Generating basic route visualizations...")
        
        visualizations = {
            'route_maps': [],
            'summary_statistics': {}
        }
        
        # Generate map data for each route
        for route in route_options:
            map_data = {
                'route_id': route['route_id'],
                'route_name': route['route_name'],
                'route_type': route['route_type'],
                'coordinates': route['route_points'],
                'origin': {
                    'name': route['origin']['city_name'],
                    'coordinates': [route['origin']['latitude'], route['origin']['longitude']],
                    'population': route['origin']['population']
                },
                'destination': {
                    'name': route['destination']['city_name'],
                    'coordinates': [route['destination']['latitude'], route['destination']['longitude']],
                    'population': route['destination']['population']
                },
                'basic_properties': {
                    'distance_km': route['total_distance_km'],
                    'route_type': route['route_type'],
                    'routing_strategy': route['routing_strategy']
                }
            }
            visualizations['route_maps'].append(map_data)
        
        # Generate summary statistics
        if route_options:
            distances = [r['total_distance_km'] for r in route_options]
            visualizations['summary_statistics'] = {
                'total_routes': len(route_options),
                'average_distance_km': sum(distances) / len(distances),
                'shortest_route_km': min(distances),
                'longest_route_km': max(distances),
                'route_types': {
                    'direct': len([r for r in route_options if r['route_type'] == 'direct']),
                    'via_intermediate': len([r for r in route_options if r['route_type'] == 'via_intermediate']),
                    'terrain_aware': len([r for r in route_options if r['route_type'] == 'terrain_aware'])
                }
            }
        
        return visualizations
    
    def _save_route_data(self, route_options: List[Dict]) -> None:
        """Save route data for other modules to analyze."""
        # Save as GeoJSON for GIS compatibility
        geojson_data = {
            "type": "FeatureCollection",
            "features": []
        }
        
        for route in route_options:
            feature = {
                "type": "Feature",
                "properties": {
                    "route_id": route['route_id'],
                    "route_name": route['route_name'],
                    "route_type": route['route_type'],
                    "distance_km": route['total_distance_km'],
                    "routing_strategy": route['routing_strategy'],
                    "origin_city": route['origin']['city_name'],
                    "destination_city": route['destination']['city_name']
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon, lat] for lat, lon in route['route_points']]
                }
            }
            geojson_data["features"].append(feature)
        
        # Save GeoJSON file
        geojson_path = self.maps_dir / "route_options.geojson"
        with open(geojson_path, 'w') as f:
            json.dump(geojson_data, f, indent=2)
        
        # Save detailed route data as JSON
        routes_path = self.maps_dir / "route_options.json"
        with open(routes_path, 'w') as f:
            json.dump(route_options, f, indent=2, default=str)
        
        self.logger.info(f"✅ Route data saved to {self.maps_dir}")