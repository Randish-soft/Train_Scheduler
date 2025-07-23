# File: Model/extractors/terrain_analysis.py
import numpy as np
import requests
from typing import Tuple, List, Dict
import math
import logging

class TerrainAnalyzer:
    def __init__(self):
        self.srtm_base_url = "https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1"
        self.elevation_api_url = "https://api.open-elevation.com/api/v1/lookup"
        self.logger = logging.getLogger(__name__)
        
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
    
    def get_elevation_profile(self, route_points: List[Tuple[float, float]], sample_rate: int = 20) -> List[Dict]:
        """Get elevation profile along a route"""
        elevations = []
        
        # For each segment between points
        for i in range(len(route_points) - 1):
            start_lat, start_lon = route_points[i]
            end_lat, end_lon = route_points[i + 1]
            
            # Interpolate points along the segment
            for j in range(sample_rate):
                ratio = j / sample_rate
                lat = start_lat + (end_lat - start_lat) * ratio
                lon = start_lon + (end_lon - start_lon) * ratio
                
                # Get elevation for this point
                elevation = self._get_point_elevation(lat, lon)
                
                elevations.append({
                    'lat': lat,
                    'lon': lon,
                    'elevation': elevation,
                    'distance_km': self._calculate_distance(route_points[0], (lat, lon))
                })
        
        # Add final point
        if route_points:
            lat, lon = route_points[-1]
            elevation = self._get_point_elevation(lat, lon)
            elevations.append({
                'lat': lat,
                'lon': lon,
                'elevation': elevation,
                'distance_km': self._calculate_distance(route_points[0], route_points[-1])
            })
        
        return elevations
    
    def _get_point_elevation(self, lat: float, lon: float) -> float:
        """Get elevation for a single point using Open Elevation API"""
        try:
            # Try to get real elevation data
            locations = {"locations": [{"latitude": lat, "longitude": lon}]}
            response = requests.post(self.elevation_api_url, json=locations, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and len(data['results']) > 0:
                    return data['results'][0].get('elevation', 100)
        except Exception as e:
            self.logger.debug(f"Failed to get elevation data: {e}")
        
        # Fallback to simulated elevation
        # Create realistic elevation based on latitude (higher towards poles, mountains)
        base_elevation = 100 + abs(lat - 50) * 20  # Base elevation increases away from 50°
        variation = np.sin(lon * 0.5) * 50 + np.sin(lat * 0.3) * 30  # Add variation
        return max(0, base_elevation + variation + np.random.normal(0, 10))
    
    def calculate_route_metrics(self, elevation_profile: List[Dict]) -> Dict:
        """Calculate route metrics from elevation profile"""
        if len(elevation_profile) < 2:
            return {
                'total_distance_km': 0,
                'total_elevation_gain': 0,
                'total_elevation_loss': 0,
                'max_elevation': 0,
                'min_elevation': 0,
                'avg_gradient': 0,
                'max_gradient': 0
            }
        
        # Extract elevations and distances
        elevations = [p['elevation'] for p in elevation_profile]
        distances = [p.get('distance_km', 0) for p in elevation_profile]
        
        # Calculate metrics
        total_distance = max(distances) if distances else 0
        elevation_gain = 0
        elevation_loss = 0
        max_gradient = 0
        
        # Calculate gradients between consecutive points
        gradients = []
        for i in range(1, len(elevation_profile)):
            elev_change = elevations[i] - elevations[i-1]
            dist_change = distances[i] - distances[i-1] if i < len(distances) else 0.1
            
            if dist_change > 0:
                # Convert to percentage gradient
                gradient = (elev_change / (dist_change * 1000)) * 100
                gradients.append(gradient)
                max_gradient = max(max_gradient, abs(gradient))
                
                if elev_change > 0:
                    elevation_gain += elev_change
                else:
                    elevation_loss += abs(elev_change)
        
        return {
            'total_distance_km': total_distance,
            'total_elevation_gain': elevation_gain,
            'total_elevation_loss': elevation_loss,
            'max_elevation': max(elevations),
            'min_elevation': min(elevations),
            'avg_gradient': np.mean(gradients) if gradients else 0,
            'max_gradient': max_gradient,
            'elevation_variance': np.var(elevations)
        }
    
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
        
        try:
            response = requests.post("http://overpass-api.de/api/interpreter", 
                                   data={'data': query}, timeout=120)
            return response.json()
        except Exception as e:
            self.logger.warning(f"Failed to identify obstacles: {e}")
            return {'elements': []}
    
    def get_terrain_cost_matrix(self, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Generate terrain cost matrix for pathfinding"""
        elevation = self.get_elevation_data(bbox)
        slope = self.calculate_slope(elevation)
        
        # Cost increases exponentially with slope
        cost_matrix = 1.0 + (slope / 10.0) ** 2
        
        # Add penalties for extreme elevations
        cost_matrix += np.where(elevation > 1000, elevation / 1000, 0)
        
        return cost_matrix
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate distance between two points in kilometers"""
        lat1, lon1 = point1
        lat2, lon2 = point2
        
        R = 6371  # Earth's radius in km
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2) * math.sin(dlon/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c