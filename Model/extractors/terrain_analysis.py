# File: model/extractors/terrain_analysis.py
import numpy as np
import requests
from typing import Tuple, List, Dict, Optional
import json
from io import BytesIO
import math

class TerrainAnalyzer:
    def __init__(self):
        self.open_elevation_url = "https://api.open-elevation.com/api/v1/lookup"
        self.cache = {}
        
    def get_elevation_profile(self, points: List[Tuple[float, float]], 
                            sample_rate: int = 100) -> List[Dict]:
        """Get elevation profile for a route with given points"""
        if len(points) < 2:
            return []
        
        # Interpolate points along the route
        interpolated_points = self._interpolate_route(points, sample_rate)
        
        # Get elevations in batches (API limit)
        batch_size = 100
        elevation_profile = []
        
        for i in range(0, len(interpolated_points), batch_size):
            batch = interpolated_points[i:i + batch_size]
            elevations = self._get_batch_elevations(batch)
            elevation_profile.extend(elevations)
        
        return elevation_profile
    
    def _interpolate_route(self, points: List[Tuple[float, float]], 
                          sample_rate: int) -> List[Tuple[float, float]]:
        """Interpolate points along route for elevation sampling"""
        if len(points) < 2:
            return points
        
        interpolated = [points[0]]
        
        for i in range(len(points) - 1):
            start_lat, start_lon = points[i]
            end_lat, end_lon = points[i + 1]
            
            # Calculate distance between points
            distance = self._haversine_distance(start_lat, start_lon, end_lat, end_lon)
            
            # Number of interpolated points based on distance and sample rate
            num_points = max(1, int(distance * sample_rate / 10))  # 10km = sample_rate points
            
            for j in range(1, num_points + 1):
                t = j / num_points
                lat = start_lat + t * (end_lat - start_lat)
                lon = start_lon + t * (end_lon - start_lon)
                interpolated.append((lat, lon))
        
        return interpolated
    
    def _get_batch_elevations(self, points: List[Tuple[float, float]]) -> List[Dict]:
        """Get elevations for a batch of points"""
        locations = [{"latitude": lat, "longitude": lon} for lat, lon in points]
        
        try:
            response = requests.post(
                self.open_elevation_url,
                json={"locations": locations},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for i, result in enumerate(data.get('results', [])):
                    lat, lon = points[i]
                    results.append({
                        'lat': lat,
                        'lon': lon,
                        'elevation': result.get('elevation', 0),
                        'distance_km': 0 if i == 0 else self._calculate_cumulative_distance(points[:i+1])
                    })
                
                return results
            
        except Exception as e:
            print(f"Elevation API error: {e}")
            
        # Fallback: mock elevation data
        return self._generate_mock_elevation(points)
    
    def _generate_mock_elevation(self, points: List[Tuple[float, float]]) -> List[Dict]:
        """Generate realistic mock elevation data"""
        results = []
        cumulative_distance = 0
        
        for i, (lat, lon) in enumerate(points):
            # Simple elevation model based on latitude (mountains in central Europe)
            base_elevation = max(0, (abs(lat - 47) * 100) + (abs(lon - 8) * 50))
            
            # Add some realistic variation
            noise = np.random.normal(0, 20)
            elevation = max(0, base_elevation + noise)
            
            if i > 0:
                prev_lat, prev_lon = points[i-1]
                cumulative_distance += self._haversine_distance(prev_lat, prev_lon, lat, lon)
            
            results.append({
                'lat': lat,
                'lon': lon,
                'elevation': elevation,
                'distance_km': cumulative_distance
            })
        
        return results
    
    def _calculate_cumulative_distance(self, points: List[Tuple[float, float]]) -> float:
        """Calculate cumulative distance along points"""
        total_distance = 0
        for i in range(1, len(points)):
            prev_lat, prev_lon = points[i-1]
            lat, lon = points[i]
            total_distance += self._haversine_distance(prev_lat, prev_lon, lat, lon)
        return total_distance
    
    def calculate_route_metrics(self, elevation_profile: List[Dict]) -> Dict:
        """Calculate route terrain metrics"""
        if len(elevation_profile) < 2:
            return {}
        
        elevations = [p['elevation'] for p in elevation_profile]
        distances = [p['distance_km'] for p in elevation_profile]
        
        # Calculate slopes between consecutive points
        slopes = []
        for i in range(1, len(elevation_profile)):
            rise = elevation_profile[i]['elevation'] - elevation_profile[i-1]['elevation']
            run = (elevation_profile[i]['distance_km'] - elevation_profile[i-1]['distance_km']) * 1000  # meters
            
            if run > 0:
                slope_percent = (rise / run) * 100
                slopes.append(abs(slope_percent))
        
        # Identify challenging sections
        steep_sections = self._identify_steep_sections(elevation_profile, threshold=3.0)
        tunnel_candidates = self._identify_tunnel_candidates(elevation_profile)
        bridge_candidates = self._identify_bridge_candidates(elevation_profile)
        
        return {
            'total_distance_km': max(distances),
            'elevation_gain_m': sum(max(0, elevation_profile[i]['elevation'] - elevation_profile[i-1]['elevation']) 
                                  for i in range(1, len(elevation_profile))),
            'elevation_loss_m': sum(max(0, elevation_profile[i-1]['elevation'] - elevation_profile[i]['elevation']) 
                                  for i in range(1, len(elevation_profile))),
            'max_elevation_m': max(elevations),
            'min_elevation_m': min(elevations),
            'avg_slope_percent': sum(slopes) / len(slopes) if slopes else 0,
            'max_slope_percent': max(slopes) if slopes else 0,
            'steep_sections': steep_sections,
            'tunnel_candidates': tunnel_candidates,
            'bridge_candidates': bridge_candidates,
            'terrain_difficulty': self._calculate_terrain_difficulty(slopes, elevations)
        }
    
    def _identify_steep_sections(self, profile: List[Dict], threshold: float = 3.0) -> List[Dict]:
        """Identify sections requiring special engineering"""
        steep_sections = []
        current_section = None
        
        for i in range(1, len(profile)):
            rise = profile[i]['elevation'] - profile[i-1]['elevation']
            run = (profile[i]['distance_km'] - profile[i-1]['distance_km']) * 1000
            
            if run > 0:
                slope_percent = abs(rise / run) * 100
                
                if slope_percent > threshold:
                    if current_section is None:
                        current_section = {
                            'start_km': profile[i-1]['distance_km'],
                            'start_elevation': profile[i-1]['elevation'],
                            'max_slope': slope_percent
                        }
                    else:
                        current_section['max_slope'] = max(current_section['max_slope'], slope_percent)
                else:
                    if current_section is not None:
                        current_section['end_km'] = profile[i-1]['distance_km']
                        current_section['end_elevation'] = profile[i-1]['elevation']
                        current_section['length_km'] = current_section['end_km'] - current_section['start_km']
                        steep_sections.append(current_section)
                        current_section = None
        
        return steep_sections
    
    def _identify_tunnel_candidates(self, profile: List[Dict]) -> List[Dict]:
        """Identify locations where tunnels might be beneficial"""
        tunnel_candidates = []
        
        for i in range(len(profile)):
            if profile[i]['elevation'] > 800:  # High elevation threshold
                # Look for mountain passes
                window_size = min(10, len(profile) - i)
                local_elevations = [profile[j]['elevation'] for j in range(i, min(i + window_size, len(profile)))]
                
                if len(local_elevations) > 3:
                    # Check if this is a peak that could be tunneled through
                    if (profile[i]['elevation'] == max(local_elevations) and 
                        profile[i]['elevation'] > profile[max(0, i-5)]['elevation'] + 200 and
                        profile[i]['elevation'] > profile[min(len(profile)-1, i+5)]['elevation'] + 200):
                        
                        tunnel_candidates.append({
                            'location_km': profile[i]['distance_km'],
                            'elevation_m': profile[i]['elevation'],
                            'estimated_length_km': 2.0,  # Rough estimate
                            'reason': 'mountain_pass'
                        })
        
        return tunnel_candidates
    
    def _identify_bridge_candidates(self, profile: List[Dict]) -> List[Dict]:
        """Identify locations where bridges might be needed"""
        bridge_candidates = []
        
        for i in range(1, len(profile) - 1):
            # Look for valleys or significant elevation drops
            if (profile[i]['elevation'] < profile[i-1]['elevation'] - 50 and 
                profile[i]['elevation'] < profile[i+1]['elevation'] - 50):
                
                bridge_candidates.append({
                    'location_km': profile[i]['distance_km'],
                    'elevation_m': profile[i]['elevation'],
                    'depth_m': min(profile[i-1]['elevation'] - profile[i]['elevation'],
                                 profile[i+1]['elevation'] - profile[i]['elevation']),
                    'reason': 'valley_crossing'
                })
        
        return bridge_candidates
    
    def _calculate_terrain_difficulty(self, slopes: List[float], elevations: List[float]) -> str:
        """Calculate overall terrain difficulty rating"""
        if not slopes or not elevations:
            return 'unknown'
        
        avg_slope = sum(slopes) / len(slopes)
        max_slope = max(slopes)
        elevation_range = max(elevations) - min(elevations)
        
        if max_slope > 5.0 or elevation_range > 1000:
            return 'very_difficult'
        elif max_slope > 3.0 or elevation_range > 500:
            return 'difficult'
        elif max_slope > 1.5 or elevation_range > 200:
            return 'moderate'
        else:
            return 'easy'
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km"""
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    def get_terrain_cost_matrix(self, bbox: Tuple[float, float, float, float], 
                              resolution: int = 50) -> np.ndarray:
        """Generate terrain cost matrix for pathfinding algorithms"""
        west, south, east, north = bbox
        
        # Create grid of points
        lats = np.linspace(south, north, resolution)
        lons = np.linspace(west, east, resolution)
        
        cost_matrix = np.ones((resolution, resolution))
        
        # Sample elevations for cost calculation
        sample_points = []
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                sample_points.append((lat, lon))
        
        # Get elevations (in batches due to API limits)
        elevations = []
        for i in range(0, len(sample_points), 50):
            batch = sample_points[i:i+50]
            batch_elevations = self._get_batch_elevations(batch)
            elevations.extend([e['elevation'] for e in batch_elevations])
        
        # Fill cost matrix
        for i in range(resolution):
            for j in range(resolution):
                idx = i * resolution + j
                if idx < len(elevations):
                    elevation = elevations[idx]
                    
                    # Calculate slope-based cost
                    if i > 0 and j > 0 and idx > resolution:
                        prev_elevation = elevations[idx - resolution - 1]
                        slope = abs(elevation - prev_elevation) / 1000  # normalize
                        cost_matrix[i, j] = 1 + slope * 10
                    
                    # Add elevation penalty for very high areas
                    if elevation > 1000:
                        cost_matrix[i, j] *= (1 + (elevation - 1000) / 1000)
        
        return cost_matrix