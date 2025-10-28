import logging
import numpy as np
from typing import Dict, Any, List, Tuple

class TerrainProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process terrain data for route planning"""
        self.logger.info("Processing terrain data")
        
        country_data = context['country_data']
        country_features = context['country_features']
        
        # Generate or load terrain data
        terrain_data = self._generate_terrain_profile(country_data, country_features)
        context['terrain_data'] = terrain_data
        
        # Calculate terrain difficulty scores
        context['terrain_difficulty'] = self._calculate_terrain_difficulty(terrain_data)
        
        self.logger.info("Terrain processing completed")
        return context
    
    def _generate_terrain_profile(self, country_data: Dict[str, Any], country_features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate terrain profile based on country data"""
        terrain_type = country_features.get('terrain_type', 'mixed')
        area = country_data['area']
        
        # Simulate terrain data (in real implementation, this would use GIS data)
        if terrain_type == 'mountainous':
            elevation_profile = self._generate_mountainous_terrain(area)
        elif terrain_type == 'flat':
            elevation_profile = self._generate_flat_terrain(area)
        else:  # mixed
            elevation_profile = self._generate_mixed_terrain(area)
        
        return {
            'type': terrain_type,
            'elevation_profile': elevation_profile,
            'slope_analysis': self._analyze_slopes(elevation_profile),
            'obstacles': self._identify_obstacles(elevation_profile, country_data),
            'rivers_lakes': self._identify_water_bodies(country_data)
        }
    
    def _generate_mountainous_terrain(self, area: float) -> List[float]:
        """Generate mountainous terrain elevation profile"""
        # Simulate random mountains
        num_points = max(100, int(area / 1000))
        elevations = np.random.normal(500, 300, num_points).tolist()
        return [max(0, e) for e in elevations]
    
    def _generate_flat_terrain(self, area: float) -> List[float]:
        """Generate flat terrain elevation profile"""
        num_points = max(100, int(area / 1000))
        return np.random.normal(50, 10, num_points).tolist()
    
    def _generate_mixed_terrain(self, area: float) -> List[float]:
        """Generate mixed terrain elevation profile"""
        num_points = max(100, int(area / 1000))
        elevations = np.random.normal(200, 150, num_points).tolist()
        return [max(0, e) for e in elevations]
    
    def _analyze_slopes(self, elevations: List[float]) -> Dict[str, Any]:
        """Analyze slopes from elevation data"""
        if len(elevations) < 2:
            return {'max_slope': 0, 'avg_slope': 0, 'difficulty_score': 0}
        
        slopes = []
        for i in range(1, len(elevations)):
            slope = abs(elevations[i] - elevations[i-1])
            slopes.append(slope)
        
        return {
            'max_slope': max(slopes) if slopes else 0,
            'avg_slope': sum(slopes) / len(slopes) if slopes else 0,
            'difficulty_score': min(1.0, sum(slopes) / (len(slopes) * 100))  # Normalize to 0-1
        }
    
    def _identify_obstacles(self, elevations: List[float], country_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify major terrain obstacles"""
        obstacles = []
        
        # Identify mountains (high elevation areas)
        high_elevation_threshold = np.percentile(elevations, 80) if elevations else 0
        for i, elev in enumerate(elevations):
            if elev > high_elevation_threshold:
                obstacles.append({
                    'type': 'mountain',
                    'position': i,
                    'elevation': elev,
                    'difficulty': 'high'
                })
        
        # Add urban areas as potential obstacles
        cities = country_data.get('cities', [])
        for city in cities:
            obstacles.append({
                'type': 'urban_area',
                'name': city.get('name', 'Unknown'),
                'population': city.get('population', 0),
                'difficulty': 'medium'
            })
        
        return obstacles
    
    def _identify_water_bodies(self, country_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify rivers and lakes that need bridging"""
        # This would typically come from GIS data
        water_bodies = []
        
        # Simulate some major rivers
        if country_data.get('has_major_rivers', True):
            water_bodies.extend([
                {'type': 'river', 'name': 'Main River', 'width_km': 0.5, 'crossing_difficulty': 'medium'},
                {'type': 'river', 'name': 'Secondary River', 'width_km': 0.2, 'crossing_difficulty': 'low'}
            ])
        
        # Simulate lakes
        if country_data.get('has_lakes', True):
            water_bodies.append({'type': 'lake', 'name': 'Central Lake', 'width_km': 2.0, 'crossing_difficulty': 'high'})
        
        return water_bodies
    
    def _calculate_terrain_difficulty(self, terrain_data: Dict[str, Any]) -> float:
        """Calculate overall terrain difficulty score (0-1)"""
        slope_difficulty = terrain_data['slope_analysis']['difficulty_score']
        obstacle_count = len(terrain_data['obstacles'])
        water_crossings = len([w for w in terrain_data['rivers_lakes'] if w['type'] == 'river'])
        
        # Weighted difficulty calculation
        difficulty = (
            slope_difficulty * 0.5 +
            min(1.0, obstacle_count / 20) * 0.3 +
            min(1.0, water_crossings / 5) * 0.2
        )
        
        return min(1.0, difficulty)