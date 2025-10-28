import logging
import numpy as np
from typing import Dict, Any, List, Tuple

class TerrainUtils:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_grade_profile(self, elevations: List[float], distances: List[float]) -> List[float]:
        """Calculate grade percentages between elevation points"""
        if len(elevations) < 2 or len(distances) < 2:
            return []
        
        grades = []
        for i in range(1, len(elevations)):
            elevation_change = elevations[i] - elevations[i-1]
            distance_change = distances[i] - distances[i-1]
            
            if distance_change > 0:
                grade = (elevation_change / distance_change) * 100  # Percentage
                grades.append(grade)
            else:
                grades.append(0.0)
        
        return grades
    
    def identify_critical_grades(self, grades: List[float], threshold: float = 2.5) -> List[Dict[str, Any]]:
        """Identify critical grades that exceed threshold"""
        critical_grades = []
        
        for i, grade in enumerate(grades):
            if abs(grade) > threshold:
                critical_grades.append({
                    'segment_index': i,
                    'grade_percentage': grade,
                    'severity': 'high' if abs(grade) > 5.0 else 'medium',
                    'mitigation_required': abs(grade) > 3.0
                })
        
        return critical_grades
    
    def calculate_earthworks_volume(self, existing_profile: List[float], 
                                  proposed_profile: List[float], 
                                  width: float = 10.0) -> Dict[str, float]:
        """Calculate cut and fill volumes for terrain modification"""
        if len(existing_profile) != len(proposed_profile):
            raise ValueError("Existing and proposed profiles must have same length")
        
        cut_volume = 0.0
        fill_volume = 0.0
        
        for existing, proposed in zip(existing_profile, proposed_profile):
            height_diff = proposed - existing
            segment_volume = abs(height_diff) * width  # Simplified volume calculation
            
            if height_diff > 0:
                fill_volume += segment_volume
            else:
                cut_volume += segment_volume
        
        # Apply bulking factor for cut material
        bulking_factor = 1.25  # 25% increase in volume when excavated
        cut_volume *= bulking_factor
        
        return {
            'cut_volume_cu_m': cut_volume,
            'fill_volume_cu_m': fill_volume,
            'net_volume_cu_m': fill_volume - cut_volume,
            'bulking_factor_applied': bulking_factor
        }
    
    def assess_soil_conditions(self, terrain_type: str, elevation: float) -> Dict[str, Any]:
        """Assess soil conditions based on terrain type and elevation"""
        soil_conditions = {
            'mountainous': {
                'soil_type': 'rocky',
                'bearing_capacity_mpa': 2.5,
                'excavation_difficulty': 'high',
                'drainage_characteristics': 'good'
            },
            'flat': {
                'soil_type': 'clay_silt',
                'bearing_capacity_mpa': 0.8,
                'excavation_difficulty': 'medium',
                'drainage_characteristics': 'poor'
            },
            'mixed': {
                'soil_type': 'variable',
                'bearing_capacity_mpa': 1.5,
                'excavation_difficulty': 'medium',
                'drainage_characteristics': 'moderate'
            }
        }
        
        base_conditions = soil_conditions.get(terrain_type, soil_conditions['mixed'])
        
        # Adjust for elevation
        if elevation > 1000:
            base_conditions['bearing_capacity_mpa'] *= 1.2
            base_conditions['excavation_difficulty'] = 'very_high'
        elif elevation < 100:
            base_conditions['bearing_capacity_mpa'] *= 0.8
            base_conditions['drainage_characteristics'] = 'very_poor'
        
        return base_conditions
    
    def calculate_tunnel_length(self, mountain_profile: List[float], 
                              max_grade: float = 2.5) -> Dict[str, Any]:
        """Calculate optimal tunnel length through mountainous terrain"""
        if not mountain_profile:
            return {'tunnel_length_km': 0, 'portal_elevations': [], 'cost_implications': {}}
        
        max_elevation = max(mountain_profile)
        min_elevation = min(mountain_profile)
        
        # Find tunnel portals at acceptable grades
        portal_indices = self._find_tunnel_portals(mountain_profile, max_grade)
        
        if len(portal_indices) < 2:
            # No feasible tunnel with current grade constraints
            return {'tunnel_length_km': 0, 'feasible': False}
        
        # Calculate tunnel length
        tunnel_length = portal_indices[-1] - portal_indices[0]
        
        return {
            'tunnel_length_km': tunnel_length * 0.1,  # Convert index to km
            'portal_elevations': [mountain_profile[i] for i in portal_indices],
            'max_grade_avoided': max_grade,
            'elevation_reduction': max_elevation - min(mountain_profile[i] for i in portal_indices),
            'feasible': True,
            'cost_implications': self._estimate_tunnel_cost(tunnel_length * 0.1)
        }
    
    def _find_tunnel_portals(self, profile: List[float], max_grade: float) -> List[int]:
        """Find optimal tunnel portal locations"""
        portals = []
        current_max = profile[0]
        
        for i, elevation in enumerate(profile):
            grade_to_max = ((current_max - elevation) / (i + 1)) * 100 if i > 0 else 0
            
            if grade_to_max > max_grade:
                portals.append(i)
                current_max = elevation
        
        return portals
    
    def _estimate_tunnel_cost(self, length_km: float) -> Dict[str, float]:
        """Estimate tunnel construction cost"""
        base_cost_per_km = 30000000  # $30M per km for tunnel
        total_cost = length_km * base_cost_per_km
        
        return {
            'excavation_cost': total_cost * 0.4,
            'lining_cost': total_cost * 0.3,
            'ventilation_cost': total_cost * 0.15,
            'safety_systems_cost': total_cost * 0.1,
            'engineering_cost': total_cost * 0.05,
            'total_estimated_cost': total_cost
        }
    
    def assess_flood_risk(self, elevation: float, proximity_to_water: float, 
                         historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess flood risk for a location"""
        base_risk = 'low'
        
        # Elevation-based risk
        if elevation < 10:
            base_risk = 'high'
        elif elevation < 50:
            base_risk = 'medium'
        
        # Proximity to water adjustment
        if proximity_to_water < 1.0:  # Within 1 km of water
            if base_risk == 'low':
                base_risk = 'medium'
            elif base_risk == 'medium':
                base_risk = 'high'
        
        # Historical data adjustment
        if historical_data.get('flood_frequency', 0) > 2:  # More than 2 floods in history
            base_risk = 'very_high'
        
        mitigation_measures = self._suggest_flood_mitigation(base_risk)
        
        return {
            'flood_risk_level': base_risk,
            'recommended_elevation_m': self._calculate_safe_elevation(base_risk),
            'mitigation_measures': mitigation_measures,
            'insurance_implications': 'high_premium' if base_risk in ['high', 'very_high'] else 'standard'
        }
    
    def _suggest_flood_mitigation(self, risk_level: str) -> List[str]:
        """Suggest flood mitigation measures based on risk level"""
        measures = {
            'low': ['adequate_drainage', 'regular_maintenance'],
            'medium': ['elevated_foundations', 'flood_barriers', 'drainage_enhancement'],
            'high': ['significant_elevation', 'flood_walls', 'pumping_systems', 'emergency_planning'],
            'very_high': ['avoid_construction', 'relocation', 'comprehensive_flood_defense_system']
        }
        return measures.get(risk_level, measures['medium'])
    
    def _calculate_safe_elevation(self, risk_level: str) -> float:
        """Calculate recommended safe elevation based on risk level"""
        safe_elevations = {
            'low': 5.0,
            'medium': 10.0,
            'high': 20.0,
            'very_high': 50.0
        }
        return safe_elevations.get(risk_level, 10.0)
    
    def analyze_wind_patterns(self, location: Dict[str, float], 
                            terrain_type: str) -> Dict[str, Any]:
        """Analyze wind patterns for a location"""
        # Simplified wind analysis based on terrain
        wind_patterns = {
            'mountainous': {
                'predominant_direction': 'variable',
                'average_speed_ms': 8.0,
                'gust_factor': 1.8,
                'turbulence_level': 'high'
            },
            'flat': {
                'predominant_direction': 'consistent',
                'average_speed_ms': 5.0,
                'gust_factor': 1.3,
                'turbulence_level': 'low'
            },
            'coastal': {
                'predominant_direction': 'onshore_offshore',
                'average_speed_ms': 7.0,
                'gust_factor': 1.6,
                'turbulence_level': 'medium'
            }
        }
        
        base_pattern = wind_patterns.get(terrain_type, wind_patterns['flat'])
        
        # Adjust for elevation
        elevation = location.get('elevation', 0)
        if elevation > 500:
            base_pattern['average_speed_ms'] *= 1.2
            base_pattern['turbulence_level'] = 'very_high'
        
        return {
            **base_pattern,
            'design_implications': self._suggest_wind_design_measures(base_pattern),
            'operational_considerations': self._suggest_wind_operational_measures(base_pattern)
        }
    
    def _suggest_wind_design_measures(self, wind_pattern: Dict[str, Any]) -> List[str]:
        """Suggest design measures based on wind patterns"""
        measures = []
        
        if wind_pattern['turbulence_level'] in ['high', 'very_high']:
            measures.extend([
                'aerodynamic_design',
                'wind_deflection_barriers',
                'reinforced_structures'
            ])
        
        if wind_pattern['average_speed_ms'] > 7.0:
            measures.append('wind_load_calculations')
        
        return measures
    
    def _suggest_wind_operational_measures(self, wind_pattern: Dict[str, Any]) -> List[str]:
        """Suggest operational measures based on wind patterns"""
        measures = []
        
        if wind_pattern['gust_factor'] > 1.5:
            measures.extend([
                'wind_speed_monitoring',
                'speed_restrictions_in_high_winds',
                'emergency_procedures'
            ])
        
        return measures