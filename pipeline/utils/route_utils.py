import logging
import numpy as np
from typing import Dict, Any, List, Tuple

class RouteUtils:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_optimal_curve_radius(self, design_speed: float, 
                                    superelevation: float = 0.0) -> float:
        """Calculate optimal curve radius based on design speed and superelevation"""
        # Using standard railway design formula
        # R = V² / (127 * (e + f))
        # where:
        # R = curve radius (meters)
        # V = design speed (km/h)
        # e = superelevation (decimal)
        # f = side friction factor
        
        # Typical side friction factor for railways
        friction_factor = 0.1
        
        if design_speed <= 0:
            return 1000.0  # Default minimum radius
        
        radius = (design_speed ** 2) / (127 * (superelevation + friction_factor))
        
        # Apply minimum radius constraints
        min_radius = self._get_minimum_radius(design_speed)
        return max(min_radius, radius)
    
    def _get_minimum_radius(self, design_speed: float) -> float:
        """Get minimum curve radius based on design speed"""
        # Standard railway design minimum radii
        if design_speed <= 80:
            return 300.0
        elif design_speed <= 120:
            return 600.0
        elif design_speed <= 160:
            return 1000.0
        elif design_speed <= 200:
            return 2000.0
        elif design_speed <= 250:
            return 4000.0
        else:  # High speed
            return 7000.0
    
    def calculate_transition_length(self, curve_radius: float, 
                                  design_speed: float, 
                                  cant: float = 0.0) -> float:
        """Calculate transition curve length"""
        # Using standard transition length formula
        # L = V³ / (47 * R * C)
        # where:
        # L = transition length (meters)
        # V = design speed (km/h)
        # R = curve radius (meters)
        # C = rate of change of cant (mm/s)
        
        if curve_radius <= 0 or design_speed <= 0:
            return 50.0  # Default minimum transition
        
        # Typical rate of change of cant (mm/s)
        cant_rate = 35.0  # Conservative value
        
        transition_length = (design_speed ** 3) / (47 * curve_radius * cant_rate)
        
        # Apply minimum and maximum constraints
        min_transition = max(20.0, design_speed)  # At least speed value in meters
        max_transition = 150.0  # Maximum practical transition
        
        return max(min_transition, min(max_transition, transition_length))
    
    def calculate_stopping_distance(self, initial_speed: float, 
                                 deceleration: float,
                                 gradient: float = 0.0) -> float:
        """Calculate stopping distance considering gradient"""
        # Convert speed to m/s
        speed_ms = initial_speed / 3.6
        
        # Adjust deceleration for gradient
        # Downhill: reduced effective deceleration
        # Uphill: increased effective deceleration
        gradient_effect = gradient / 100  # Convert percentage to decimal
        effective_deceleration = deceleration - (9.81 * gradient_effect)
        
        # Ensure deceleration is positive
        effective_deceleration = max(0.1, effective_deceleration)
        
        # Calculate stopping distance: s = v² / (2 * a)
        stopping_distance = (speed_ms ** 2) / (2 * effective_deceleration)
        
        return stopping_distance
    
    def optimize_station_spacing(self, route_length: float, 
                               demand_pattern: Dict[str, Any],
                               train_performance: Dict[str, Any]) -> List[float]:
        """Optimize station spacing along a route"""
        total_length = route_length
        min_spacing = 2.0  # Minimum station spacing in km
        max_spacing = 20.0  # Maximum station spacing in km
        
        # Calculate optimal spacing based on demand and performance
        base_spacing = self._calculate_base_spacing(demand_pattern, train_performance)
        
        # Adjust for route length
        num_stations = max(2, int(total_length / base_spacing) + 1)
        actual_spacing = total_length / (num_stations - 1)
        
        # Ensure spacing is within limits
        actual_spacing = max(min_spacing, min(max_spacing, actual_spacing))
        
        # Generate spacing list
        spacings = [actual_spacing] * (num_stations - 1)
        
        return spacings
    
    def _calculate_base_spacing(self, demand_pattern: Dict[str, Any],
                              train_performance: Dict[str, Any]) -> float:
        """Calculate base station spacing"""
        # Consider demand density
        demand_density = demand_pattern.get('density', 'medium')
        density_factors = {
            'very_high': 3.0,
            'high': 5.0,
            'medium': 8.0,
            'low': 12.0,
            'very_low': 15.0
        }
        
        base_spacing = density_factors.get(demand_density, 8.0)
        
        # Adjust for train performance
        max_speed = train_performance.get('max_speed_kmh', 120)
        if max_speed > 200:
            base_spacing *= 1.5  # Higher speed trains need longer spacing
        elif max_speed < 80:
            base_spacing *= 0.7  # Lower speed trains can have closer spacing
        
        return base_spacing
    
    def calculate_energy_consumption(self, route_profile: Dict[str, Any],
                                  train_specs: Dict[str, Any],
                                  operating_pattern: Dict[str, Any]) -> Dict[str, float]:
        """Calculate energy consumption for a route"""
        distance = route_profile.get('total_distance_km', 0)
        elevation_gain = route_profile.get('total_elevation_gain_m', 0)
        num_stations = route_profile.get('num_stations', 0)
        
        # Base energy consumption (kWh/km)
        base_consumption = train_specs.get('base_energy_kwh_km', 15.0)
        
        # Acceleration energy
        stops_energy = num_stations * train_specs.get('acceleration_energy_kwh', 5.0)
        
        # Elevation energy
        elevation_energy = elevation_gain * train_specs.get('elevation_energy_kwh_m', 0.02)
        
        # Operating pattern adjustment
        pattern_factor = operating_pattern.get('efficiency_factor', 1.0)
        
        total_energy = (base_consumption * distance + stops_energy + elevation_energy) * pattern_factor
        
        return {
            'total_energy_kwh': total_energy,
            'energy_per_km_kwh': total_energy / distance if distance > 0 else 0,
            'regenerative_braking_savings': total_energy * 0.15,  # 15% savings typical
            'base_consumption': base_consumption * distance,
            'acceleration_consumption': stops_energy,
            'elevation_consumption': elevation_energy
        }
    
    def optimize_travel_times(self, station_sequence: List[Dict[str, Any]],
                            train_capabilities: Dict[str, Any],
                            track_conditions: Dict[str, Any]) -> List[float]:
        """Optimize travel times between stations"""
        travel_times = []
        
        for i in range(len(station_sequence) - 1):
            current_station = station_sequence[i]
            next_station = station_sequence[i + 1]
            
            segment_time = self._calculate_segment_time(
                current_station, next_station, train_capabilities, track_conditions
            )
            travel_times.append(segment_time)
        
        return travel_times
    
    def _calculate_segment_time(self, start_station: Dict[str, Any],
                              end_station: Dict[str, Any],
                              train_capabilities: Dict[str, Any],
                              track_conditions: Dict[str, Any]) -> float:
        """Calculate travel time for a segment between two stations"""
        distance = end_station.get('distance_km', 0) - start_station.get('distance_km', 0)
        if distance <= 0:
            return 1.0  # Minimum time
        
        # Get speed limits
        max_speed = min(
            train_capabilities.get('max_speed_kmh', 120),
            track_conditions.get('max_speed_kmh', 120)
        )
        
        # Calculate acceleration and deceleration times
        accel_time = self._calculate_acceleration_time(max_speed, train_capabilities)
        decel_time = self._calculate_deceleration_time(max_speed, train_capabilities)
        
        # Distance covered during acceleration and deceleration
        accel_distance = self._calculate_acceleration_distance(max_speed, train_capabilities)
        decel_distance = self._calculate_deceleration_distance(max_speed, train_capabilities)
        
        # Check if segment is long enough for full speed
        if distance <= (accel_distance + decel_distance):
            # Short segment - trapezoidal speed profile
            cruise_distance = 0
            cruise_time = 0
        else:
            # Long segment - full speed achieved
            cruise_distance = distance - accel_distance - decel_distance
            cruise_time = cruise_distance / max_speed * 60  # Convert to minutes
        
        total_time = accel_time + cruise_time + decel_time
        
        return max(0.5, total_time)  # Minimum 30 seconds
    
    def _calculate_acceleration_time(self, target_speed: float, 
                                  train_capabilities: Dict[str, Any]) -> float:
        """Calculate time to accelerate to target speed"""
        acceleration = train_capabilities.get('acceleration_ms2', 0.7)
        if acceleration <= 0:
            return 1.0
        
        speed_ms = target_speed / 3.6
        time = speed_ms / acceleration
        return time / 60  # Convert to minutes
    
    def _calculate_deceleration_time(self, initial_speed: float,
                                  train_capabilities: Dict[str, Any]) -> float:
        """Calculate time to decelerate from initial speed"""
        deceleration = train_capabilities.get('deceleration_ms2', 0.8)
        if deceleration <= 0:
            return 1.0
        
        speed_ms = initial_speed / 3.6
        time = speed_ms / deceleration
        return time / 60  # Convert to minutes
    
    def _calculate_acceleration_distance(self, target_speed: float,
                                      train_capabilities: Dict[str, Any]) -> float:
        """Calculate distance covered during acceleration"""
        acceleration = train_capabilities.get('acceleration_ms2', 0.7)
        if acceleration <= 0:
            return 0.1
        
        speed_ms = target_speed / 3.6
        distance = (speed_ms ** 2) / (2 * acceleration)
        return distance / 1000  # Convert to km
    
    def _calculate_deceleration_distance(self, initial_speed: float,
                                      train_capabilities: Dict[str, Any]) -> float:
        """Calculate distance covered during deceleration"""
        deceleration = train_capabilities.get('deceleration_ms2', 0.8)
        if deceleration <= 0:
            return 0.1
        
        speed_ms = initial_speed / 3.6
        distance = (speed_ms ** 2) / (2 * deceleration)
        return distance / 1000  # Convert to km
    
    def calculate_capacity_requirements(self, demand_forecast: Dict[str, Any],
                                      service_level: str = 'standard') -> Dict[str, Any]:
        """Calculate capacity requirements based on demand forecast"""
        peak_demand = demand_forecast.get('peak_hour_passengers', 1000)
        average_demand = demand_forecast.get('average_hour_passengers', 500)
        
        # Service level factors
        service_factors = {
            'premium': 0.7,   # More capacity than needed
            'standard': 0.8,   # Standard loading
            'basic': 0.9      # Higher loading
        }
        
        loading_factor = service_factors.get(service_level, 0.8)
        
        # Calculate required capacity
        required_peak_capacity = peak_demand / loading_factor
        required_average_capacity = average_demand / loading_factor
        
        return {
            'peak_capacity_required': required_peak_capacity,
            'average_capacity_required': required_average_capacity,
            'loading_factor_used': loading_factor,
            'service_level': service_level,
            'recommendations': self._generate_capacity_recommendations(required_peak_capacity)
        }
    
    def _generate_capacity_recommendations(self, peak_capacity: float) -> List[str]:
        """Generate capacity-related recommendations"""
        recommendations = []
        
        if peak_capacity > 5000:
            recommendations.extend([
                "Consider high-capacity rolling stock",
                "Implement express services to increase effective capacity",
                "Plan for future capacity expansion"
            ])
        elif peak_capacity > 2000:
            recommendations.extend([
                "Use standard capacity trains with good frequency",
                "Consider coupled train sets during peak hours"
            ])
        else:
            recommendations.append("Standard capacity planning sufficient")
        
        return recommendations