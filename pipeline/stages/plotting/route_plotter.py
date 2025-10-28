import logging
import numpy as np
from typing import Dict, Any, List, Tuple

class RoutePlotter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def plot_routes(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plot optimal rail routes based on analysis"""
        self.logger.info("Plotting rail routes")
        
        demand_data = context['demand_data']
        terrain_data = context['terrain_data']
        constraints = context['constraints']
        budget_constraints = context['budget_constraints']
        
        priority_routes = demand_data['priority_routes']
        proposed_routes = []
        
        for priority_route in priority_routes:
            route_plan = self._plot_single_route(priority_route, terrain_data, constraints, budget_constraints)
            if route_plan:
                proposed_routes.append(route_plan)
        
        context['proposed_routes'] = proposed_routes
        context['route_statistics'] = self._calculate_route_statistics(proposed_routes)
        
        self.logger.info(f"Plotted {len(proposed_routes)} routes")
        return context
    
    def _plot_single_route(self, priority_route: Dict[str, Any], terrain_data: Dict[str, Any],
                          constraints: Dict[str, Any], budget_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Plot a single rail route between cities"""
        cities = priority_route['cities']
        route_name = priority_route['route']
        distance = priority_route['distance_km']
        
        self.logger.debug(f"Plotting route: {route_name}")
        
        # Generate possible alignments
        alignments = self._generate_possible_alignments(cities, terrain_data, distance)
        
        # Select best alignment
        best_alignment = self._select_best_alignment(alignments, terrain_data, constraints, budget_constraints)
        
        if not best_alignment:
            self.logger.warning(f"No feasible alignment found for {route_name}")
            return None
        
        # Calculate route details
        route_details = self._calculate_route_details(best_alignment, priority_route, terrain_data)
        
        return {
            'name': route_name,
            'cities_served': cities,
            'alignment': best_alignment,
            'details': route_details,
            'cost_estimation': self._estimate_route_cost(route_details, terrain_data),
            'construction_timeline': self._estimate_construction_timeline(route_details, constraints),
            'priority': priority_route['priority']
        }
    
    def _generate_possible_alignments(self, cities: List[str], terrain_data: Dict[str, Any], 
                                    straight_line_distance: float) -> List[Dict[str, Any]]:
        """Generate multiple possible alignments between cities"""
        alignments = []
        
        # Generate different alignment strategies
        strategies = ['direct', 'terrain_following', 'cost_optimized', 'population_serving']
        
        for strategy in strategies:
            alignment = self._generate_alignment_strategy(strategy, cities, terrain_data, straight_line_distance)
            if alignment:
                alignments.append(alignment)
        
        return alignments
    
    def _generate_alignment_strategy(self, strategy: str, cities: List[str], terrain_data: Dict[str, Any],
                                   straight_line_distance: float) -> Dict[str, Any]:
        """Generate alignment based on specific strategy"""
        base_alignment = {
            'strategy': strategy,
            'segments': [],
            'total_distance_km': straight_line_distance * self._get_distance_multiplier(strategy),
            'elevation_changes': [],
            'terrain_crossings': []
        }
        
        if strategy == 'direct':
            # Most direct route, potentially more expensive due to terrain
            base_alignment['description'] = 'Most direct route between endpoints'
            base_alignment['cost_factor'] = 1.2  # Higher due to potential tunneling/bridging
            base_alignment['travel_time_minutes'] = straight_line_distance / 100 * 60  # 100 km/h average
            
        elif strategy == 'terrain_following':
            # Follows natural terrain, potentially longer but cheaper
            base_alignment['description'] = 'Route following natural terrain contours'
            base_alignment['cost_factor'] = 0.9
            base_alignment['travel_time_minutes'] = straight_line_distance * 1.2 / 80 * 60  # 80 km/h average
            
        elif strategy == 'cost_optimized':
            # Balances distance and construction cost
            base_alignment['description'] = 'Cost-optimized route considering terrain'
            base_alignment['cost_factor'] = 0.8
            base_alignment['travel_time_minutes'] = straight_line_distance * 1.1 / 90 * 60  # 90 km/h average
            
        elif strategy == 'population_serving':
            # Routes through intermediate population centers
            base_alignment['description'] = 'Route serving intermediate population centers'
            base_alignment['cost_factor'] = 1.1
            base_alignment['travel_time_minutes'] = straight_line_distance * 1.3 / 70 * 60  # 70 km/h average
        
        # Add terrain-specific adjustments
        terrain_difficulty = terrain_data.get('terrain_difficulty', 0.5)
        base_alignment['cost_factor'] *= (1 + terrain_difficulty * 0.5)
        
        return base_alignment
    
    def _get_distance_multiplier(self, strategy: str) -> float:
        """Get distance multiplier for different strategies"""
        multipliers = {
            'direct': 1.0,
            'terrain_following': 1.2,
            'cost_optimized': 1.1,
            'population_serving': 1.3
        }
        return multipliers.get(strategy, 1.1)
    
    def _select_best_alignment(self, alignments: List[Dict[str, Any]], terrain_data: Dict[str, Any],
                             constraints: Dict[str, Any], budget_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best alignment from possible options"""
        if not alignments:
            return None
        
        scored_alignments = []
        
        for alignment in alignments:
            score = self._score_alignment(alignment, terrain_data, constraints, budget_constraints)
            alignment['score'] = score
            scored_alignments.append(alignment)
        
        # Select alignment with highest score
        best_alignment = max(scored_alignments, key=lambda x: x['score'])
        
        return best_alignment
    
    def _score_alignment(self, alignment: Dict[str, Any], terrain_data: Dict[str, Any],
                        constraints: Dict[str, Any], budget_constraints: Dict[str, Any]) -> float:
        """Score an alignment based on multiple factors"""
        # Cost factor (lower is better)
        cost_score = 1.0 / alignment['cost_factor']
        
        # Time factor (shorter travel time is better)
        max_time = 300  # 5 hours maximum reasonable time
        time_score = 1.0 - (alignment['travel_time_minutes'] / max_time)
        
        # Terrain compatibility
        terrain_difficulty = terrain_data.get('terrain_difficulty', 0.5)
        if alignment['strategy'] == 'terrain_following' and terrain_difficulty > 0.7:
            terrain_score = 0.8
        elif alignment['strategy'] == 'direct' and terrain_difficulty < 0.3:
            terrain_score = 0.9
        else:
            terrain_score = 0.7
        
        # Budget constraints
        budget_sufficiency = constraints['budget_limitations']['budget_sufficiency']
        if budget_sufficiency < 0.5 and alignment['cost_factor'] < 1.0:
            budget_score = 1.2  # Boost low-cost options when budget is tight
        else:
            budget_score = 1.0
        
        # Weighted total score
        total_score = (
            cost_score * 0.4 +
            time_score * 0.3 +
            terrain_score * 0.3
        ) * budget_score
        
        return max(0, min(1.0, total_score))
    
    def _calculate_route_details(self, alignment: Dict[str, Any], priority_route: Dict[str, Any],
                               terrain_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed route specifications"""
        distance = alignment['total_distance_km']
        terrain_difficulty = terrain_data.get('terrain_difficulty', 0.5)
        
        # Determine track type based on distance and priority
        if distance > 200 and priority_route['priority'] in ['critical', 'high']:
            track_type = 'high_speed'
            max_speed = 250  # km/h
        elif distance > 100:
            track_type = 'regional'
            max_speed = 160  # km/h
        else:
            track_type = 'commuter'
            max_speed = 120  # km/h
        
        # Adjust for terrain
        if terrain_difficulty > 0.7:
            max_speed *= 0.8  # Reduce speed in difficult terrain
            track_type = 'mountain' if track_type == 'regional' else track_type
        
        # Calculate infrastructure requirements
        infrastructure = self._calculate_infrastructure_requirements(distance, terrain_difficulty, track_type)
        
        return {
            'track_type': track_type,
            'max_design_speed_kmh': max_speed,
            'estimated_travel_time_minutes': alignment['travel_time_minutes'],
            'infrastructure_requirements': infrastructure,
            'electrification': self._determine_electrification(track_type, distance),
            'signaling_system': self._determine_signaling_system(track_type, max_speed),
            'station_spacing_km': self._calculate_station_spacing(track_type, distance)
        }
    
    def _calculate_infrastructure_requirements(self, distance: float, terrain_difficulty: float, 
                                             track_type: str) -> Dict[str, Any]:
        """Calculate infrastructure requirements for the route"""
        # Base requirements
        requirements = {
            'earthworks_volume_cu_m': distance * 10000 * (1 + terrain_difficulty),
            'bridges_count': int(distance * 0.1 * (1 + terrain_difficulty)),
            'tunnels_km': distance * 0.05 * terrain_difficulty,
            'stations_count': max(2, int(distance / self._get_station_spacing(track_type))),
            'maintenance_depots': max(1, int(distance / 100))
        }
        
        # Adjust based on track type
        if track_type == 'high_speed':
            requirements['earthworks_volume_cu_m'] *= 1.5  # More grading for high speed
            requirements['bridges_count'] *= 1.2
            requirements['track_quality'] = 'premium'
        elif track_type == 'mountain':
            requirements['tunnels_km'] *= 2.0  # More tunneling in mountains
            requirements['bridges_count'] *= 1.5
            requirements['track_quality'] = 'mountain_special'
        else:
            requirements['track_quality'] = 'standard'
        
        return requirements
    
    def _get_station_spacing(self, track_type: str) -> float:
        """Get typical station spacing for track type"""
        spacing = {
            'commuter': 5,      # 5 km
            'regional': 15,     # 15 km
            'high_speed': 50,   # 50 km
            'mountain': 20      # 20 km
        }
        return spacing.get(track_type, 10)
    
    def _determine_electrification(self, track_type: str, distance: float) -> str:
        """Determine electrification system"""
        if track_type == 'high_speed':
            return '25kV_50Hz'  # Standard for high speed
        elif distance > 50:
            return '25kV_50Hz'  # AC for longer distances
        else:
            return '1.5kV_DC'   # DC for shorter distances
    
    def _determine_signaling_system(self, track_type: str, max_speed: float) -> str:
        """Determine appropriate signaling system"""
        if max_speed > 200:
            return 'ETCS_Level_2'
        elif max_speed > 120:
            return 'ETCS_Level_1'
        else:
            return 'traditional_block'
    
    def _calculate_station_spacing(self, track_type: str, distance: float) -> List[float]:
        """Calculate optimal station spacing along route"""
        spacing = self._get_station_spacing(track_type)
        stations_count = max(2, int(distance / spacing))
        
        # Distribute stations evenly
        segment_length = distance / (stations_count - 1)
        return [segment_length] * (stations_count - 1)
    
    def _estimate_route_cost(self, route_details: Dict[str, Any], terrain_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate cost for the route"""
        terrain_difficulty = terrain_data.get('terrain_difficulty', 0.5)
        infrastructure = route_details['infrastructure_requirements']
        distance = infrastructure['earthworks_volume_cu_m'] / 10000  # Approximate distance
        
        # Base cost per km (USD)
        base_cost_per_km = {
            'commuter': 15000000,
            'regional': 20000000,
            'high_speed': 35000000,
            'mountain': 28000000
        }
        
        track_type = route_details['track_type']
        base_cost = base_cost_per_km.get(track_type, 20000000)
        
        # Adjust for terrain difficulty
        terrain_multiplier = 1.0 + (terrain_difficulty * 0.8)
        
        # Infrastructure cost components
        earthworks_cost = infrastructure['earthworks_volume_cu_m'] * 50  # $50 per m³
        bridges_cost = infrastructure['bridges_count'] * 5000000  # $5M per bridge
        tunnels_cost = infrastructure['tunnels_km'] * 30000000  # $30M per km of tunnel
        stations_cost = infrastructure['stations_count'] * 2000000  # $2M per station
        electrification_cost = distance * 1000000  # $1M per km
        signaling_cost = distance * 500000  # $0.5M per km
        
        total_cost = (
            earthworks_cost + bridges_cost + tunnels_cost + 
            stations_cost + electrification_cost + signaling_cost
        ) * terrain_multiplier
        
        return {
            'total_estimated_cost': total_cost,
            'cost_per_km': total_cost / distance if distance > 0 else 0,
            'breakdown': {
                'earthworks': earthworks_cost,
                'bridges': bridges_cost,
                'tunnels': tunnels_cost,
                'stations': stations_cost,
                'electrification': electrification_cost,
                'signaling': signaling_cost,
                'contingency': total_cost * 0.15  # 15% contingency
            },
            'terrain_multiplier': terrain_multiplier
        }
    
    def _estimate_construction_timeline(self, route_details: Dict[str, Any], 
                                      constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate construction timeline"""
        infrastructure = route_details['infrastructure_requirements']
        temporal_constraints = constraints['temporal_limitations']
        
        # Base construction time in months
        base_months = infrastructure['tunnels_km'] * 6 + infrastructure['bridges_count'] * 3
        
        # Add time for stations and track work
        base_months += infrastructure['stations_count'] * 2
        base_months += infrastructure['earthworks_volume_cu_m'] / 100000  # 100,000 m³ per month
        
        # Apply constraints
        regulatory_delay = temporal_constraints['estimated_min_months'] * 0.3  # 30% for permitting
        
        total_months = base_months + regulatory_delay
        
        return {
            'design_phase_months': 12,
            'permitting_phase_months': regulatory_delay,
            'construction_phase_months': base_months,
            'testing_phase_months': 6,
            'total_months': total_months,
            'phased_implementation_possible': base_months > 24
        }
    
    def _calculate_route_statistics(self, proposed_routes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for all proposed routes"""
        if not proposed_routes:
            return {}
        
        total_distance = sum(route['alignment']['total_distance_km'] for route in proposed_routes)
        total_cost = sum(route['cost_estimation']['total_estimated_cost'] for route in proposed_routes)
        total_construction_time = sum(route['construction_timeline']['total_months'] for route in proposed_routes)
        
        track_types = {}
        for route in proposed_routes:
            track_type = route['details']['track_type']
            track_types[track_type] = track_types.get(track_type, 0) + 1
        
        return {
            'total_routes': len(proposed_routes),
            'total_network_distance_km': total_distance,
            'total_estimated_cost': total_cost,
            'average_cost_per_km': total_cost / total_distance if total_distance > 0 else 0,
            'total_construction_time_months': total_construction_time,
            'track_type_distribution': track_types,
            'priority_breakdown': {
                'critical': len([r for r in proposed_routes if r['priority'] == 'critical']),
                'high': len([r for r in proposed_routes if r['priority'] == 'high']),
                'medium': len([r for r in proposed_routes if r['priority'] == 'medium']),
                'low': len([r for r in proposed_routes if r['priority'] == 'low'])
            }
        }