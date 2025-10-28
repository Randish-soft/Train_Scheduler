import logging
import numpy as np
from typing import Dict, Any, List

class RailOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def optimize_rail_system(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize the entire rail system design"""
        self.logger.info("Optimizing rail system design")
        
        proposed_routes = context.get('proposed_routes', [])
        constraints = context['constraints']
        budget_constraints = context['budget_constraints']
        
        optimized_routes = []
        optimization_results = []
        
        for route in proposed_routes:
            optimized_route, optimizations = self._optimize_single_route(route, constraints, budget_constraints)
            optimized_routes.append(optimized_route)
            optimization_results.append(optimizations)
        
        context['optimized_routes'] = optimized_routes
        context['optimization_results'] = optimization_results
        context['system_optimization'] = self._optimize_system_level(optimized_routes, constraints)
        
        self.logger.info("Rail system optimization completed")
        return context
    
    def _optimize_single_route(self, route: Dict[str, Any], constraints: Dict[str, Any],
                             budget_constraints: Dict[str, Any]) -> tuple:
        """Optimize a single route for cost, efficiency, and constructability"""
        route_name = route['name']
        self.logger.debug(f"Optimizing route: {route_name}")
        
        optimizations = {
            'cost_reductions': [],
            'efficiency_improvements': [],
            'constructability_enhancements': [],
            'total_cost_savings': 0,
            'total_time_savings': 0
        }
        
        original_cost = route['cost_estimation']['total_estimated_cost']
        original_time = route['construction_timeline']['total_months']
        
        # Apply various optimization techniques
        optimizations.update(self._optimize_earthworks(route, constraints))
        optimizations.update(self._optimize_bridge_design(route))
        optimizations.update(self._optimize_station_design(route))
        optimizations.update(self._optimize_track_alignment(route))
        optimizations.update(self._optimize_electrification(route))
        
        # Calculate total savings
        cost_savings = sum(opt.get('cost_saving', 0) for opt in optimizations['cost_reductions'])
        time_savings = sum(opt.get('time_saving', 0) for opt in optimizations['constructability_enhancements'])
        
        optimizations['total_cost_savings'] = cost_savings
        optimizations['total_time_savings'] = time_savings
        
        # Apply optimizations to route
        optimized_route = self._apply_optimizations_to_route(route, optimizations)
        
        return optimized_route, optimizations
    
    def _optimize_earthworks(self, route: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize earthworks and grading"""
        optimizations = {'cost_reductions': [], 'constructability_enhancements': []}
        
        infrastructure = route['details']['infrastructure_requirements']
        earthworks_volume = infrastructure['earthworks_volume_cu_m']
        terrain_difficulty = route.get('terrain_difficulty', 0.5)
        
        # Optimization 1: Cut-and-fill balance
        if earthworks_volume > 1000000:  # Large earthworks project
            cut_fill_saving = earthworks_volume * 0.1 * 50  # 10% reduction at $50/m³
            optimizations['cost_reductions'].append({
                'type': 'cut_fill_optimization',
                'description': 'Optimized earthworks cut-and-fill balance',
                'cost_saving': cut_fill_saving,
                'impact': 'reduced earthworks volume by 10%'
            })
        
        # Optimization 2: Soil stabilization instead of removal
        if terrain_difficulty > 0.6:
            stabilization_saving = earthworks_volume * 0.05 * 30  # 5% stabilization at $30/m³ vs $50/m³ removal
            optimizations['cost_reductions'].append({
                'type': 'soil_stabilization',
                'description': 'Used soil stabilization instead of full removal',
                'cost_saving': stabilization_saving,
                'impact': 'reduced earthworks cost through stabilization'
            })
        
        # Optimization 3: Phased earthworks for large projects
        if earthworks_volume > 2000000:
            optimizations['constructability_enhancements'].append({
                'type': 'phased_earthworks',
                'description': 'Implemented phased earthworks approach',
                'time_saving': 3,  # months
                'impact': 'reduced construction timeline through better sequencing'
            })
        
        return optimizations
    
    def _optimize_bridge_design(self, route: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize bridge design and construction"""
        optimizations = {'cost_reductions': [], 'efficiency_improvements': []}
        
        infrastructure = route['details']['infrastructure_requirements']
        bridge_count = infrastructure['bridges_count']
        
        if bridge_count > 0:
            # Optimization 1: Standardized bridge designs
            standardization_saving = bridge_count * 500000  # $500K per bridge from standardization
            optimizations['cost_reductions'].append({
                'type': 'standardized_bridge_design',
                'description': 'Used standardized bridge designs',
                'cost_saving': standardization_saving,
                'impact': 'reduced design costs and construction time'
            })
            
            # Optimization 2: Prefabricated bridge elements
            prefab_saving = bridge_count * 300000  # $300K per bridge from prefabrication
            optimizations['cost_reductions'].append({
                'type': 'prefabricated_bridge_elements',
                'description': 'Used prefabricated bridge elements',
                'cost_saving': prefab_saving,
                'impact': 'reduced on-site construction time and cost'
            })
            
            # Optimization 3: Optimized span lengths
            optimizations['efficiency_improvements'].append({
                'type': 'optimized_span_lengths',
                'description': 'Optimized bridge span lengths for material efficiency',
                'impact': 'reduced material usage while maintaining structural integrity'
            })
        
        return optimizations
    
    def _optimize_station_design(self, route: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize station design and construction"""
        optimizations = {'cost_reductions': [], 'constructability_enhancements': []}
        
        stations = route.get('stations', [])
        infrastructure = route['details']['infrastructure_requirements']
        station_count = infrastructure['stations_count']
        
        if station_count > 0:
            # Optimization 1: Modular station design
            modular_saving = station_count * 300000  # $300K per station
            optimizations['cost_reductions'].append({
                'type': 'modular_station_design',
                'description': 'Used modular designs for stations',
                'cost_saving': modular_saving,
                'impact': 'reduced construction time and cost through standardization'
            })
            
            # Optimization 2: Shared facilities for nearby stations
            if station_count > 3:
                shared_facilities_saving = (station_count - 2) * 100000  # $100K per station for shared facilities
                optimizations['cost_reductions'].append({
                    'type': 'shared_facilities',
                    'description': 'Shared maintenance and operational facilities',
                    'cost_saving': shared_facilities_saving,
                    'impact': 'reduced operational costs through facility sharing'
                })
            
            # Optimization 3: Prefabricated platform elements
            platform_saving = station_count * 150000  # $150K per station
            optimizations['cost_reductions'].append({
                'type': 'prefabricated_platforms',
                'description': 'Used prefabricated platform elements',
                'cost_saving': platform_saving,
                'impact': 'reduced on-site construction time'
            })
        
        return optimizations
    
    def _optimize_track_alignment(self, route: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize track alignment for efficiency and cost"""
        optimizations = {'cost_reductions': [], 'efficiency_improvements': []}
        
        alignment = route['alignment']
        route_details = route['details']
        
        # Optimization 1: Gradient optimization
        optimizations['efficiency_improvements'].append({
            'type': 'gradient_optimization',
            'description': 'Optimized track gradients for energy efficiency',
            'impact': 'reduced energy consumption and improved operational efficiency'
        })
        
        # Optimization 2: Curve radius optimization
        curve_saving = alignment['total_distance_km'] * 10000  # $10K per km from better curves
        optimizations['cost_reductions'].append({
            'type': 'curve_optimization',
            'description': 'Optimized curve radii for cost and performance',
            'cost_saving': curve_saving,
            'impact': 'balanced construction cost with operational efficiency'
        })
        
        # Optimization 3: Transition spiral optimization
        if route_details['track_type'] == 'high_speed':
            optimizations['efficiency_improvements'].append({
                'type': 'transition_spiral_optimization',
                'description': 'Optimized transition spirals for high-speed comfort',
                'impact': 'impro passenger comfort at high speeds'
            })
        
        return optimizations
    
    def _optimize_electrification(self, route: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize electrification system"""
        optimizations = {'cost_reductions': [], 'efficiency_improvements': []}
        
        route_details = route['details']
        electrification = route_details['electrification']
        distance = route['alignment']['total_distance_km']
        
        # Optimization 1: Substation spacing optimization
        if electrification.startswith('25kV'):
            substation_saving = (distance / 50) * 1000000  # $1M per substation optimized spacing
            optimizations['cost_reductions'].append({
                'type': 'substation_spacing_optimization',
                'description': 'Optimized substation spacing for 25kV AC system',
                'cost_saving': substation_saving,
                'impact': 'reduced number of substations while maintaining power quality'
            })
        
        # Optimization 2: Energy recovery systems
        if route_details['track_type'] in ['commuter', 'regional']:
            optimizations['efficiency_improvements'].append({
                'type': 'regenerative_braking',
                'description': 'Implemented regenerative braking energy recovery',
                'impact': 'reduced energy consumption by 15-20%'
            })
        
        # Optimization 3: Smart power management
        optimizations['efficiency_improvements'].append({
            'type': 'smart_power_management',
            'description': 'Implemented smart power load management',
            'impact': 'optimized power usage and reduced peak demand charges'
        })
        
        return optimizations
    
    def _apply_optimizations_to_route(self, route: Dict[str, Any], optimizations: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimization results to the route"""
        optimized_route = route.copy()
        
        # Apply cost savings
        original_cost = route['cost_estimation']['total_estimated_cost']
        cost_savings = optimizations['total_cost_savings']
        optimized_cost = max(original_cost * 0.7, original_cost - cost_savings)  # Minimum 30% reduction
        
        optimized_route['cost_estimation']['total_estimated_cost'] = optimized_cost
        optimized_route['cost_estimation']['optimization_savings'] = cost_savings
        optimized_route['cost_estimation']['savings_percentage'] = (cost_savings / original_cost) * 100
        
        # Apply time savings
        original_time = route['construction_timeline']['total_months']
        time_savings = optimizations['total_time_savings']
        optimized_time = max(original_time * 0.8, original_time - time_savings)  # Minimum 20% reduction
        
        optimized_route['construction_timeline']['total_months'] = optimized_time
        optimized_route['construction_timeline']['optimization_savings_months'] = time_savings
        
        # Add optimization details
        optimized_route['optimizations_applied'] = len(optimizations['cost_reductions']) + len(optimizations['efficiency_improvements'])
        optimized_route['optimization_details'] = optimizations
        
        return optimized_route
    
    def _optimize_system_level(self, optimized_routes: List[Dict[str, Any]], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Perform system-level optimization across all routes"""
        total_original_cost = sum(route.get('original_cost', route['cost_estimation']['total_estimated_cost']) 
                                for route in optimized_routes)
        total_optimized_cost = sum(route['cost_estimation']['total_estimated_cost'] for route in optimized_routes)
        total_savings = total_original_cost - total_optimized_cost
        
        total_original_time = sum(route.get('original_time', route['construction_timeline']['total_months'])
                                for route in optimized_routes)
        total_optimized_time = sum(route['construction_timeline']['total_months'] for route in optimized_routes)
        total_time_savings = total_original_time - total_optimized_time
        
        # System-wide optimizations
        system_optimizations = {
            'shared_maintenance_facilities': self._optimize_maintenance_facilities(optimized_routes),
            'unified_control_center': self._optimize_control_systems(optimized_routes),
            'standardized_rolling_stock': self._optimize_rolling_stock(optimized_routes),
            'integrated_ticketing': self._optimize_ticketing_systems(optimized_routes)
        }
        
        return {
            'total_system_optimizations': len(system_optimizations),
            'cost_optimization': {
                'original_total_cost': total_original_cost,
                'optimized_total_cost': total_optimized_cost,
                'total_savings': total_savings,
                'savings_percentage': (total_savings / total_original_cost) * 100 if total_original_cost > 0 else 0
            },
            'time_optimization': {
                'original_total_time_months': total_original_time,
                'optimized_total_time_months': total_optimized_time,
                'total_time_savings_months': total_time_savings,
                'savings_percentage': (total_time_savings / total_original_time) * 100 if total_original_time > 0 else 0
            },
            'system_optimizations': system_optimizations,
            'recommendations': self._generate_system_recommendations(optimized_routes, constraints)
        }
    
    def _optimize_maintenance_facilities(self, routes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize maintenance facility placement and sharing"""
        total_distance = sum(route['alignment']['total_distance_km'] for route in routes)
        optimal_depots = max(1, int(total_distance / 150))  # One depot per 150 km
        
        return {
            'type': 'shared_maintenance_facilities',
            'description': 'Optimized maintenance depot placement and sharing',
            'estimated_savings': optimal_depots * 2000000,  # $2M per shared depot
            'impact': 'reduced capital and operational costs through facility sharing'
        }
    
    def _optimize_control_systems(self, routes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize control and signaling systems"""
        signaling_types = set(route['details']['signaling_system'] for route in routes)
        
        if len(signaling_types) > 1:
            return {
                'type': 'unified_control_center',
                'description': 'Unified control center for all routes',
                'estimated_savings': 5000000,  # $5M from unified system
                'impact': 'improved operational efficiency and reduced control system costs'
            }
        else:
            return {
                'type': 'standardized_control_systems',
                'description': 'Standardized control systems across network',
                'estimated_savings': 3000000,  # $3M from standardization
                'impact': 'reduced training and maintenance costs'
            }
    
    def _optimize_rolling_stock(self, routes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize rolling stock procurement and utilization"""
        track_types = set(route['details']['track_type'] for route in routes)
        
        if len(track_types) == 1:
            return {
                'type': 'standardized_rolling_stock',
                'description': 'Single rolling stock type for entire network',
                'estimated_savings': 10000000,  # $10M from bulk procurement
                'impact': 'reduced procurement, training, and maintenance costs'
            }
        else:
            return {
                'type': 'optimized_fleet_mix',
                'description': 'Optimized fleet mix for different route types',
                'estimated_savings': 5000000,  # $5M from right-sizing fleet
                'impact': 'balanced performance requirements with cost efficiency'
            }
    
    def _optimize_ticketing_systems(self, routes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize ticketing and fare collection systems"""
        return {
            'type': 'integrated_ticketing_system',
            'description': 'Unified ticketing system across all routes',
            'estimated_savings': 2000000,  # $2M from single system
            'impact': 'improved passenger experience and reduced operational complexity'
        }
    
    def _generate_system_recommendations(self, optimized_routes: List[Dict[str, Any]], 
                                       constraints: Dict[str, Any]) -> List[str]:
        """Generate system-level recommendations"""
        recommendations = []
        
        total_cost = sum(route['cost_estimation']['total_estimated_cost'] for route in optimized_routes)
        budget_sufficiency = constraints['budget_limitations']['budget_sufficiency']
        
        if budget_sufficiency < 0.8:
            recommendations.extend([
                "Consider phased implementation starting with highest priority routes",
                "Explore public-private partnership models for funding",
                "Prioritize routes with highest benefit-cost ratios"
            ])
        
        if len(optimized_routes) > 5:
            recommendations.append("Implement network-wide operational control center")
        
        # Check for interoperability opportunities
        electrification_systems = set(route['details']['electrification'] for route in optimized_routes)
        if len(electrification_systems) > 1:
            recommendations.append("Consider standardizing electrification system for future network expansion")
        
        return recommendations