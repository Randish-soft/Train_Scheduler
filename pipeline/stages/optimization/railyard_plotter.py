import logging
import numpy as np
from typing import Dict, Any, List

class RailyardPlotter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def plot_railyards(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plot railyard layouts and configurations"""
        self.logger.info("Plotting railyard layouts")
        
        railyard_plan = context.get('railyard_plan', {})
        optimized_routes = context.get('optimized_routes', [])
        
        if not railyard_plan.get('configured_railyards'):
            self.logger.warning("No railyards to plot")
            return context
        
        plotted_railyards = []
        
        for railyard in railyard_plan['configured_railyards']:
            layout = self._plot_single_railyard(railyard, optimized_routes)
            plotted_railyards.append({
                **railyard,
                'layout': layout,
                'integration_plan': self._plan_network_integration(railyard, optimized_routes)
            })
        
        context['plotted_railyards'] = plotted_railyards
        context['railyard_layout_analysis'] = self._analyze_railyard_layouts(plotted_railyards)
        
        self.logger.info(f"Plotted layouts for {len(plotted_railyards)} railyards")
        return context
    
    def _plot_single_railyard(self, railyard: Dict[str, Any], optimized_routes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Plot layout for a single railyard"""
        yard_type = railyard['yard_type']
        facilities = railyard['facilities']
        
        self.logger.debug(f"Plotting layout for railyard: {railyard['name']}")
        
        # Generate base layout based on yard type
        if yard_type == 'primary':
            layout = self._plot_primary_railyard(railyard, facilities)
        elif yard_type == 'secondary':
            layout = self._plot_secondary_railyard(railyard, facilities)
        else:
            layout = self._plot_tertiary_railyard(railyard, facilities)
        
        # Add yard-specific configurations
        layout['track_arrangement'] = self._design_track_arrangement(railyard, facilities)
        layout['building_placement'] = self._plan_building_placement(railyard, facilities)
        layout['service_areas'] = self._design_service_areas(railyard, facilities)
        layout['access_routes'] = self._plan_access_routes(railyard, optimized_routes)
        
        return layout
    
    def _plot_primary_railyard(self, railyard: Dict[str, Any], facilities: Dict[str, Any]) -> Dict[str, Any]:
        """Plot layout for primary railyard"""
        land_area = railyard['land_requirement_hectares']
        
        return {
            'layout_type': 'primary_comprehensive',
            'total_area_hectares': land_area,
            'zones': {
                'maintenance_zone': {
                    'area_hectares': land_area * 0.4,
                    'facilities': ['heavy_maintenance', 'light_maintenance', 'inspection_pits', 'specialized_shops'],
                    'track_count': facilities.get('maintenance_bays', 0) + facilities.get('heavy_maintenance_bays', 0)
                },
                'storage_zone': {
                    'area_hectares': land_area * 0.3,
                    'facilities': ['storage_tracks', 'cleaning_bays'],
                    'track_count': facilities.get('storage_tracks', 0)
                },
                'operations_zone': {
                    'area_hectares': land_area * 0.2,
                    'facilities': ['operations_center', 'crew_facilities', 'admin_building', 'training_facility'],
                    'building_count': 4
                },
                'service_zone': {
                    'area_hectares': land_area * 0.1,
                    'facilities': ['fueling_station', 'warehouse', 'utility_building'],
                    'building_count': 3
                }
            },
            'circulation_pattern': 'double_ended',
            'expansion_capability': True,
            'security_level': 'high'
        }
    
    def _plot_secondary_railyard(self, railyard: Dict[str, Any], facilities: Dict[str, Any]) -> Dict[str, Any]:
        """Plot layout for secondary railyard"""
        land_area = railyard['land_requirement_hectares']
        
        return {
            'layout_type': 'secondary_standard',
            'total_area_hectares': land_area,
            'zones': {
                'maintenance_zone': {
                    'area_hectares': land_area * 0.5,
                    'facilities': ['light_maintenance', 'inspection_pits', 'cleaning_bays'],
                    'track_count': facilities.get('maintenance_bays', 0)
                },
                'storage_zone': {
                    'area_hectares': land_area * 0.3,
                    'facilities': ['storage_tracks'],
                    'track_count': facilities.get('storage_tracks', 0)
                },
                'operations_zone': {
                    'area_hectares': land_area * 0.2,
                    'facilities': ['crew_facilities', 'admin_building'],
                    'building_count': 2
                }
            },
            'circulation_pattern': 'single_ended',
            'expansion_capability': True,
            'security_level': 'medium'
        }
    
    def _plot_tertiary_railyard(self, railyard: Dict[str, Any], facilities: Dict[str, Any]) -> Dict[str, Any]:
        """Plot layout for tertiary railyard"""
        land_area = railyard['land_requirement_hectares']
        
        return {
            'layout_type': 'tertiary_basic',
            'total_area_hectares': land_area,
            'zones': {
                'combined_zone': {
                    'area_hectares': land_area * 0.8,
                    'facilities': ['maintenance_bays', 'storage_tracks', 'cleaning_bays'],
                    'track_count': facilities.get('maintenance_bays', 0) + facilities.get('storage_tracks', 0)
                },
                'support_zone': {
                    'area_hectares': land_area * 0.2,
                    'facilities': ['crew_facilities', 'utility_building'],
                    'building_count': 2
                }
            },
            'circulation_pattern': 'simple_loop',
            'expansion_capability': False,
            'security_level': 'basic'
        }
    
    def _design_track_arrangement(self, railyard: Dict[str, Any], facilities: Dict[str, Any]) -> Dict[str, Any]:
        """Design track arrangement for the railyard"""
        storage_tracks = facilities.get('storage_tracks', 0)
        maintenance_bays = facilities.get('maintenance_bays', 0)
        inspection_pits = facilities.get('inspection_pits', 0)
        
        total_tracks = storage_tracks + maintenance_bays + inspection_pights
        
        return {
            'total_tracks': total_tracks,
            'storage_tracks': storage_tracks,
            'maintenance_tracks': maintenance_bays,
            'inspection_tracks': inspection_pits,
            'arrangement_type': 'ladder' if total_tracks > 10 else 'fan',
            'turnout_count': total_tracks * 2,  # Rough estimate
            'crossover_locations': ['entrance', 'mid_yard'] if total_tracks > 5 else ['entrance'],
            'circulation_efficiency': self._calculate_circulation_efficiency(total_tracks, railyard['yard_type'])
        }
    
    def _calculate_circulation_efficiency(self, total_tracks: int, yard_type: str) -> float:
        """Calculate circulation efficiency score"""
        base_efficiency = 0.8  # Base efficiency
        
        # Adjust based on yard type and size
        if yard_type == 'primary':
            type_bonus = 0.1
        elif yard_type == 'secondary':
            type_bonus = 0.05
        else:
            type_bonus = 0.0
        
        # Size adjustment (larger yards can be less efficient)
        size_penalty = min(0.2, (total_tracks - 10) * 0.01) if total_tracks > 10 else 0
        
        return max(0.5, min(1.0, base_efficiency + type_bonus - size_penalty))
    
    def _plan_building_placement(self, railyard: Dict[str, Any], facilities: Dict[str, Any]) -> Dict[str, Any]:
        """Plan building placement within railyard"""
        buildings = []
        
        if facilities.get('operations_center', False):
            buildings.append({
                'type': 'operations_center',
                'size_sq_m': 2000,
                'location': 'front_center',
                'floors': 2,
                'function': 'network_control'
            })
        
        if facilities.get('crew_facilities', False):
            buildings.append({
                'type': 'crew_building',
                'size_sq_m': 1500,
                'location': 'front_left',
                'floors': 1,
                'function': 'crew_rest_reporting'
            })
        
        if facilities.get('admin_building', False):
            buildings.append({
                'type': 'administration',
                'size_sq_m': 1000,
                'location': 'front_right',
                'floors': 1,
                'function': 'management_offices'
            })
        
        if facilities.get('training_facility', False):
            buildings.append({
                'type': 'training_center',
                'size_sq_m': 2500,
                'location': 'side_rear',
                'floors': 1,
                'function': 'crew_training'
            })
        
        return {
            'buildings': buildings,
            'total_building_area_sq_m': sum(bld['size_sq_m'] for bld in buildings),
            'placement_strategy': 'functional_clustering',
            'access_roads': len(buildings) * 2  # Rough estimate
        }
    
    def _design_service_areas(self, railyard: Dict[str, Any], facilities: Dict[str, Any]) -> Dict[str, Any]:
        """Design service and utility areas"""
        service_areas = []
        
        if facilities.get('fueling_station', False):
            service_areas.append({
                'type': 'fueling_station',
                'size_sq_m': 500,
                'location': 'service_zone',
                'capacity': 'dual_track',
                'safety_features': ['fire_suppression', 'containment_berm']
            })
        
        service_areas.append({
            'type': 'utility_building',
            'size_sq_m': 300,
            'location': 'service_zone',
            'functions': ['electrical_substation', 'water_pump', 'compressed_air']
        })
        
        if facilities.get('cleaning_facilities', 0) > 0:
            service_areas.append({
                'type': 'cleaning_bays',
                'size_sq_m': 800,
                'location': 'maintenance_zone',
                'capacity': facilities['cleaning_facilities'],
                'equipment': ['high_pressure_washers', 'water_recycling']
            })
        
        return {
            'service_areas': service_areas,
            'total_service_area_sq_m': sum(area['size_sq_m'] for area in service_areas),
            'utility_connections': ['power', 'water', 'drainage', 'communications'],
            'environmental_considerations': self._assess_environmental_considerations(railyard)
        }
    
    def _assess_environmental_considerations(self, railyard: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Assess environmental considerations for railyard"""
        considerations = []
        
        # Basic environmental considerations
        considerations.extend([
            {
                'aspect': 'stormwater_management',
                'requirements': ['detention_pond', 'oil_separators', 'filtration_system'],
                'compliance_level': 'high'
            },
            {
                'aspect': 'noise_mitigation',
                'requirements': ['sound_barriers', 'operational_restrictions'],
                'compliance_level': 'medium'
            },
            {
                'aspect': 'hazardous_materials',
                'requirements': ['containment_storage', 'spill_response_plan'],
                'compliance_level': 'high'
            }
        ])
        
        # Add terrain-specific considerations
        if railyard['terrain_suitability'] < 0.7:
            considerations.append({
                'aspect': 'earthworks_stabilization',
                'requirements': ['retaining_walls', 'drainage_improvements'],
                'compliance_level': 'medium'
            })
        
        return considerations
    
    def _plan_access_routes(self, railyard: Dict[str, Any], optimized_routes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Plan access routes connecting railyard to main lines"""
        serving_routes = railyard.get('serving_routes', [])
        
        connections = []
        for route_name in serving_routes:
            route = next((r for r in optimized_routes if r['name'] == route_name), None)
            if route:
                connections.append({
                    'route_name': route_name,
                    'connection_type': 'direct' if railyard['type'] == 'terminal_based' else 'spur',
                    'connection_length_km': self._estimate_connection_length(railyard, route),
                    'track_configuration': 'single_track',  # Most connections are single track
                    'gradient_requirements': 'minimal'  # Yards prefer flat connections
                })
        
        return {
            'main_line_connections': connections,
            'road_access': self._plan_road_access(railyard),
            'employee_access': ['main_gate', 'pedestrian_entrance'],
            'emergency_access': ['all_weather_roads', 'fire_lanes']
        }
    
    def _estimate_connection_length(self, railyard: Dict[str, Any], route: Dict[str, Any]) -> float:
        """Estimate connection length from railyard to main line"""
        # Simplified estimation
        if railyard['type'] == 'terminal_based':
            return 0.5  # 500 meters for terminal connections
        elif railyard['type'] == 'hub_based':
            return 1.0  # 1 km for hub connections
        else:
            return 2.0  # 2 km for strategic locations
    
    def _plan_road_access(self, railyard: Dict[str, Any]) -> Dict[str, Any]:
        """Plan road access to railyard"""
        yard_type = railyard['yard_type']
        
        if yard_type == 'primary':
            return {
                'access_roads': 2,
                'road_type': 'dual_carriageway',
                'parking_spaces': 200,
                'loading_bays': 4,
                'security_gates': 2
            }
        elif yard_type == 'secondary':
            return {
                'access_roads': 1,
                'road_type': 'single_carriageway',
                'parking_spaces': 100,
                'loading_bays': 2,
                'security_gates': 1
            }
        else:
            return {
                'access_roads': 1,
                'road_type': 'access_road',
                'parking_spaces': 50,
                'loading_bays': 1,
                'security_gates': 1
            }
    
    def _plan_network_integration(self, railyard: Dict[str, Any], optimized_routes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Plan how railyard integrates with the network"""
        serving_routes = railyard.get('serving_routes', [])
        
        integration_plan = {
            'served_routes': serving_routes,
            'operational_integration': self._plan_operational_integration(railyard, serving_routes),
            'maintenance_scheduling': self._plan_maintenance_scheduling(railyard, serving_routes),
            'crew_management': self._plan_crew_management(railyard)
        }
        
        return integration_plan
    
    def _plan_operational_integration(self, railyard: Dict[str, Any], serving_routes: List[str]) -> Dict[str, Any]:
        """Plan operational integration with network"""
        return {
            'train_movements_per_day': len(serving_routes) * 10,  # Rough estimate
            'peak_operation_hours': ['06:00-09:00', '16:00-19:00'],
            'coordination_requirements': ['central_control', 'route_scheduling'],
            'communication_systems': ['radio', 'data_network', 'cctv']
        }
    
    def _plan_maintenance_scheduling(self, railyard: Dict[str, Any], serving_routes: List[str]) -> Dict[str, Any]:
        """Plan maintenance scheduling coordination"""
        return {
            'preventive_maintenance_slots': len(serving_routes) * 2,
            'emergency_maintenance_capacity': 2,
            'coordination_with_operations': 'integrated_scheduling',
            'maintenance_windows': ['overnight', 'weekend_peaks']
        }
    
    def _plan_crew_management(self, railyard: Dict[str, Any]) -> Dict[str, Any]:
        """Plan crew management and scheduling"""
        yard_type = railyard['yard_type']
        
        if yard_type == 'primary':
            crew_capacity = 100
            shift_pattern = 'three_shifts'
        elif yard_type == 'secondary':
            crew_capacity = 50
            shift_pattern = 'two_shifts'
        else:
            crew_capacity = 20
            shift_pattern = 'single_shift'
        
        return {
            'crew_capacity': crew_capacity,
            'shift_pattern': shift_pattern,
            'facilities': ['locker_rooms', 'break_rooms', 'training_rooms'],
            'transport_arrangements': ['crew_buses', 'parking']
        }
    
    def _analyze_railyard_layouts(self, plotted_railyards: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall railyard layout efficiency"""
        total_area = sum(yard['layout']['total_area_hectares'] for yard in plotted_railyards)
        total_tracks = sum(yard['layout']['track_arrangement']['total_tracks'] for yard in plotted_railyards)
        
        primary_yards = [yard for yard in plotted_railyards if yard['yard_type'] == 'primary']
        secondary_yards = [yard for yard in plotted_railyards if yard['yard_type'] == 'secondary']
        tertiary_yards = [yard for yard in plotted_railyards if yard['yard_type'] == 'tertiary']
        
        avg_efficiency = np.mean([
            yard['layout']['track_arrangement']['circulation_efficiency'] 
            for yard in plotted_railyards
        ]) if plotted_railyards else 0
        
        return {
            'total_railyards_plotted': len(plotted_railyards),
            'total_land_area_hectares': total_area,
            'total_tracks': total_tracks,
            'yard_type_distribution': {
                'primary': len(primary_yards),
                'secondary': len(secondary_yards),
                'tertiary': len(tertiary_yards)
            },
            'average_circulation_efficiency': avg_efficiency,
            'layout_optimization_score': self._calculate_layout_optimization_score(plotted_railyards),
            'recommendations': self._generate_layout_recommendations(plotted_railyards)
        }
    
    def _calculate_layout_optimization_score(self, plotted_railyards: List[Dict[str, Any]]) -> float:
        """Calculate overall layout optimization score"""
        if not plotted_railyards:
            return 0.0
        
        scores = []
        for yard in plotted_railyards:
            layout = yard['layout']
            
            # Efficiency score
            efficiency = layout['track_arrangement']['circulation_efficiency']
            
            # Space utilization score
            total_area = layout['total_area_hectares']
            utilized_area = sum(zone['area_hectares'] for zone in layout['zones'].values())
            utilization = utilized_area / total_area if total_area > 0 else 0
            
            # Integration score
            integration = len(yard['integration_plan']['served_routes']) / 5  # Normalize to 5 routes
            
            yard_score = (efficiency * 0.4 + utilization * 0.3 + integration * 0.3)
            scores.append(yard_score)
        
        return sum(scores) / len(scores)
    
    def _generate_layout_recommendations(self, plotted_railyards: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for layout optimization"""
        recommendations = []
        
        low_efficiency_yards = [
            yard for yard in plotted_railyards 
            if yard['layout']['track_arrangement']['circulation_efficiency'] < 0.7
        ]
        
        if low_efficiency_yards:
            recommendations.append(
                f"Improve circulation patterns at {len(low_efficiency_yards)} railyards"
            )
        
        primary_yards = [yard for yard in plotted_railyards if yard['yard_type'] == 'primary']
        if len(primary_yards) == 0 and len(plotted_railyards) > 3:
            recommendations.append("Designate at least one primary railyard for major maintenance")
        
        # Check for expansion capabilities
        non_expandable = [
            yard for yard in plotted_railyards 
            if not yard['layout']['expansion_capability'] and yard['yard_type'] != 'tertiary'
        ]
        if non_expandable:
            recommendations.append("Ensure expansion capability for non-tertiary railyards")
        
        return recommendations