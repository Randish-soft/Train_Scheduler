import logging
import numpy as np
from typing import Dict, Any, List

class RailyardOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize railyard placement and configuration"""
        self.logger.info("Optimizing railyard placement")
        
        optimized_routes = context.get('optimized_routes', [])
        train_selection = context.get('train_selection', {})
        terrain_data = context['terrain_data']
        
        # Analyze railyard requirements
        railyard_requirements = self._analyze_railyard_requirements(optimized_routes, train_selection)
        
        # Find optimal railyard locations
        railyard_locations = self._find_optimal_railyard_locations(railyard_requirements, optimized_routes, terrain_data)
        
        # Configure railyard facilities
        configured_railyards = self._configure_railyard_facilities(railyard_locations, railyard_requirements)
        
        context['railyard_plan'] = {
            'requirements': railyard_requirements,
            'locations': railyard_locations,
            'configured_railyards': configured_railyards,
            'optimization_metrics': self._calculate_railyard_metrics(configured_railyards, railyard_requirements)
        }
        
        self.logger.info(f"Optimized {len(configured_railyards)} railyard locations")
        return context
    
    def _analyze_railyard_requirements(self, optimized_routes: List[Dict[str, Any]], 
                                     train_selection: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze railyard requirements based on fleet and operations"""
        total_trains = 0
        train_types = {}
        route_coverage = {}
        
        for route in optimized_routes:
            route_name = route['name']
            if route_name in train_selection:
                fleet_info = train_selection[route_name]['fleet_requirements']
                train_info = train_selection[route_name]['selected_train']
                
                route_trains = fleet_info['total_fleet_size']
                total_trains += route_trains
                
                train_type = train_info['name']
                train_types[train_type] = train_types.get(train_type, 0) + route_trains
                
                route_coverage[route_name] = {
                    'trains_assigned': route_trains,
                    'train_type': train_type,
                    'route_length_km': route['alignment']['total_distance_km'],
                    'daily_trips': len(route.get('timetables', {}).get('daily_schedule', {}).get('train_departures', [])),
                    'maintenance_needs': self._calculate_route_maintenance_needs(route, fleet_info)
                }
        
        # Calculate maintenance facility requirements
        maintenance_requirements = self._calculate_maintenance_requirements(total_trains, train_types)
        
        # Calculate operational facility requirements
        operational_requirements = self._calculate_operational_requirements(route_coverage)
        
        return {
            'total_trains': total_trains,
            'train_type_distribution': train_types,
            'route_coverage': route_coverage,
            'maintenance_requirements': maintenance_requirements,
            'operational_requirements': operational_requirements,
            'total_railyards_needed': self._calculate_total_railyards_needed(total_trains, route_coverage)
        }
    
    def _calculate_route_maintenance_needs(self, route: Dict[str, Any], fleet_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate maintenance needs for a route"""
        daily_distance = fleet_info['estimated_annual_distance_km'] / 365
        trains = fleet_info['total_fleet_size']
        
        return {
            'daily_maintenance_hours': daily_distance * 0.01,  # 0.01 hours per km
            'weekly_inspections': trains,
            'monthly_maintenance': trains * 0.25,  # 25% of trains per month
            'annual_overhauls': trains * 0.08  # 8% of trains per year
        }
    
    def _calculate_maintenance_requirements(self, total_trains: int, train_types: Dict[str, int]) -> Dict[str, Any]:
        """Calculate maintenance facility requirements"""
        return {
            'daily_maintenance_bays': max(1, int(total_trains * 0.1)),  # 10% of fleet daily
            'heavy_maintenance_bays': max(1, int(total_trains * 0.02)),  # 2% of fleet
            'cleaning_facilities': max(1, int(total_trains * 0.05)),  # 5% of fleet
            'inspection_pits': max(1, int(total_trains * 0.08)),  # 8% of fleet
            'specialized_shops': self._calculate_specialized_shops(train_types)
        }
    
    def _calculate_specialized_shops(self, train_types: Dict[str, int]) -> List[Dict[str, Any]]:
        """Calculate specialized maintenance shops needed"""
        shops = []
        
        for train_type, count in train_types.items():
            if count > 10:  # Only need specialized shop for significant fleets
                shops.append({
                    'train_type': train_type,
                    'shop_size': 'large' if count > 30 else 'medium',
                    'special_equipment': self._get_special_equipment(train_type)
                })
        
        return shops
    
    def _get_special_equipment(self, train_type: str) -> List[str]:
        """Get special equipment needed for train type"""
        equipment = {
            'High Speed Train': ['wheel_lathe', 'aerodynamic_testing', 'bogie_press'],
            'Regional EMU': ['wheel_lathe', 'pantograph_testing'],
            'Commuter DMU': ['engine_test_cell', 'wheel_lathe'],
            'Mountain Train': ['cog_wheel_lathe', 'special_brake_testing']
        }
        return equipment.get(train_type, ['wheel_lathe', 'basic_testing'])
    
    def _calculate_operational_requirements(self, route_coverage: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate operational facility requirements"""
        total_routes = len(route_coverage)
        total_daily_trips = sum(info['daily_trips'] for info in route_coverage.values())
        
        return {
            'crew_bases': max(1, int(total_routes / 3)),  # One base per 3 routes
            'operations_centers': max(1, int(total_routes / 5)),  # One center per 5 routes
            'fueling_stations': sum(1 for info in route_coverage.values() if 'DMU' in info['train_type']),
            'electrification_substations': sum(1 for info in route_coverage.values() if 'DMU' not in info['train_type']),
            'storage_tracks': int(total_daily_trips * 0.3)  # Storage for 30% of daily trips
        }
    
    def _calculate_total_railyards_needed(self, total_trains: int, route_coverage: Dict[str, Any]) -> int:
        """Calculate total number of railyards needed"""
        base_yards = max(1, int(total_trains / 50))  # One yard per 50 trains
        geographic_coverage = max(1, int(len(route_coverage) / 4))  # One yard per 4 routes
        
        return max(base_yards, geographic_coverage)
    
    def _find_optimal_railyard_locations(self, requirements: Dict[str, Any], 
                                       optimized_routes: List[Dict[str, Any]],
                                       terrain_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find optimal locations for railyards"""
        num_yards = requirements['total_railyards_needed']
        route_network = self._analyze_route_network(optimized_routes)
        
        candidate_locations = self._generate_candidate_locations(route_network, terrain_data, num_yards * 3)
        
        # Score and select best locations
        scored_locations = []
        for location in candidate_locations:
            score = self._score_railyard_location(location, route_network, requirements, terrain_data)
            scored_locations.append((location, score))
        
        # Select top locations
        scored_locations.sort(key=lambda x: x[1], reverse=True)
        selected_locations = [loc for loc, score in scored_locations[:num_yards]]
        
        return selected_locations
    
    def _analyze_route_network(self, optimized_routes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the route network for optimal railyard placement"""
        junctions = {}
        terminals = {}
        major_stations = {}
        
        for route in optimized_routes:
            route_name = route['name']
            stations = route.get('stations', [])
            
            if stations:
                # Start and end terminals
                start_station = stations[0]
                end_station = stations[-1]
                
                terminals[start_station['name']] = {
                    'position_km': start_station['position_km'],
                    'type': 'terminal',
                    'routes': terminals.get(start_station['name'], {}).get('routes', []) + [route_name]
                }
                
                terminals[end_station['name']] = {
                    'position_km': end_station['position_km'],
                    'type': 'terminal',
                    'routes': terminals.get(end_station['name'], {}).get('routes', []) + [route_name]
                }
            
            # Identify major stations (hubs)
            for station in stations:
                if station['type'] in ['major', 'regional']:
                    major_stations[station['name']] = {
                        'position_km': station['position_km'],
                        'type': station['type'],
                        'routes': major_stations.get(station['name'], {}).get('routes', []) + [route_name]
                    }
        
        return {
            'junctions': junctions,
            'terminals': terminals,
            'major_stations': major_stations,
            'total_routes': len(optimized_routes),
            'network_density': self._calculate_network_density(optimized_routes)
        }
    
    def _calculate_network_density(self, optimized_routes: List[Dict[str, Any]]) -> float:
        """Calculate network density for placement optimization"""
        if not optimized_routes:
            return 0.0
        
        total_length = sum(route['alignment']['total_distance_km'] for route in optimized_routes)
        # Simplified density calculation
        return total_length / len(optimized_routes)
    
    def _generate_candidate_locations(self, route_network: Dict[str, Any], 
                                    terrain_data: Dict[str, Any], num_candidates: int) -> List[Dict[str, Any]]:
        """Generate candidate locations for railyards"""
        candidates = []
        
        # Consider terminals first
        for station_name, info in route_network['terminals'].items():
            candidates.append({
                'name': f"{station_name}_Terminal",
                'type': 'terminal_based',
                'position_km': info['position_km'],
                'serving_routes': info['routes'],
                'terrain_suitability': self._assess_terrain_suitability(info['position_km'], terrain_data),
                'land_requirement_hectares': 20.0,  # Base requirement
                'connectivity_score': len(info['routes']) * 10
            })
        
        # Consider major stations
        for station_name, info in route_network['major_stations'].items():
            if len(info['routes']) > 1:  # Only stations serving multiple routes
                candidates.append({
                    'name': f"{station_name}_Hub",
                    'type': 'hub_based',
                    'position_km': info['position_km'],
                    'serving_routes': info['routes'],
                    'terrain_suitability': self._assess_terrain_suitability(info['position_km'], terrain_data),
                    'land_requirement_hectares': 15.0,
                    'connectivity_score': len(info['routes']) * 8
                })
        
        # Generate additional candidates if needed
        while len(candidates) < num_candidates:
            # Create synthetic candidates at strategic points
            synthetic_pos = len(candidates) * 50  # Spread out every 50km
            candidates.append({
                'name': f"Synthetic_Location_{len(candidates) + 1}",
                'type': 'strategic',
                'position_km': synthetic_pos,
                'serving_routes': [],
                'terrain_suitability': 0.7,
                'land_requirement_hectares': 25.0,
                'connectivity_score': 5
            })
        
        return candidates
    
    def _assess_terrain_suitability(self, position_km: float, terrain_data: Dict[str, Any]) -> float:
        """Assess terrain suitability for railyard location"""
        # Simplified terrain assessment
        terrain_difficulty = terrain_data.get('terrain_difficulty', 0.5)
        
        # Lower difficulty is better for railyards
        suitability = 1.0 - (terrain_difficulty * 0.7)  # 70% impact from terrain difficulty
        
        # Additional factors would consider slope, drainage, etc.
        return max(0.1, min(1.0, suitability))
    
    def _score_railyard_location(self, location: Dict[str, Any], route_network: Dict[str, Any],
                               requirements: Dict[str, Any], terrain_data: Dict[str, Any]) -> float:
        """Score a railyard location based on multiple factors"""
        # Connectivity score (40%)
        connectivity_score = location['connectivity_score'] / (len(route_network['routes']) * 10) if route_network['routes'] else 0
        
        # Terrain suitability score (25%)
        terrain_score = location['terrain_suitability']
        
        # Land availability score (20%)
        land_score = 1.0 - (location['land_requirement_hectares'] / 50)  # Normalize to 50 hectares
        
        # Operational efficiency score (15%)
        operational_score = len(location['serving_routes']) / requirements['total_railyards_needed'] if requirements['total_railyards_needed'] > 0 else 0
        
        total_score = (
            connectivity_score * 0.4 +
            terrain_score * 0.25 +
            land_score * 0.2 +
            operational_score * 0.15
        )
        
        return max(0, min(1.0, total_score))
    
    def _configure_railyard_facilities(self, railyard_locations: List[Dict[str, Any]],
                                     requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Configure facilities for each railyard"""
        configured_railyards = []
        total_trains = requirements['total_trains']
        trains_per_yard = max(1, total_trains // len(railyard_locations)) if railyard_locations else 0
        
        for i, location in enumerate(railyard_locations):
            # Distribute facilities based on yard importance
            if i == 0:
                # Primary railyard gets full facilities
                yard_type = 'primary'
                facilities = self._configure_primary_railyard(trains_per_yard, requirements)
            elif i < len(railyard_locations) // 2:
                # Secondary railyards get moderate facilities
                yard_type = 'secondary'
                facilities = self._configure_secondary_railyard(trains_per_yard, requirements)
            else:
                # Tertiary railyards get basic facilities
                yard_type = 'tertiary'
                facilities = self._configure_tertiary_railyard(trains_per_yard, requirements)
            
            configured_yard = {
                **location,
                'yard_type': yard_type,
                'facilities': facilities,
                'assigned_trains': trains_per_yard,
                'estimated_cost': self._estimate_railyard_cost(facilities, location),
                'operational_capacity': self._calculate_operational_capacity(facilities)
            }
            
            configured_railyards.append(configured_yard)
        
        return configured_railyards
    
    def _configure_primary_railyard(self, trains_assigned: int, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Configure facilities for primary railyard"""
        maintenance_req = requirements['maintenance_requirements']
        operational_req = requirements['operational_requirements']
        
        return {
            'maintenance_bays': maintenance_req['daily_maintenance_bays'],
            'heavy_maintenance_bays': maintenance_req['heavy_maintenance_bays'],
            'cleaning_facilities': maintenance_req['cleaning_facilities'],
            'inspection_pits': maintenance_req['inspection_pits'],
            'specialized_shops': maintenance_req['specialized_shops'],
            'crew_facilities': operational_req['crew_bases'],
            'operations_center': True,
            'fueling_station': True,
            'training_facility': True,
            'admin_building': True,
            'storage_tracks': operational_req['storage_tracks'] // 2  # Primary gets half of storage
        }
    
    def _configure_secondary_railyard(self, trains_assigned: int, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Configure facilities for secondary railyard"""
        maintenance_req = requirements['maintenance_requirements']
        
        return {
            'maintenance_bays': max(1, maintenance_req['daily_maintenance_bays'] // 2),
            'heavy_maintenance_bays': 0,  # No heavy maintenance in secondary
            'cleaning_facilities': max(1, maintenance_req['cleaning_facilities'] // 2),
            'inspection_pits': max(1, maintenance_req['inspection_pits'] // 2),
            'specialized_shops': [],  # No specialized shops
            'crew_facilities': True,
            'operations_center': False,
            'fueling_station': True,
            'training_facility': False,
            'admin_building': True,
            'storage_tracks': max(5, trains_assigned * 2)
        }
    
    def _configure_tertiary_railyard(self, trains_assigned: int, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Configure facilities for tertiary railyard"""
        return {
            'maintenance_bays': 2,
            'heavy_maintenance_bays': 0,
            'cleaning_facilities': 1,
            'inspection_pits': 1,
            'specialized_shops': [],
            'crew_facilities': True,
            'operations_center': False,
            'fueling_station': True,
            'training_facility': False,
            'admin_building': False,
            'storage_tracks': max(3, trains_assigned)
        }
    
    def _estimate_railyard_cost(self, facilities: Dict[str, Any], location: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate cost for railyard construction"""
        base_land_cost = location['land_requirement_hectares'] * 500000  # $500K per hectare
        
        facility_costs = {
            'maintenance_bays': facilities.get('maintenance_bays', 0) * 1000000,
            'heavy_maintenance_bays': facilities.get('heavy_maintenance_bays', 0) * 2000000,
            'cleaning_facilities': facilities.get('cleaning_facilities', 0) * 500000,
            'inspection_pits': facilities.get('inspection_pits', 0) * 300000,
            'storage_tracks': facilities.get('storage_tracks', 0) * 100000,
            'buildings': self._calculate_building_costs(facilities)
        }
        
        total_construction_cost = sum(facility_costs.values())
        total_cost = base_land_cost + total_construction_cost
        
        return {
            'land_acquisition': base_land_cost,
            'construction_costs': total_construction_cost,
            'facility_breakdown': facility_costs,
            'total_estimated_cost': total_cost,
            'contingency': total_cost * 0.15  # 15% contingency
        }
    
    def _calculate_building_costs(self, facilities: Dict[str, Any]) -> int:
        """Calculate building construction costs"""
        building_cost = 0
        
        if facilities.get('operations_center', False):
            building_cost += 5000000  # $5M for operations center
        
        if facilities.get('crew_facilities', False):
            building_cost += 2000000  # $2M for crew facilities
        
        if facilities.get('training_facility', False):
            building_cost += 3000000  # $3M for training facility
        
        if facilities.get('admin_building', False):
            building_cost += 1500000  # $1.5M for admin building
        
        return building_cost
    
    def _calculate_operational_capacity(self, facilities: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate operational capacity of railyard"""
        maintenance_capacity = facilities.get('maintenance_bays', 0) * 4  # 4 trains per bay per day
        storage_capacity = facilities.get('storage_tracks', 0) * 2  # 2 trains per track
        
        return {
            'daily_maintenance_capacity': maintenance_capacity,
            'storage_capacity': storage_capacity,
            'crew_handling_capacity': facilities.get('crew_facilities', 0) * 50,  # 50 crew per facility
            'overall_utilization_limit': 0.8  # 80% utilization for efficiency
        }
    
    def _calculate_railyard_metrics(self, configured_railyards: List[Dict[str, Any]],
                                  requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall railyard optimization metrics"""
        total_cost = sum(yard['estimated_cost']['total_estimated_cost'] for yard in configured_railyards)
        total_capacity = sum(yard['operational_capacity']['daily_maintenance_capacity'] for yard in configured_railyards)
        required_capacity = requirements['total_trains'] * 0.1  # 10% of fleet needs daily maintenance
        
        capacity_sufficiency = total_capacity / required_capacity if required_capacity > 0 else 1.0
        
        return {
            'total_railyards': len(configured_railyards),
            'total_estimated_cost': total_cost,
            'maintenance_capacity_sufficiency': capacity_sufficiency,
            'average_cost_per_railyard': total_cost / len(configured_railyards) if configured_railyards else 0,
            'coverage_efficiency': self._calculate_coverage_efficiency(configured_railyards, requirements),
            'recommendations': self._generate_railyard_recommendations(capacity_sufficiency, configured_railyards)
        }
    
    def _calculate_coverage_efficiency(self, configured_railyards: List[Dict[str, Any]],
                                    requirements: Dict[str, Any]) -> float:
        """Calculate how efficiently railyards cover the network"""
        total_routes = requirements.get('route_coverage', {}).get('total_routes', 0)
        if total_routes == 0:
            return 0.0
        
        covered_routes = set()
        for yard in configured_railyards:
            covered_routes.update(yard.get('serving_routes', []))
        
        coverage_ratio = len(covered_routes) / total_routes
        return min(1.0, coverage_ratio)
    
    def _generate_railyard_recommendations(self, capacity_sufficiency: float,
                                         configured_railyards: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for railyard optimization"""
        recommendations = []
        
        if capacity_sufficiency < 0.8:
            recommendations.extend([
                "Consider expanding maintenance facilities at primary railyards",
                "Implement shift operations to increase maintenance capacity",
                "Plan for future railyard expansion as fleet grows"
            ])
        
        if len(configured_railyards) > 5:
            recommendations.append("Consider consolidating smaller railyards for efficiency")
        
        primary_yards = [yard for yard in configured_railyards if yard['yard_type'] == 'primary']
        if len(primary_yards) == 0:
            recommendations.append("Designate at least one primary railyard with full maintenance capabilities")
        
        return recommendations