import logging
import numpy as np
from typing import Dict, Any, List

class TrainSelector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.train_fleet = self._initialize_train_fleet()
    
    def select_trains(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate trains for each route"""
        self.logger.info("Selecting trains for routes")
        
        optimized_routes = context.get('optimized_routes', [])
        demand_data = context['demand_data']
        budget_constraints = context['budget_constraints']
        
        selected_trains = {}
        
        for route in optimized_routes:
            route_name = route['name']
            train_selection = self._select_trains_for_route(route, demand_data, budget_constraints)
            selected_trains[route_name] = train_selection
        
        context['train_selection'] = selected_trains
        context['fleet_analysis'] = self._analyze_fleet_requirements(selected_trains, budget_constraints)
        
        self.logger.info(f"Selected trains for {len(selected_trains)} routes")
        return context
    
    def _initialize_train_fleet(self) -> Dict[str, Any]:
        """Initialize available train types"""
        return {
            'regional_emu': {
                'name': 'Regional EMU',
                'type': 'electric_multiple_unit',
                'max_speed_kmh': 160,
                'acceleration_ms2': 0.7,
                'deceleration_ms2': 0.8,
                'capacity_seated': 300,
                'capacity_standing': 200,
                'train_length_m': 100,
                'power_consumption_kwh_km': 15,
                'procurement_cost_usd': 30000000,
                'maintenance_cost_usd_km': 2.5,
                'lifespan_years': 30,
                'suitable_track_types': ['regional', 'commuter'],
                'electrification': ['25kV_50Hz', '1.5kV_DC']
            },
            'high_speed_train': {
                'name': 'High Speed Train',
                'type': 'electric_locomotive',
                'max_speed_kmh': 300,
                'acceleration_ms2': 0.5,
                'deceleration_ms2': 0.6,
                'capacity_seated': 400,
                'capacity_standing': 50,
                'train_length_m': 200,
                'power_consumption_kwh_km': 25,
                'procurement_cost_usd': 50000000,
                'maintenance_cost_usd_km': 4.0,
                'lifespan_years': 25,
                'suitable_track_types': ['high_speed'],
                'electrification': ['25kV_50Hz']
            },
            'commuter_dmu': {
                'name': 'Commuter DMU',
                'type': 'diesel_multiple_unit',
                'max_speed_kmh': 120,
                'acceleration_ms2': 0.6,
                'deceleration_ms2': 0.7,
                'capacity_seated': 250,
                'capacity_standing': 150,
                'train_length_m': 80,
                'fuel_consumption_l_km': 3.5,
                'procurement_cost_usd': 20000000,
                'maintenance_cost_usd_km': 3.0,
                'lifespan_years': 25,
                'suitable_track_types': ['commuter', 'regional'],
                'electrification': ['none']
            },
            'mountain_train': {
                'name': 'Mountain Train',
                'type': 'electric_multiple_unit',
                'max_speed_kmh': 100,
                'acceleration_ms2': 0.4,
                'deceleration_ms2': 0.5,
                'capacity_seated': 200,
                'capacity_standing': 100,
                'train_length_m': 60,
                'power_consumption_kwh_km': 20,
                'procurement_cost_usd': 35000000,
                'maintenance_cost_usd_km': 3.5,
                'lifespan_years': 30,
                'suitable_track_types': ['mountain'],
                'electrification': ['25kV_50Hz', '1.5kV_DC'],
                'special_features': ['cog_wheel', 'strong_brakes']
            }
        }
    
    def _select_trains_for_route(self, route: Dict[str, Any], demand_data: Dict[str, Any],
                               budget_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate trains for a specific route"""
        route_name = route['name']
        route_details = route['details']
        stations = route.get('stations', [])
        
        self.logger.debug(f"Selecting trains for route: {route_name}")
        
        # Get route requirements
        track_type = route_details['track_type']
        max_speed = route_details['max_design_speed_kmh']
        electrification = route_details['electrification']
        distance = route['alignment']['total_distance_km']
        
        # Calculate passenger demand
        passenger_demand = self._calculate_route_demand(route, demand_data, stations)
        
        # Find suitable train types
        suitable_trains = self._find_suitable_trains(track_type, max_speed, electrification)
        
        if not suitable_trains:
            self.logger.warning(f"No suitable trains found for route {route_name}")
            return {}
        
        # Select best train based on multiple factors
        best_train = self._select_best_train(suitable_trains, passenger_demand, distance, budget_constraints)
        
        # Calculate fleet requirements
        fleet_requirements = self._calculate_fleet_requirements(best_train, passenger_demand, route)
        
        return {
            'selected_train': best_train,
            'fleet_requirements': fleet_requirements,
            'operational_costs': self._calculate_operational_costs(best_train, fleet_requirements, distance),
            'passenger_capacity_analysis': self._analyze_capacity(best_train, passenger_demand)
        }
    
    def _calculate_route_demand(self, route: Dict[str, Any], demand_data: Dict[str, Any],
                              stations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate passenger demand for the route"""
        route_name = route['name']
        
        # Find demand data for this route
        demand_corridors = demand_data.get('demand_corridors', [])
        route_demand = next((corridor for corridor in demand_corridors if corridor['route'] == route_name), None)
        
        if route_demand:
            base_demand = route_demand['estimated_ridership']['daily_riders']
        else:
            # Estimate based on station populations
            total_station_demand = sum(station.get('estimated_daily_passengers', 0) for station in stations)
            base_demand = total_station_demand * 0.3  # Assume 30% use train
        
        # Calculate peak demand
        peak_hour_demand = base_demand * 0.15  # 15% of daily in peak hour
        average_load = base_demand / 16  # Assume 16 operating hours
        
        return {
            'daily_riders': base_demand,
            'peak_hour_riders': peak_hour_demand,
            'average_hourly_riders': average_load,
            'seasonal_variation': 1.2,  # 20% higher in peak season
            'growth_rate_annual': 0.03  # 3% annual growth
        }
    
    def _find_suitable_trains(self, track_type: str, max_speed: float, electrification: str) -> List[Dict[str, Any]]:
        """Find train types suitable for route characteristics"""
        suitable_trains = []
        
        for train_id, train_spec in self.train_fleet.items():
            # Check track type compatibility
            if track_type not in train_spec['suitable_track_types']:
                continue
            
            # Check speed capability
            if train_spec['max_speed_kmh'] < max_speed * 0.8:  # Need 80% of max speed capability
                continue
            
            # Check electrification compatibility
            if electrification not in train_spec['electrification']:
                continue
            
            suitable_trains.append({**train_spec, 'id': train_id})
        
        return suitable_trains
    
    def _select_best_train(self, suitable_trains: List[Dict[str, Any]], passenger_demand: Dict[str, Any],
                         distance: float, budget_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best train from suitable options"""
        scored_trains = []
        
        for train in suitable_trains:
            score = self._score_train(train, passenger_demand, distance, budget_constraints)
            scored_trains.append((train, score))
        
        # Select train with highest score
        best_train, best_score = max(scored_trains, key=lambda x: x[1])
        
        return {**best_train, 'selection_score': best_score}
    
    def _score_train(self, train: Dict[str, Any], passenger_demand: Dict[str, Any],
                   distance: float, budget_constraints: Dict[str, Any]) -> float:
        """Score a train based on multiple factors"""
        peak_demand = passenger_demand['peak_hour_riders']
        train_capacity = train['capacity_seated'] + train['capacity_standing']
        
        # Capacity score (closer to peak demand is better)
        capacity_ratio = train_capacity / peak_demand if peak_demand > 0 else 1.0
        if capacity_ratio < 0.8:
            capacity_score = 0.3  # Under capacity
        elif capacity_ratio > 1.5:
            capacity_score = 0.7  # Over capacity
        else:
            capacity_score = 1.0  # Good match
        
        # Cost score (lower cost is better)
        budget = budget_constraints['rolling_stock_allocation']
        estimated_fleet_cost = self._estimate_fleet_cost(train, passenger_demand, distance)
        cost_ratio = estimated_fleet_cost / budget if budget > 0 else 1.0
        cost_score = 1.0 / (1.0 + cost_ratio)
        
        # Efficiency score
        if 'power_consumption_kwh_km' in train:
            efficiency = 1.0 / (train['power_consumption_kwh_km'] / 20)  # Normalize to 20 kWh/km
        else:
            efficiency = 0.7  # Diesel trains less efficient
        
        # Comfort score (based on train type)
        comfort_scores = {
            'high_speed_train': 0.9,
            'regional_emu': 0.8,
            'commuter_dmu': 0.6,
            'mountain_train': 0.7
        }
        comfort_score = comfort_scores.get(train['id'], 0.7)
        
        # Weighted total score
        total_score = (
            capacity_score * 0.4 +
            cost_score * 0.3 +
            efficiency * 0.2 +
            comfort_score * 0.1
        )
        
        return total_score
    
    def _estimate_fleet_cost(self, train: Dict[str, Any], passenger_demand: Dict[str, Any],
                           distance: float) -> float:
        """Estimate total fleet cost for the route"""
        peak_demand = passenger_demand['peak_hour_riders']
        train_capacity = train['capacity_seated'] + train['capacity_standing']
        
        # Estimate number of trains needed
        trains_needed = max(1, int(peak_demand / train_capacity * 1.2))  # 20% buffer
        
        # Procurement cost
        procurement_cost = trains_needed * train['procurement_cost_usd']
        
        # 10-year operational cost
        annual_distance = distance * 4 * 365  # 4 trips per day
        maintenance_cost = trains_needed * annual_distance * train['maintenance_cost_usd_km'] * 10
        
        return procurement_cost + maintenance_cost
    
    def _calculate_fleet_requirements(self, train: Dict[str, Any], passenger_demand: Dict[str, Any],
                                    route: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed fleet requirements"""
        peak_demand = passenger_demand['peak_hour_riders']
        train_capacity = train['capacity_seated'] + train['capacity_standing']
        route_distance = route['alignment']['total_distance_km']
        travel_time = route['details']['estimated_travel_time_minutes'] / 60  # Convert to hours
        
        # Calculate minimum trains for service
        min_trains_peak = max(1, int(peak_demand / train_capacity))
        
        # Calculate trains for frequency
        target_frequency_peak = 4  # trains per hour in peak
        cycle_time = (travel_time * 2) + 0.5  # Round trip + turnaround
        trains_for_frequency = int(target_frequency_peak * cycle_time)
        
        # Required fleet size is maximum of both calculations
        required_fleet = max(min_trains_peak, trains_for_frequency)
        
        # Reserve trains for maintenance
        maintenance_reserve = max(1, int(required_fleet * 0.2))  # 20% reserve
        
        total_fleet = required_fleet + maintenance_reserve
        
        return {
            'minimum_operational_trains': required_fleet,
            'maintenance_reserve': maintenance_reserve,
            'total_fleet_size': total_fleet,
            'peak_frequency_trains_per_hour': target_frequency_peak,
            'off_peak_frequency_trains_per_hour': max(1, target_frequency_peak // 2),
            'average_utilization_hours_day': 16,
            'estimated_annual_distance_km': route_distance * 4 * 365 * required_fleet
        }
    
    def _calculate_operational_costs(self, train: Dict[str, Any], fleet_requirements: Dict[str, Any],
                                   distance: float) -> Dict[str, Any]:
        """Calculate operational costs for the selected train"""
        total_fleet = fleet_requirements['total_fleet_size']
        annual_distance = fleet_requirements['estimated_annual_distance_km']
        
        # Energy costs
        if 'power_consumption_kwh_km' in train:
            energy_consumption = annual_distance * train['power_consumption_kwh_km']
            energy_cost = energy_consumption * 0.12  # $0.12 per kWh
        else:
            fuel_consumption = annual_distance * train['fuel_consumption_l_km']
            energy_cost = fuel_consumption * 1.2  # $1.2 per liter
        
        # Maintenance costs
        maintenance_cost = annual_distance * train['maintenance_cost_usd_km']
        
        # Staff costs (estimate)
        staff_cost = total_fleet * 200000  # $200K per train per year for staff
        
        # Total annual operational cost
        total_annual_cost = energy_cost + maintenance_cost + staff_cost
        
        return {
            'energy_cost_annual': energy_cost,
            'maintenance_cost_annual': maintenance_cost,
            'staff_cost_annual': staff_cost,
            'total_operational_cost_annual': total_annual_cost,
            'cost_per_passenger_km': total_annual_cost / (annual_distance * 100)  # Estimate 100 passengers per train
        }
    
    def _analyze_capacity(self, train: Dict[str, Any], passenger_demand: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze capacity utilization"""
        train_capacity = train['capacity_seated'] + train['capacity_standing']
        peak_demand = passenger_demand['peak_hour_riders']
        average_demand = passenger_demand['average_hourly_riders']
        
        peak_utilization = min(1.0, peak_demand / train_capacity) if train_capacity > 0 else 1.0
        average_utilization = min(1.0, average_demand / train_capacity) if train_capacity > 0 else 1.0
        
        return {
            'train_capacity': train_capacity,
            'peak_hour_utilization': peak_utilization,
            'average_utilization': average_utilization,
            'capacity_adequacy': 'adequate' if peak_utilization < 0.9 else 'insufficient',
            'recommendations': self._generate_capacity_recommendations(peak_utilization, average_utilization)
        }
    
    def _generate_capacity_recommendations(self, peak_utilization: float, average_utilization: float) -> List[str]:
        """Generate capacity-related recommendations"""
        recommendations = []
        
        if peak_utilization > 0.9:
            recommendations.extend([
                "Consider longer trains or coupled sets during peak hours",
                "Implement demand-based pricing to spread peak loads",
                "Plan for future fleet expansion"
            ])
        
        if average_utilization < 0.4:
            recommendations.append("Consider smaller trains or reduced frequency during off-peak hours")
        
        if peak_utilization > 1.2:
            recommendations.append("Immediate capacity expansion required")
        
        return recommendations
    
    def _analyze_fleet_requirements(self, selected_trains: Dict[str, Any], 
                                  budget_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall fleet requirements across all routes"""
        total_procurement_cost = 0
        total_operational_cost_annual = 0
        total_trains = 0
        train_types = {}
        
        for route_name, selection in selected_trains.items():
            if not selection:
                continue
                
            train = selection['selected_train']
            fleet = selection['fleet_requirements']
            operational_costs = selection['operational_costs']
            
            total_trains += fleet['total_fleet_size']
            total_procurement_cost += fleet['total_fleet_size'] * train['procurement_cost_usd']
            total_operational_cost_annual += operational_costs['total_operational_cost_annual']
            
            train_type = train['name']
            train_types[train_type] = train_types.get(train_type, 0) + fleet['total_fleet_size']
        
        budget = budget_constraints['rolling_stock_allocation']
        budget_sufficiency = budget / total_procurement_cost if total_procurement_cost > 0 else 1.0
        
        return {
            'total_fleet_size': total_trains,
            'total_procurement_cost': total_procurement_cost,
            'total_operational_cost_annual': total_operational_cost_annual,
            'budget_sufficiency': budget_sufficiency,
            'train_type_distribution': train_types,
            'recommendations': self._generate_fleet_recommendations(budget_sufficiency, train_types)
        }
    
    def _generate_fleet_recommendations(self, budget_sufficiency: float, train_types: Dict[str, int]) -> List[str]:
        """Generate fleet-wide recommendations"""
        recommendations = []
        
        if budget_sufficiency < 0.8:
            recommendations.extend([
                "Consider phased procurement of rolling stock",
                "Explore leasing options for initial operations",
                "Prioritize routes with highest demand for new trains"
            ])
        
        if len(train_types) > 3:
            recommendations.append("Consider standardizing train types to reduce maintenance complexity")
        
        if budget_sufficiency > 1.2:
            recommendations.append("Consider investing in additional trains for future expansion or improved frequency")
        
        return recommendations