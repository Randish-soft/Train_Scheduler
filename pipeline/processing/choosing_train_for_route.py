"""
Train Selection Module
======================

Selects appropriate train types for each route based on:
- Route characteristics (distance, demand, terrain)
- Economic factors (cost, efficiency, capacity)
- Infrastructure requirements
- Regional preferences and standards

Author: Miguel Ibrahim E
"""

import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class TrainType(Enum):
    """Enumeration of available train types."""
    HIGH_SPEED_ELECTRIC = "High-Speed Electric"
    EXPRESS_ELECTRIC = "Express Electric" 
    REGIONAL_ELECTRIC = "Regional Electric"
    REGIONAL_DIESEL = "Regional Diesel"
    COMMUTER_ELECTRIC = "Commuter Electric"
    LIGHT_RAIL = "Light Rail"
    MONORAIL = "Monorail"


@dataclass
class TrainSpecification:
    """Train specification data class."""
    type: TrainType
    max_speed: int  # km/h
    capacity: int  # passengers
    range: int  # km
    cost_per_unit: float  # USD
    operating_cost_per_km: float  # USD per km
    infrastructure_requirements: List[str]
    terrain_suitability: List[str]  # flat, hilly, mountainous
    electrification_required: bool
    maintenance_complexity: str  # low, medium, high


class TrainSelector:
    """Main class for selecting optimal trains for routes."""
    
    def __init__(self, route_data: Dict[str, Any]):
        self.route_data = route_data
        self.logger = logging.getLogger(__name__)
        self.train_catalog = self._initialize_train_catalog()
        
    def select(self) -> Dict[str, Any]:
        """
        Main selection method that matches trains to routes.
        
        Returns:
            Dictionary containing train selection results
        """
        self.logger.info("🚂 Selecting optimal trains for routes...")
        
        # Get route information
        demand_data = self.route_data.get('demand', {})
        demand_matrix = demand_data.get('demand_matrix', {})
        terrain_data = self.route_data.get('terrain', {})
        
        # Select trains for each route
        route_recommendations = {}
        fleet_requirements = {}
        
        for city1_name, destinations in demand_matrix.items():
            for city2_name, route_info in destinations.items():
                route_key = f"{city1_name} - {city2_name}"
                
                # Select optimal train for this route
                train_recommendation = self._select_train_for_route(
                    route_info, terrain_data
                )
                
                route_recommendations[route_key] = train_recommendation
                
                # Update fleet requirements
                train_type = train_recommendation['train_type']
                if train_type not in fleet_requirements:
                    fleet_requirements[train_type] = 0
                fleet_requirements[train_type] += train_recommendation['units_needed']
        
        # Calculate total fleet costs and requirements
        fleet_summary = self._calculate_fleet_summary(fleet_requirements)
        
        # Generate train deployment strategy
        deployment_strategy = self._generate_deployment_strategy(route_recommendations)
        
        results = {
            'route_recommendations': route_recommendations,
            'fleet_requirements': fleet_requirements,
            'fleet_summary': fleet_summary,
            'deployment_strategy': deployment_strategy,
            'total_routes': len(route_recommendations)
        }
        
        self.logger.info("✅ Train selection completed")
        return results
    
    def _initialize_train_catalog(self) -> Dict[TrainType, TrainSpecification]:
        """Initialize catalog of available train types and specifications."""
        return {
            TrainType.HIGH_SPEED_ELECTRIC: TrainSpecification(
                type=TrainType.HIGH_SPEED_ELECTRIC,
                max_speed=320,
                capacity=400,
                range=1000,
                cost_per_unit=80000000,  # $80M
                operating_cost_per_km=2.50,
                infrastructure_requirements=['electrification', 'dedicated_tracks', 'advanced_signaling'],
                terrain_suitability=['flat', 'hilly'],
                electrification_required=True,
                maintenance_complexity='high'
            ),
            TrainType.EXPRESS_ELECTRIC: TrainSpecification(
                type=TrainType.EXPRESS_ELECTRIC,
                max_speed=200,
                capacity=300,
                range=800,
                cost_per_unit=45000000,  # $45M
                operating_cost_per_km=1.80,
                infrastructure_requirements=['electrification', 'standard_tracks', 'modern_signaling'],
                terrain_suitability=['flat', 'hilly', 'mountainous'],
                electrification_required=True,
                maintenance_complexity='medium'
            ),
            TrainType.REGIONAL_ELECTRIC: TrainSpecification(
                type=TrainType.REGIONAL_ELECTRIC,
                max_speed=160,
                capacity=200,
                range=500,
                cost_per_unit=25000000,  # $25M
                operating_cost_per_km=1.20,
                infrastructure_requirements=['electrification', 'standard_tracks'],
                terrain_suitability=['flat', 'hilly', 'mountainous'],
                electrification_required=True,
                maintenance_complexity='medium'
            ),
            TrainType.REGIONAL_DIESEL: TrainSpecification(
                type=TrainType.REGIONAL_DIESEL,
                max_speed=140,
                capacity=180,
                range=600,
                cost_per_unit=15000000,  # $15M
                operating_cost_per_km=1.50,
                infrastructure_requirements=['standard_tracks', 'basic_signaling'],
                terrain_suitability=['flat', 'hilly', 'mountainous'],
                electrification_required=False,
                maintenance_complexity='low'
            ),
            TrainType.COMMUTER_ELECTRIC: TrainSpecification(
                type=TrainType.COMMUTER_ELECTRIC,
                max_speed=120,
                capacity=150,
                range=200,
                cost_per_unit=12000000,  # $12M
                operating_cost_per_km=0.90,
                infrastructure_requirements=['electrification', 'standard_tracks'],
                terrain_suitability=['flat', 'hilly'],
                electrification_required=True,
                maintenance_complexity='low'
            ),
            TrainType.LIGHT_RAIL: TrainSpecification(
                type=TrainType.LIGHT_RAIL,
                max_speed=80,
                capacity=100,
                range=100,
                cost_per_unit=8000000,  # $8M
                operating_cost_per_km=0.60,
                infrastructure_requirements=['electrification', 'light_rail_tracks'],
                terrain_suitability=['flat'],
                electrification_required=True,
                maintenance_complexity='low'
            ),
            TrainType.MONORAIL: TrainSpecification(
                type=TrainType.MONORAIL,
                max_speed=100,
                capacity=120,
                range=150,
                cost_per_unit=20000000,  # $20M
                operating_cost_per_km=1.00,
                infrastructure_requirements=['monorail_guideway', 'electrification'],
                terrain_suitability=['flat', 'hilly'],
                electrification_required=True,
                maintenance_complexity='medium'
            )
        }
    
    def _select_train_for_route(self, route_info: Dict[str, Any], terrain_data: Dict[str, Any]) -> Dict[str, Any]:
        """Select the optimal train type for a specific route."""
        distance = route_info.get('distance_km', 0)
        demand_score = route_info.get('demand_score', 0)
        daily_passengers = route_info.get('daily_passengers', 0)
        population_served = route_info.get('population_served', 0)
        
        # Get terrain difficulty
        terrain_difficulty = terrain_data.get('terrain_difficulty', 'Medium')
        terrain_type = self._map_terrain_difficulty(terrain_difficulty)
        
        # Score each train type for this route
        train_scores = {}
        
        for train_type, spec in self.train_catalog.items():
            score = self._calculate_train_score(
                spec, distance, demand_score, daily_passengers, 
                population_served, terrain_type
            )
            train_scores[train_type] = score
        
        # Select the best scoring train
        best_train_type = max(train_scores, key=train_scores.get)
        best_spec = self.train_catalog[best_train_type]
        
        # Calculate operational parameters
        operational_params = self._calculate_operational_parameters(
            best_spec, distance, daily_passengers
        )
        
        return {
            'train_type': best_train_type.value,
            'specification': {
                'max_speed': best_spec.max_speed,
                'capacity': best_spec.capacity,
                'cost_per_unit': best_spec.cost_per_unit,
                'operating_cost_per_km': best_spec.operating_cost_per_km,
                'electrification_required': best_spec.electrification_required,
                'maintenance_complexity': best_spec.maintenance_complexity
            },
            'operational_parameters': operational_params,
            'suitability_score': train_scores[best_train_type],
            'alternative_options': self._get_alternative_options(train_scores, best_train_type)
        }
    
    def _map_terrain_difficulty(self, difficulty: str) -> str:
        """Map terrain difficulty to terrain type."""
        if difficulty.lower() == 'high':
            return 'mountainous'
        elif difficulty.lower() == 'medium':
            return 'hilly'
        else:
            return 'flat'
    
    def _calculate_train_score(self, spec: TrainSpecification, distance: float, 
                              demand_score: float, daily_passengers: int,
                              population_served: int, terrain_type: str) -> float:
        """Calculate suitability score for a train type on a specific route."""
        score = 0.0
        
        # Distance suitability (30% weight)
        distance_score = self._calculate_distance_score(spec, distance)
        score += distance_score * 0.3
        
        # Capacity suitability (25% weight)
        capacity_score = self._calculate_capacity_score(spec, daily_passengers)
        score += capacity_score * 0.25
        
        # Economic efficiency (20% weight)
        economic_score = self._calculate_economic_score(spec, distance, daily_passengers)
        score += economic_score * 0.2
        
        # Terrain suitability (15% weight)
        terrain_score = 1.0 if terrain_type in spec.terrain_suitability else 0.3
        score += terrain_score * 0.15
        
        # Demand level suitability (10% weight)
        demand_level_score = min(demand_score * 10, 1.0)  # Normalize demand score
        score += demand_level_score * 0.1
        
        return score
    
    def _calculate_distance_score(self, spec: TrainSpecification, distance: float) -> float:
        """Calculate how well a train type suits the route distance."""
        if distance <= 50:
            # Short routes
            if spec.type in [TrainType.COMMUTER_ELECTRIC, TrainType.LIGHT_RAIL]:
                return 1.0
            elif spec.type in [TrainType.REGIONAL_ELECTRIC, TrainType.REGIONAL_DIESEL]:
                return 0.8
            else:
                return 0.4
        elif distance <= 150:
            # Medium routes
            if spec.type in [TrainType.REGIONAL_ELECTRIC, TrainType.REGIONAL_DIESEL]:
                return 1.0
            elif spec.type == TrainType.EXPRESS_ELECTRIC:
                return 0.9
            elif spec.type == TrainType.COMMUTER_ELECTRIC:
                return 0.6
            else:
                return 0.5
        elif distance <= 400:
            # Long routes
            if spec.type == TrainType.EXPRESS_ELECTRIC:
                return 1.0
            elif spec.type == TrainType.HIGH_SPEED_ELECTRIC:
                return 0.9
            elif spec.type in [TrainType.REGIONAL_ELECTRIC, TrainType.REGIONAL_DIESEL]:
                return 0.7
            else:
                return 0.3
        else:
            # Very long routes
            if spec.type == TrainType.HIGH_SPEED_ELECTRIC:
                return 1.0
            elif spec.type == TrainType.EXPRESS_ELECTRIC:
                return 0.8
            else:
                return 0.4
    
    def _calculate_capacity_score(self, spec: TrainSpecification, daily_passengers: int) -> float:
        """Calculate how well train capacity matches demand."""
        if daily_passengers == 0:
            return 0.5  # Neutral score for unknown demand
        
        # Calculate trips needed per day (assuming 80% capacity utilization)
        trips_needed = daily_passengers / (spec.capacity * 0.8)
        
        # Optimal range: 2-8 trips per day
        if 2 <= trips_needed <= 8:
            return 1.0
        elif 1 <= trips_needed < 2:
            return 0.8  # Slight overcapacity
        elif 8 < trips_needed <= 12:
            return 0.7  # Slight undercapacity
        elif trips_needed < 1:
            return 0.6  # Significant overcapacity
        else:
            return 0.3  # Significant undercapacity
    
    def _calculate_economic_score(self, spec: TrainSpecification, distance: float, 
                                 daily_passengers: int) -> float:
        """Calculate economic efficiency score."""
        if daily_passengers == 0 or distance == 0:
            return 0.5
        
        # Calculate annual operating cost
        annual_km = distance * 2 * 365  # Return trips daily
        annual_operating_cost = annual_km * spec.operating_cost_per_km
        
        # Calculate revenue (rough estimate)
        annual_passengers = daily_passengers * 365
        avg_ticket_price = 5 + (distance * 0.08)  # Base + distance component
        annual_revenue = annual_passengers * avg_ticket_price
        
        # Calculate profit margin
        if annual_revenue > 0:
            profit_margin = (annual_revenue - annual_operating_cost) / annual_revenue
            
            # Score based on profit margin
            if profit_margin > 0.3:
                return 1.0
            elif profit_margin > 0.1:
                return 0.8
            elif profit_margin > 0:
                return 0.6
            else:
                return 0.2
        else:
            return 0.1
    
    def _calculate_operational_parameters(self, spec: TrainSpecification, 
                                        distance: float, daily_passengers: int) -> Dict[str, Any]:
        """Calculate operational parameters for the selected train."""
        # Calculate number of trains needed
        if daily_passengers > 0:
            trips_per_day = max(1, daily_passengers // (spec.capacity * 0.8))
            units_needed = max(1, int(trips_per_day / 4))  # Assuming 4 trips per train per day
        else:
            trips_per_day = 2  # Minimum service
            units_needed = 1
        
        # Calculate travel time
        # Assuming average speed is 70% of max speed for regional routes
        avg_speed = spec.max_speed * 0.7
        travel_time_hours = distance / avg_speed
        
        # Calculate frequency
        if trips_per_day >= 8:
            frequency = "Every 2-3 hours"
        elif trips_per_day >= 4:
            frequency = "Every 4-6 hours"
        else:
            frequency = "2-3 times daily"
        
        return {
            'units_needed': units_needed,
            'trips_per_day': trips_per_day,
            'travel_time_hours': round(travel_time_hours, 1),
            'frequency': frequency,
            'estimated_travel_time': f"{int(travel_time_hours)}h {int((travel_time_hours % 1) * 60)}m",
            'daily_operating_cost': distance * 2 * spec.operating_cost_per_km * trips_per_day,
            'total_unit_cost': units_needed * spec.cost_per_unit
        }
    
    def _get_alternative_options(self, train_scores: Dict[TrainType, float], 
                               selected_type: TrainType) -> List[Dict[str, Any]]:
        """Get alternative train options for the route."""
        # Sort by score, excluding the selected type
        alternatives = [(train_type, score) for train_type, score in train_scores.items() 
                       if train_type != selected_type]
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 2 alternatives
        result = []
        for train_type, score in alternatives[:2]:
            spec = self.train_catalog[train_type]
            result.append({
                'train_type': train_type.value,
                'suitability_score': score,
                'max_speed': spec.max_speed,
                'capacity': spec.capacity,
                'cost_per_unit': spec.cost_per_unit
            })
        
        return result
    
    def _calculate_fleet_summary(self, fleet_requirements: Dict[str, int]) -> Dict[str, Any]:
        """Calculate summary of fleet requirements and costs."""
        total_units = sum(fleet_requirements.values())
        total_cost = 0
        
        fleet_composition = {}
        
        for train_type_name, units in fleet_requirements.items():
            # Find the train type enum
            train_type = None
            for tt in TrainType:
                if tt.value == train_type_name:
                    train_type = tt
                    break
            
            if train_type and train_type in self.train_catalog:
                spec = self.train_catalog[train_type]
                unit_cost = spec.cost_per_unit
                type_total_cost = units * unit_cost
                total_cost += type_total_cost
                
                fleet_composition[train_type_name] = {
                    'units': units,
                    'unit_cost': unit_cost,
                    'total_cost': type_total_cost,
                    'percentage_of_fleet': (units / total_units) * 100 if total_units > 0 else 0
                }
        
        return {
            'total_units': total_units,
            'total_fleet_cost': total_cost,
            'fleet_composition': fleet_composition,
            'average_cost_per_unit': total_cost / total_units if total_units > 0 else 0
        }
    
    def _generate_deployment_strategy(self, route_recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy for train deployment across the network."""
        # Group routes by train type
        train_type_routes = {}
        
        for route, recommendation in route_recommendations.items():
            train_type = recommendation['train_type']
            if train_type not in train_type_routes:
                train_type_routes[train_type] = []
            train_type_routes[train_type].append(route)
        
        # Create deployment phases
        phases = []
        
        # Phase 1: High-capacity, high-demand routes
        phase1_types = ['High-Speed Electric', 'Express Electric']
        phase1_routes = []
        for train_type in phase1_types:
            if train_type in train_type_routes:
                phase1_routes.extend(train_type_routes[train_type])
        
        if phase1_routes:
            phases.append({
                'phase': 1,
                'description': 'High-capacity intercity services',
                'routes': phase1_routes,
                'timeline': 'Years 1-3',
                'priority': 'High'
            })
        
        # Phase 2: Regional connections
        phase2_types = ['Regional Electric', 'Regional Diesel']
        phase2_routes = []
        for train_type in phase2_types:
            if train_type in train_type_routes:
                phase2_routes.extend(train_type_routes[train_type])
        
        if phase2_routes:
            phases.append({
                'phase': 2,
                'description': 'Regional connectivity',
                'routes': phase2_routes,
                'timeline': 'Years 2-5',
                'priority': 'Medium'
            })
        
        # Phase 3: Local and urban services
        phase3_types = ['Commuter Electric', 'Light Rail', 'Monorail']
        phase3_routes = []
        for train_type in phase3_types:
            if train_type in train_type_routes:
                phase3_routes.extend(train_type_routes[train_type])
        
        if phase3_routes:
            phases.append({
                'phase': 3,
                'description': 'Urban and commuter services',
                'routes': phase3_routes,
                'timeline': 'Years 4-7',
                'priority': 'Medium'
            })
        
        return {
            'deployment_phases': phases,
            'total_phases': len(phases),
            'recommended_approach': 'Phased deployment starting with high-demand intercity routes'
        }