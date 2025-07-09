"""
BCPC Pipeline: Route Optimization Module

This module optimizes rail routes using mixed-integer programming and multi-criteria
decision analysis. It considers cost, time, capacity, environmental impact,
and other factors to select optimal routes and configurations.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
from itertools import combinations

logger = logging.getLogger(__name__)

@dataclass
class OptimizationCriteria:
    """Criteria for route optimization."""
    cost_weight: float = 0.35
    time_weight: float = 0.25
    capacity_weight: float = 0.15
    environmental_weight: float = 0.15
    feasibility_weight: float = 0.10
    
    def __post_init__(self):
        """Normalize weights to sum to 1.0."""
        total = (self.cost_weight + self.time_weight + self.capacity_weight + 
                self.environmental_weight + self.feasibility_weight)
        if total != 1.0:
            self.cost_weight /= total
            self.time_weight /= total
            self.capacity_weight /= total
            self.environmental_weight /= total
            self.feasibility_weight /= total

@dataclass
class OptimizedRoute:
    """Optimized route with calculated scores."""
    route_option: Any  # RouteOption from route_mapper
    demand_match_score: float
    cost_efficiency_score: float
    time_efficiency_score: float
    environmental_score: float
    feasibility_score: float
    capacity_utilization_score: float
    overall_score: float
    rank: int = 0
    selected_train_type: Optional[str] = None
    annual_operating_cost: float = 0.0
    annual_revenue_potential: float = 0.0
    payback_period_years: float = 0.0
    risk_assessment: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NetworkOptimization:
    """Network-level optimization results."""
    optimized_routes: List[OptimizedRoute]
    network_cost: float
    network_capacity: int
    network_coverage_score: float
    total_annual_demand: int
    construction_phases: List[Dict[str, Any]]
    investment_schedule: Dict[int, float]  # year -> investment amount
    roi_projection: Dict[str, float]

class RouteOptimizer:
    """Optimizes rail routes using multi-criteria decision analysis."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize the route optimizer."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Default optimization criteria
        self.default_criteria = OptimizationCriteria()
        
        # Scoring parameters
        self.cost_thresholds = {
            'excellent': 30_000_000,    # $30M per km
            'good': 50_000_000,         # $50M per km
            'acceptable': 80_000_000,   # $80M per km
            'poor': 150_000_000         # $150M per km
        }
        
        self.time_thresholds = {
            'excellent': 2.0,    # 2 hours or less
            'good': 4.0,         # 4 hours or less
            'acceptable': 6.0,   # 6 hours or less
            'poor': 10.0         # 10 hours or less
        }
        
        # Risk factors
        self.risk_factors = {
            'geological': 0.15,
            'environmental': 0.20,
            'political': 0.10,
            'financial': 0.25,
            'technical': 0.20,
            'schedule': 0.10
        }
    
    def optimize_routes(self, routes: Dict[str, List], demand_data: Dict[str, Any], 
                       train_data: List) -> Dict[str, List[OptimizedRoute]]:
        """
        Optimize all route options using multi-criteria analysis.
        
        Args:
            routes: Dictionary of route options from RouteMapper
            demand_data: Demand analysis results
            train_data: Available train specifications
            
        Returns:
            Dictionary of optimized routes for each city pair
        """
        try:
            logger.info(f"Starting route optimization for {len(routes)} route sets")
            
            optimized_routes = {}
            
            for route_key, route_options in routes.items():
                if not route_options:
                    continue
                
                try:
                    logger.info(f"Optimizing routes for: {route_key}")
                    
                    # Get demand data for this route
                    route_demand = self._get_route_demand_data(route_key, demand_data)
                    
                    # Optimize each route option
                    optimized_options = []
                    for route_option in route_options:
                        optimized_route = self._optimize_single_route(
                            route_option, route_demand, train_data
                        )
                        optimized_options.append(optimized_route)
                    
                    # Rank the optimized routes
                    ranked_routes = self._rank_routes(optimized_options)
                    optimized_routes[route_key] = ranked_routes
                    
                except Exception as e:
                    logger.warning(f"Error optimizing routes for {route_key}: {e}")
                    continue
            
            logger.info(f"Route optimization complete for {len(optimized_routes)} route sets")
            return optimized_routes
            
        except Exception as e:
            logger.error(f"Route optimization failed: {e}")
            raise
    
    def _optimize_single_route(self, route_option, demand_data: Dict, train_data: List) -> OptimizedRoute:
        """Optimize a single route option."""
        try:
            # Calculate individual scoring components
            demand_score = self._calculate_demand_match_score(route_option, demand_data)
            cost_score = self._calculate_cost_efficiency_score(route_option)
            time_score = self._calculate_time_efficiency_score(route_option)
            env_score = self._calculate_environmental_score(route_option)
            feasibility_score = self._calculate_feasibility_score(route_option)
            capacity_score = self._calculate_capacity_utilization_score(route_option, demand_data)
            
            # Calculate overall weighted score
            overall_score = self._calculate_overall_score(
                demand_score, cost_score, time_score, env_score, feasibility_score, capacity_score
            )
            
            # Select optimal train type for this route
            selected_train = self._select_optimal_train_for_route(route_option, demand_data, train_data)
            
            # Calculate financial metrics
            annual_operating_cost = self._calculate_annual_operating_cost(route_option, selected_train)
            annual_revenue = self._calculate_annual_revenue_potential(route_option, demand_data)
            payback_period = self._calculate_payback_period(route_option, annual_operating_cost, annual_revenue)
            
            # Perform risk assessment
            risk_assessment = self._perform_risk_assessment(route_option)
            
            return OptimizedRoute(
                route_option=route_option,
                demand_match_score=demand_score,
                cost_efficiency_score=cost_score,
                time_efficiency_score=time_score,
                environmental_score=env_score,
                feasibility_score=feasibility_score,
                capacity_utilization_score=capacity_score,
                overall_score=overall_score,
                selected_train_type=selected_train.get('name') if selected_train else None,
                annual_operating_cost=annual_operating_cost,
                annual_revenue_potential=annual_revenue,
                payback_period_years=payback_period,
                risk_assessment=risk_assessment
            )
            
        except Exception as e:
            logger.error(f"Error optimizing single route: {e}")
            raise
    
    def _calculate_demand_match_score(self, route_option, demand_data: Dict) -> float:
        """Calculate how well the route matches demand patterns."""
        try:
            if not demand_data:
                return 0.5
            
            annual_demand = demand_data.get('annual_demand', 0)
            peak_demand = demand_data.get('peak_hourly_demand', 0)
            
            # Score based on demand levels
            if annual_demand < 50_000:
                demand_level_score = 0.3
            elif annual_demand < 200_000:
                demand_level_score = 0.6
            elif annual_demand < 500_000:
                demand_level_score = 0.8
            elif annual_demand < 1_000_000:
                demand_level_score = 0.9
            else:
                demand_level_score = 1.0
            
            # Consider route distance vs demand density
            distance = route_option.total_distance_km
            demand_density = annual_demand / distance if distance > 0 else 0
            
            if demand_density > 5000:  # > 5000 passengers per km per year
                density_score = 1.0
            elif demand_density > 2000:
                density_score = 0.8
            elif demand_density > 1000:
                density_score = 0.6
            elif demand_density > 500:
                density_score = 0.4
            else:
                density_score = 0.2
            
            # Combine scores
            return (demand_level_score * 0.6 + density_score * 0.4)
            
        except Exception as e:
            logger.warning(f"Error calculating demand match score: {e}")
            return 0.5
    
    def _calculate_cost_efficiency_score(self, route_option) -> float:
        """Calculate cost efficiency score (lower cost = higher score)."""
        try:
            total_cost = route_option.total_construction_cost
            distance = route_option.total_distance_km
            cost_per_km = total_cost / distance if distance > 0 else float('inf')
            
            # Score based on cost thresholds
            if cost_per_km <= self.cost_thresholds['excellent']:
                return 1.0
            elif cost_per_km <= self.cost_thresholds['good']:
                return 0.8
            elif cost_per_km <= self.cost_thresholds['acceptable']:
                return 0.6
            elif cost_per_km <= self.cost_thresholds['poor']:
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            logger.warning(f"Error calculating cost efficiency score: {e}")
            return 0.5
    
    def _calculate_time_efficiency_score(self, route_option) -> float:
        """Calculate time efficiency score (shorter travel time = higher score)."""
        try:
            travel_time = route_option.travel_time_hours
            
            # Score based on time thresholds
            if travel_time <= self.time_thresholds['excellent']:
                return 1.0
            elif travel_time <= self.time_thresholds['good']:
                return 0.8
            elif travel_time <= self.time_thresholds['acceptable']:
                return 0.6
            elif travel_time <= self.time_thresholds['poor']:
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            logger.warning(f"Error calculating time efficiency score: {e}")
            return 0.5
    
    def _calculate_environmental_score(self, route_option) -> float:
        """Calculate environmental score (lower impact = higher score)."""
        try:
            env_impact = route_option.environmental_impact_score
            
            # Invert the score (lower impact should give higher score)
            if env_impact <= 2.0:
                return 1.0
            elif env_impact <= 4.0:
                return 0.8
            elif env_impact <= 6.0:
                return 0.6
            elif env_impact <= 8.0:
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            logger.warning(f"Error calculating environmental score: {e}")
            return 0.5
    
    def _calculate_feasibility_score(self, route_option) -> float:
        """Calculate feasibility score."""
        try:
            # Use the feasibility score from the route option
            feasibility = route_option.feasibility_score
            return min(1.0, max(0.0, feasibility / 10.0))  # Normalize to 0-1
            
        except Exception as e:
            logger.warning(f"Error calculating feasibility score: {e}")
            return 0.5
    
    def _calculate_capacity_utilization_score(self, route_option, demand_data: Dict) -> float:
        """Calculate capacity utilization score."""
        try:
            if not demand_data:
                return 0.5
            
            annual_demand = demand_data.get('annual_demand', 0)
            peak_demand = demand_data.get('peak_hourly_demand', 0)
            
            # Estimate capacity based on route characteristics
            # This is simplified - in reality would depend on selected train type
            if route_option.max_speed_kmh > 250:  # High-speed rail
                estimated_hourly_capacity = 2000
            elif route_option.max_speed_kmh > 160:  # Fast conventional rail
                estimated_hourly_capacity = 1500
            else:  # Regional rail
                estimated_hourly_capacity = 800
            
            # Calculate utilization
            if estimated_hourly_capacity > 0:
                utilization = peak_demand / estimated_hourly_capacity
                
                # Optimal utilization is around 70-80%
                if 0.6 <= utilization <= 0.8:
                    return 1.0
                elif 0.4 <= utilization <= 0.9:
                    return 0.8
                elif 0.2 <= utilization <= 0.95:
                    return 0.6
                else:
                    return 0.3
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating capacity utilization score: {e}")
            return 0.5
    
    def _calculate_overall_score(self, demand_score: float, cost_score: float, time_score: float,
                               env_score: float, feasibility_score: float, capacity_score: float) -> float:
        """Calculate weighted overall score."""
        try:
            criteria = self.default_criteria
            
            overall_score = (
                demand_score * criteria.cost_weight +  # Demand affects cost considerations
                cost_score * criteria.cost_weight +
                time_score * criteria.time_weight +
                env_score * criteria.environmental_weight +
                feasibility_score * criteria.feasibility_weight +
                capacity_score * criteria.capacity_weight
            )
            
            return min(1.0, max(0.0, overall_score))
            
        except Exception as e:
            logger.warning(f"Error calculating overall score: {e}")
            return 0.5
    
    def _select_optimal_train_for_route(self, route_option, demand_data: Dict, train_data: List) -> Optional[Dict]:
        """Select the optimal train type for this specific route."""
        try:
            if not train_data:
                return None
            
            best_train = None
            best_score = 0.0
            
            for train in train_data:
                # Calculate train suitability score for this route
                score = self._calculate_train_route_suitability(train, route_option, demand_data)
                
                if score > best_score:
                    best_score = score
                    best_train = train
            
            return best_train.__dict__ if hasattr(best_train, '__dict__') else best_train
            
        except Exception as e:
            logger.warning(f"Error selecting optimal train: {e}")
            return None
    
    def _calculate_train_route_suitability(self, train, route_option, demand_data: Dict) -> float:
        """Calculate how suitable a train is for a specific route."""
        try:
            score = 0.0
            
            # Speed suitability
            if hasattr(train, 'operating_speed_kmh'):
                train_speed = train.operating_speed_kmh
                route_max_speed = route_option.max_speed_kmh
                
                if train_speed >= route_max_speed:
                    speed_score = 1.0
                else:
                    speed_score = train_speed / route_max_speed
                
                score += speed_score * 0.3
            
            # Capacity suitability
            if hasattr(train, 'capacity_total') and demand_data:
                train_capacity = train.capacity_total
                peak_demand = demand_data.get('peak_hourly_demand', 0)
                
                if peak_demand > 0:
                    capacity_ratio = train_capacity / peak_demand
                    if 1.0 <= capacity_ratio <= 2.0:  # 1x to 2x capacity is optimal
                        capacity_score = 1.0
                    elif capacity_ratio < 1.0:
                        capacity_score = capacity_ratio
                    else:
                        capacity_score = 2.0 / capacity_ratio
                    
                    score += capacity_score * 0.3
            
            # Cost efficiency
            if hasattr(train, 'operating_cost_per_km'):
                # Lower operating cost is better
                cost_score = max(0.0, 1.0 - (train.operating_cost_per_km / 50.0))  # Normalize
                score += cost_score * 0.2
            
            # Terrain suitability
            if hasattr(train, 'maximum_grade_percent'):
                route_max_grade = max(abs(seg.grade_percent) for seg in route_option.route_segments) if route_option.route_segments else 0
                if train.maximum_grade_percent >= route_max_grade:
                    terrain_score = 1.0
                else:
                    terrain_score = train.maximum_grade_percent / route_max_grade
                
                score += terrain_score * 0.2
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.warning(f"Error calculating train route suitability: {e}")
            return 0.5
    
    def _calculate_annual_operating_cost(self, route_option, selected_train: Optional[Dict]) -> float:
        """Calculate annual operating cost for the route."""
        try:
            if not selected_train:
                return 0.0
            
            distance = route_option.total_distance_km
            
            # Estimate annual train-km based on service frequency
            # Assume 20 round trips per day average
            daily_train_km = distance * 2 * 20  # Round trips
            annual_train_km = daily_train_km * 365
            
            # Operating cost per km
            cost_per_km = selected_train.get('operating_cost_per_km', 10.0)
            
            # Annual maintenance cost
            maintenance_cost = selected_train.get('maintenance_cost_per_year', 1_000_000)
            
            # Total annual operating cost
            total_cost = annual_train_km * cost_per_km + maintenance_cost
            
            return total_cost
            
        except Exception as e:
            logger.warning(f"Error calculating annual operating cost: {e}")
            return 0.0
    
    def _calculate_annual_revenue_potential(self, route_option, demand_data: Dict) -> float:
        """Calculate annual revenue potential for the route."""
        try:
            if not demand_data:
                return 0.0
            
            annual_demand = demand_data.get('annual_demand', 0)
            distance = route_option.total_distance_km
            
            # Estimate fare based on distance and route type
            if route_option.max_speed_kmh > 250:  # High-speed rail
                fare_per_km = 0.25  # $0.25 per km
            elif route_option.max_speed_kmh > 160:  # Fast rail
                fare_per_km = 0.15  # $0.15 per km
            else:  # Regional rail
                fare_per_km = 0.10  # $0.10 per km
            
            average_fare = distance * fare_per_km
            annual_revenue = annual_demand * average_fare
            
            return annual_revenue
            
        except Exception as e:
            logger.warning(f"Error calculating annual revenue potential: {e}")
            return 0.0
    
    def _calculate_payback_period(self, route_option, annual_operating_cost: float, 
                                annual_revenue: float) -> float:
        """Calculate payback period in years."""
        try:
            construction_cost = route_option.total_construction_cost
            annual_profit = annual_revenue - annual_operating_cost
            
            if annual_profit <= 0:
                return 999.0  # Never pays back
            
            payback_years = construction_cost / annual_profit
            return min(payback_years, 100.0)  # Cap at 100 years
            
        except Exception as e:
            logger.warning(f"Error calculating payback period: {e}")
            return 999.0
    
    def _perform_risk_assessment(self, route_option) -> Dict[str, Any]:
        """Perform comprehensive risk assessment for the route."""
        try:
            risks = {}
            
            # Geological risk
            geological_risk = self._assess_geological_risk(route_option)
            risks['geological'] = geological_risk
            
            # Environmental risk
            env_risk = min(10.0, route_option.environmental_impact_score)
            risks['environmental'] = env_risk
            
            # Technical risk
            technical_risk = self._assess_technical_risk(route_option)
            risks['technical'] = technical_risk
            
            # Financial risk
            financial_risk = self._assess_financial_risk(route_option)
            risks['financial'] = financial_risk
            
            # Political/regulatory risk
            political_risk = 5.0  # Default medium risk
            risks['political'] = political_risk
            
            # Schedule risk
            schedule_risk = self._assess_schedule_risk(route_option)
            risks['schedule'] = schedule_risk
            
            # Calculate overall risk score
            overall_risk = sum(
                risks[factor] * weight
                for factor, weight in self.risk_factors.items()
                if factor in risks
            )
            
            risks['overall'] = overall_risk
            risks['risk_level'] = self._categorize_risk_level(overall_risk)
            
            return risks
            
        except Exception as e:
            logger.warning(f"Error performing risk assessment: {e}")
            return {'overall': 5.0, 'risk_level': 'medium'}
    
    def _assess_geological_risk(self, route_option) -> float:
        """Assess geological risk based on route characteristics."""
        try:
            risk_score = 1.0  # Base low risk
            
            # Check terrain difficulty
            terrain_difficulty = route_option.terrain_difficulty_score
            risk_score += terrain_difficulty * 0.5
            
            # Check for tunnels (higher geological risk)
            tunnel_length = route_option.infrastructure_summary.get('tunnel_km', 0)
            total_length = route_option.total_distance_km
            tunnel_ratio = tunnel_length / total_length if total_length > 0 else 0
            
            risk_score += tunnel_ratio * 5.0
            
            return min(10.0, risk_score)
            
        except Exception as e:
            logger.warning(f"Error assessing geological risk: {e}")
            return 5.0
    
    def _assess_technical_risk(self, route_option) -> float:
        """Assess technical risk based on route complexity."""
        try:
            risk_score = 1.0
            
            # Complex infrastructure increases risk
            infrastructure = route_option.infrastructure_summary
            special_infrastructure = (
                infrastructure.get('tunnel_km', 0) +
                infrastructure.get('bridge_km', 0) +
                infrastructure.get('viaduct_km', 0)
            )
            
            total_length = route_option.total_distance_km
            complexity_ratio = special_infrastructure / total_length if total_length > 0 else 0
            
            risk_score += complexity_ratio * 6.0
            
            # High-speed rail has higher technical risk
            if route_option.max_speed_kmh > 250:
                risk_score += 2.0
            elif route_option.max_speed_kmh > 200:
                risk_score += 1.0
            
            return min(10.0, risk_score)
            
        except Exception as e:
            logger.warning(f"Error assessing technical risk: {e}")
            return 5.0
    
    def _assess_financial_risk(self, route_option) -> float:
        """Assess financial risk based on cost and revenue projections."""
        try:
            risk_score = 1.0
            
            # High construction cost increases financial risk
            distance = route_option.total_distance_km
            cost_per_km = route_option.total_construction_cost / distance if distance > 0 else 0
            
            if cost_per_km > 100_000_000:  # > $100M per km
                risk_score += 4.0
            elif cost_per_km > 60_000_000:  # > $60M per km
                risk_score += 2.0
            elif cost_per_km > 30_000_000:  # > $30M per km
                risk_score += 1.0
            
            # Long payback period increases risk
            # Note: This would require payback period calculation
            # For now, use cost as proxy
            
            return min(10.0, risk_score)
            
        except Exception as e:
            logger.warning(f"Error assessing financial risk: {e}")
            return 5.0
    
    def _assess_schedule_risk(self, route_option) -> float:
        """Assess schedule risk based on route complexity."""
        try:
            risk_score = 2.0  # Base schedule risk
            
            # Complex infrastructure increases schedule risk
            infrastructure = route_option.infrastructure_summary
            
            # Tunnels have high schedule risk
            tunnel_km = infrastructure.get('tunnel_km', 0)
            risk_score += tunnel_km * 0.5
            
            # Bridges and viaducts have moderate schedule risk
            bridge_viaduct_km = infrastructure.get('bridge_km', 0) + infrastructure.get('viaduct_km', 0)
            risk_score += bridge_viaduct_km * 0.3
            
            # Very long routes have higher schedule risk
            if route_option.total_distance_km > 500:
                risk_score += 2.0
            elif route_option.total_distance_km > 200:
                risk_score += 1.0
            
            return min(10.0, risk_score)
            
        except Exception as e:
            logger.warning(f"Error assessing schedule risk: {e}")
            return 5.0
    
    def _categorize_risk_level(self, overall_risk: float) -> str:
        """Categorize overall risk level."""
        if overall_risk <= 3.0:
            return 'low'
        elif overall_risk <= 5.0:
            return 'medium'
        elif overall_risk <= 7.0:
            return 'high'
        else:
            return 'extreme'
    
    def _rank_routes(self, optimized_routes: List[OptimizedRoute]) -> List[OptimizedRoute]:
        """Rank optimized routes by overall score."""
        try:
            # Sort by overall score (descending)
            ranked_routes = sorted(optimized_routes, key=lambda r: r.overall_score, reverse=True)
            
            # Assign ranks
            for i, route in enumerate(ranked_routes):
                route.rank = i + 1
            
            return ranked_routes
            
        except Exception as e:
            logger.error(f"Error ranking routes: {e}")
            return optimized_routes
    
    def select_best_route(self, optimized_routes: Dict[str, List[OptimizedRoute]]) -> Dict[str, OptimizedRoute]:
        """Select the best route for each city pair."""
        try:
            best_routes = {}
            
            for route_key, routes in optimized_routes.items():
                if routes:
                    # Select the top-ranked route
                    best_route = routes[0]  # Already sorted by rank
                    best_routes[route_key] = best_route
                    
                    logger.info(f"Best route for {route_key}: {best_route.route_option.route_name} "
                              f"(score: {best_route.overall_score:.3f})")
            
            return best_routes
            
        except Exception as e:
            logger.error(f"Error selecting best routes: {e}")
            return {}
    
    def optimize_network(self, best_routes: Dict[str, OptimizedRoute], 
                        total_budget: Optional[float] = None) -> NetworkOptimization:
        """Optimize the entire network considering budget constraints and priorities."""
        try:
            logger.info("Optimizing network-level configuration")
            
            if not best_routes:
                return NetworkOptimization(
                    optimized_routes=[], network_cost=0.0, network_capacity=0,
                    network_coverage_score=0.0, total_annual_demand=0,
                    construction_phases=[], investment_schedule={}, roi_projection={}
                )
            
            # Calculate network metrics
            network_cost = sum(route.route_option.total_construction_cost for route in best_routes.values())
            network_capacity = sum(route.annual_revenue_potential for route in best_routes.values())
            total_demand = sum(self._get_route_annual_demand(route) for route in best_routes.values())
            
            # Calculate network coverage score
            coverage_score = self._calculate_network_coverage_score(best_routes)
            
            # Create construction phases
            construction_phases = self._create_construction_phases(best_routes, total_budget)
            
            # Create investment schedule
            investment_schedule = self._create_investment_schedule(construction_phases)
            
            # Calculate ROI projection
            roi_projection = self._calculate_roi_projection(best_routes, investment_schedule)
            
            network_optimization = NetworkOptimization(
                optimized_routes=list(best_routes.values()),
                network_cost=network_cost,
                network_capacity=int(total_demand),
                network_coverage_score=coverage_score,
                total_annual_demand=int(total_demand),
                construction_phases=construction_phases,
                investment_schedule=investment_schedule,
                roi_projection=roi_projection
            )
            
            logger.info(f"Network optimization complete. Total cost: ${network_cost:,.0f}, "
                       f"Coverage score: {coverage_score:.2f}")
            
            return network_optimization
            
        except Exception as e:
            logger.error(f"Network optimization failed: {e}")
            raise
    
    def _get_route_demand_data(self, route_key: str, demand_data: Dict) -> Dict:
        """Extract demand data for a specific route."""
        try:
            if hasattr(demand_data, 'city_pairs'):
                # Find matching demand pair
                for pair in demand_data.city_pairs:
                    pair_key = f"{pair.origin_city}-{pair.destination_city}"
                    reverse_key = f"{pair.destination_city}-{pair.origin_city}"
                    
                    if route_key == pair_key or route_key == reverse_key:
                        return {
                            'annual_demand': pair.total_annual_demand,
                            'peak_hourly_demand': pair.daily_passengers // 16,  # Assume 16-hour service day
                            'demand_density': pair.demand_density,
                            'revenue_potential': pair.revenue_potential
                        }
            
            # Fallback to direct lookup
            return demand_data.get(route_key, {})
            
        except Exception as e:
            logger.warning(f"Error getting route demand data: {e}")
            return {}
    
    def _get_route_annual_demand(self, optimized_route: OptimizedRoute) -> int:
        """Get annual demand for a route."""
        try:
            # This would normally come from demand analysis
            # For now, estimate based on revenue potential and fare
            revenue = optimized_route.annual_revenue_potential
            
            # Estimate average fare
            distance = optimized_route.route_option.total_distance_km
            if optimized_route.route_option.max_speed_kmh > 250:
                avg_fare = distance * 0.25
            elif optimized_route.route_option.max_speed_kmh > 160:
                avg_fare = distance * 0.15
            else:
                avg_fare = distance * 0.10
            
            if avg_fare > 0:
                return int(revenue / avg_fare)
            
            return 100000  # Default estimate
            
        except Exception as e:
            logger.warning(f"Error getting route annual demand: {e}")
            return 100000
    
    def _calculate_network_coverage_score(self, best_routes: Dict[str, OptimizedRoute]) -> float:
        """Calculate network coverage score based on connectivity."""
        try:
            if not best_routes:
                return 0.0
            
            # Extract unique cities from routes
            cities = set()
            for route_key in best_routes.keys():
                origin, destination = route_key.split('-', 1)
                cities.add(origin)
                cities.add(destination)
            
            num_cities = len(cities)
            num_routes = len(best_routes)
            
            # Calculate connectivity score
            # Maximum possible routes between n cities = n*(n-1)/2
            max_possible_routes = num_cities * (num_cities - 1) // 2 if num_cities > 1 else 1
            connectivity_ratio = num_routes / max_possible_routes
            
            # Score based on network density
            if connectivity_ratio >= 0.8:
                coverage_score = 10.0
            elif connectivity_ratio >= 0.6:
                coverage_score = 8.0
            elif connectivity_ratio >= 0.4:
                coverage_score = 6.0
            elif connectivity_ratio >= 0.2:
                coverage_score = 4.0
            else:
                coverage_score = 2.0
            
            return coverage_score
            
        except Exception as e:
            logger.warning(f"Error calculating network coverage score: {e}")
            return 5.0
    
    def _create_construction_phases(self, best_routes: Dict[str, OptimizedRoute], 
                                  total_budget: Optional[float]) -> List[Dict[str, Any]]:
        """Create phased construction plan based on priorities and budget."""
        try:
            # Sort routes by priority (combination of score and demand)
            route_priorities = []
            for route_key, route in best_routes.items():
                priority_score = (
                    route.overall_score * 0.6 +
                    route.demand_match_score * 0.4
                )
                
                route_priorities.append({
                    'route_key': route_key,
                    'route': route,
                    'priority_score': priority_score,
                    'cost': route.route_option.total_construction_cost,
                    'payback_period': route.payback_period_years
                })
            
            # Sort by priority score (descending) and payback period (ascending)
            route_priorities.sort(key=lambda x: (x['priority_score'], -x['payback_period']), reverse=True)
            
            # Create phases
            phases = []
            current_phase_cost = 0.0
            current_phase_routes = []
            phase_budget_limit = total_budget / 3 if total_budget else 10_000_000_000  # Default $10B per phase
            
            phase_number = 1
            
            for route_info in route_priorities:
                route_cost = route_info['cost']
                
                # Check if adding this route exceeds phase budget
                if current_phase_cost + route_cost > phase_budget_limit and current_phase_routes:
                    # Complete current phase
                    phases.append({
                        'phase': phase_number,
                        'routes': current_phase_routes.copy(),
                        'total_cost': current_phase_cost,
                        'duration_years': self._estimate_construction_duration(current_phase_routes),
                        'priority_level': 'high' if phase_number == 1 else 'medium' if phase_number == 2 else 'low'
                    })
                    
                    # Start new phase
                    phase_number += 1
                    current_phase_routes = []
                    current_phase_cost = 0.0
                
                # Add route to current phase
                current_phase_routes.append(route_info)
                current_phase_cost += route_cost
            
            # Add final phase if there are remaining routes
            if current_phase_routes:
                phases.append({
                    'phase': phase_number,
                    'routes': current_phase_routes,
                    'total_cost': current_phase_cost,
                    'duration_years': self._estimate_construction_duration(current_phase_routes),
                    'priority_level': 'high' if phase_number == 1 else 'medium' if phase_number == 2 else 'low'
                })
            
            return phases
            
        except Exception as e:
            logger.error(f"Error creating construction phases: {e}")
            return []
    
    def _estimate_construction_duration(self, phase_routes: List[Dict]) -> float:
        """Estimate construction duration for a phase."""
        try:
            if not phase_routes:
                return 0.0
            
            # Base construction time estimates
            base_years_per_km = {
                'surface': 0.2,      # 5 km per year
                'tunnel': 0.8,       # 1.25 km per year
                'bridge': 0.4,       # 2.5 km per year
                'viaduct': 0.3       # 3.3 km per year
            }
            
            max_duration = 0.0
            
            for route_info in phase_routes:
                route = route_info['route']
                infrastructure = route.route_option.infrastructure_summary
                
                # Calculate construction time for each infrastructure type
                duration = 0.0
                for infra_type, length_km in infrastructure.items():
                    if infra_type.endswith('_km'):
                        infra_key = infra_type[:-3]  # Remove '_km' suffix
                        years_per_km = base_years_per_km.get(infra_key, 0.3)
                        duration += length_km * years_per_km
                
                # Add base project setup time
                duration += 2.0  # 2 years for planning and setup
                
                # Routes can be built in parallel, so take the maximum
                max_duration = max(max_duration, duration)
            
            return min(max_duration, 15.0)  # Cap at 15 years
            
        except Exception as e:
            logger.warning(f"Error estimating construction duration: {e}")
            return 5.0
    
    def _create_investment_schedule(self, construction_phases: List[Dict]) -> Dict[int, float]:
        """Create year-by-year investment schedule."""
        try:
            schedule = {}
            current_year = 0
            
            for phase in construction_phases:
                phase_cost = phase['total_cost']
                phase_duration = phase['duration_years']
                
                # Distribute cost over construction period
                # Typical S-curve: 20% first third, 60% middle third, 20% final third
                if phase_duration <= 1:
                    # Single year project
                    schedule[current_year] = schedule.get(current_year, 0) + phase_cost
                else:
                    third_duration = phase_duration / 3
                    
                    # First third (20% of cost)
                    first_third_cost = phase_cost * 0.2
                    first_third_years = max(1, int(third_duration))
                    annual_cost_1 = first_third_cost / first_third_years
                    
                    for year in range(current_year, current_year + first_third_years):
                        schedule[year] = schedule.get(year, 0) + annual_cost_1
                    
                    # Middle third (60% of cost)
                    middle_third_cost = phase_cost * 0.6
                    middle_start = current_year + first_third_years
                    middle_third_years = max(1, int(third_duration))
                    annual_cost_2 = middle_third_cost / middle_third_years
                    
                    for year in range(middle_start, middle_start + middle_third_years):
                        schedule[year] = schedule.get(year, 0) + annual_cost_2
                    
                    # Final third (20% of cost)
                    final_third_cost = phase_cost * 0.2
                    final_start = middle_start + middle_third_years
                    final_third_years = max(1, int(phase_duration - first_third_years - middle_third_years))
                    annual_cost_3 = final_third_cost / final_third_years
                    
                    for year in range(final_start, final_start + final_third_years):
                        schedule[year] = schedule.get(year, 0) + annual_cost_3
                
                # Move to next phase start time (with some overlap allowed)
                current_year += max(1, int(phase_duration * 0.8))  # 20% overlap
            
            return schedule
            
        except Exception as e:
            logger.error(f"Error creating investment schedule: {e}")
            return {}
    
    def _calculate_roi_projection(self, best_routes: Dict[str, OptimizedRoute], 
                                investment_schedule: Dict[int, float]) -> Dict[str, float]:
        """Calculate return on investment projection."""
        try:
            total_investment = sum(investment_schedule.values())
            total_annual_revenue = sum(route.annual_revenue_potential for route in best_routes.values())
            total_annual_operating_cost = sum(route.annual_operating_cost for route in best_routes.values())
            
            annual_net_profit = total_annual_revenue - total_annual_operating_cost
            
            # Calculate various ROI metrics
            roi_metrics = {
                'simple_payback_years': total_investment / annual_net_profit if annual_net_profit > 0 else 999,
                'annual_roi_percent': (annual_net_profit / total_investment) * 100 if total_investment > 0 else 0,
                'npv_10_years': self._calculate_npv(annual_net_profit, total_investment, 10, 0.05),
                'irr_percent': self._estimate_irr(annual_net_profit, total_investment),
                'break_even_year': self._calculate_break_even_year(investment_schedule, annual_net_profit)
            }
            
            return roi_metrics
            
        except Exception as e:
            logger.error(f"Error calculating ROI projection: {e}")
            return {}
    
    def _calculate_npv(self, annual_cash_flow: float, initial_investment: float, 
                      years: int, discount_rate: float) -> float:
        """Calculate Net Present Value."""
        try:
            npv = -initial_investment
            
            for year in range(1, years + 1):
                discounted_cash_flow = annual_cash_flow / ((1 + discount_rate) ** year)
                npv += discounted_cash_flow
            
            return npv
            
        except Exception as e:
            logger.warning(f"Error calculating NPV: {e}")
            return 0.0
    
    def _estimate_irr(self, annual_cash_flow: float, initial_investment: float) -> float:
        """Estimate Internal Rate of Return using simple approximation."""
        try:
            if annual_cash_flow <= 0:
                return 0.0
            
            # Simple IRR approximation for perpetual cash flows
            # IRR ≈ annual_cash_flow / initial_investment
            irr = (annual_cash_flow / initial_investment) * 100
            
            return min(irr, 50.0)  # Cap at 50%
            
        except Exception as e:
            logger.warning(f"Error estimating IRR: {e}")
            return 0.0
    
    def _calculate_break_even_year(self, investment_schedule: Dict[int, float], 
                                 annual_net_profit: float) -> float:
        """Calculate when cumulative profits exceed cumulative investments."""
        try:
            if annual_net_profit <= 0:
                return 999.0
            
            cumulative_investment = 0.0
            cumulative_profit = 0.0
            
            max_year = max(investment_schedule.keys()) if investment_schedule else 0
            
            for year in range(max_year + 20):  # Check up to 20 years after construction
                # Add investment for this year
                yearly_investment = investment_schedule.get(year, 0)
                cumulative_investment += yearly_investment
                
                # Add profit for this year (starts after first operational year)
                if year > max_year:  # Operational years
                    cumulative_profit += annual_net_profit
                
                # Check if break-even achieved
                if cumulative_profit >= cumulative_investment:
                    return float(year)
            
            return 999.0  # No break-even within reasonable timeframe
            
        except Exception as e:
            logger.warning(f"Error calculating break-even year: {e}")
            return 999.0
    
    def get_optimization_summary(self, optimized_routes: Dict[str, List[OptimizedRoute]]) -> Dict[str, Any]:
        """Generate comprehensive optimization summary."""
        try:
            if not optimized_routes:
                return {'error': 'No optimized routes available'}
            
            all_routes = []
            for routes in optimized_routes.values():
                all_routes.extend(routes)
            
            if not all_routes:
                return {'error': 'No routes in optimization results'}
            
            # Calculate summary statistics
            scores = [route.overall_score for route in all_routes]
            costs = [route.route_option.total_construction_cost for route in all_routes]
            distances = [route.route_option.total_distance_km for route in all_routes]
            travel_times = [route.route_option.travel_time_hours for route in all_routes]
            
            summary = {
                'total_route_options': len(all_routes),
                'total_city_pairs': len(optimized_routes),
                'score_statistics': {
                    'min': round(min(scores), 3),
                    'max': round(max(scores), 3),
                    'average': round(sum(scores) / len(scores), 3),
                    'median': round(sorted(scores)[len(scores) // 2], 3)
                },
                'cost_statistics': {
                    'min': round(min(costs), 0),
                    'max': round(max(costs), 0),
                    'average': round(sum(costs) / len(costs), 0),
                    'total': round(sum(costs), 0)
                },
                'distance_statistics': {
                    'min': round(min(distances), 1),
                    'max': round(max(distances), 1),
                    'average': round(sum(distances) / len(distances), 1),
                    'total': round(sum(distances), 1)
                },
                'time_statistics': {
                    'min': round(min(travel_times), 2),
                    'max': round(max(travel_times), 2),
                    'average': round(sum(travel_times) / len(travel_times), 2)
                },
                'best_routes_by_criteria': self._get_best_routes_by_criteria(all_routes),
                'risk_distribution': self._analyze_risk_distribution(all_routes),
                'feasibility_analysis': self._analyze_feasibility_distribution(all_routes)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating optimization summary: {e}")
            return {'error': 'Could not generate optimization summary'}
    
    def _get_best_routes_by_criteria(self, all_routes: List[OptimizedRoute]) -> Dict[str, Dict]:
        """Get best routes for each optimization criterion."""
        try:
            best_routes = {}
            
            # Best overall score
            best_overall = max(all_routes, key=lambda r: r.overall_score)
            best_routes['overall'] = {
                'route': f"{best_overall.route_option.origin_city} - {best_overall.route_option.destination_city}",
                'name': best_overall.route_option.route_name,
                'score': round(best_overall.overall_score, 3)
            }
            
            # Best cost efficiency
            best_cost = max(all_routes, key=lambda r: r.cost_efficiency_score)
            best_routes['cost_efficiency'] = {
                'route': f"{best_cost.route_option.origin_city} - {best_cost.route_option.destination_city}",
                'name': best_cost.route_option.route_name,
                'score': round(best_cost.cost_efficiency_score, 3)
            }
            
            # Best time efficiency
            best_time = max(all_routes, key=lambda r: r.time_efficiency_score)
            best_routes['time_efficiency'] = {
                'route': f"{best_time.route_option.origin_city} - {best_time.route_option.destination_city}",
                'name': best_time.route_option.route_name,
                'score': round(best_time.time_efficiency_score, 3)
            }
            
            # Best environmental
            best_env = max(all_routes, key=lambda r: r.environmental_score)
            best_routes['environmental'] = {
                'route': f"{best_env.route_option.origin_city} - {best_env.route_option.destination_city}",
                'name': best_env.route_option.route_name,
                'score': round(best_env.environmental_score, 3)
            }
            
            # Shortest payback period
            best_payback = min(all_routes, key=lambda r: r.payback_period_years)
            best_routes['fastest_payback'] = {
                'route': f"{best_payback.route_option.origin_city} - {best_payback.route_option.destination_city}",
                'name': best_payback.route_option.route_name,
                'payback_years': round(best_payback.payback_period_years, 1)
            }
            
            return best_routes
            
        except Exception as e:
            logger.warning(f"Error getting best routes by criteria: {e}")
            return {}
    
    def _analyze_risk_distribution(self, all_routes: List[OptimizedRoute]) -> Dict[str, Any]:
        """Analyze risk distribution across all routes."""
        try:
            risk_levels = []
            for route in all_routes:
                if route.risk_assessment and 'risk_level' in route.risk_assessment:
                    risk_levels.append(route.risk_assessment['risk_level'])
                else:
                    risk_levels.append('unknown')
            
            # Count risk levels
            risk_counts = {}
            for level in ['low', 'medium', 'high', 'extreme', 'unknown']:
                risk_counts[level] = risk_levels.count(level)
            
            # Calculate percentages
            total_routes = len(risk_levels)
            risk_percentages = {
                level: round((count / total_routes) * 100, 1) if total_routes > 0 else 0
                for level, count in risk_counts.items()
            }
            
            return {
                'risk_counts': risk_counts,
                'risk_percentages': risk_percentages,
                'total_routes': total_routes
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing risk distribution: {e}")
            return {}
    
    def _analyze_feasibility_distribution(self, all_routes: List[OptimizedRoute]) -> Dict[str, Any]:
        """Analyze feasibility distribution across all routes."""
        try:
            feasibility_scores = [route.feasibility_score for route in all_routes]
            
            # Categorize feasibility
            categories = {'high': 0, 'medium': 0, 'low': 0}
            
            for score in feasibility_scores:
                if score >= 7.0:
                    categories['high'] += 1
                elif score >= 4.0:
                    categories['medium'] += 1
                else:
                    categories['low'] += 1
            
            total_routes = len(feasibility_scores)
            percentages = {
                category: round((count / total_routes) * 100, 1) if total_routes > 0 else 0
                for category, count in categories.items()
            }
            
            return {
                'feasibility_counts': categories,
                'feasibility_percentages': percentages,
                'average_feasibility': round(sum(feasibility_scores) / len(feasibility_scores), 2) if feasibility_scores else 0
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing feasibility distribution: {e}")
            return {}
