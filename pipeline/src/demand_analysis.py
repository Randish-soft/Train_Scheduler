"""
BCPC Pipeline: Demand Analysis Module

This module analyzes passenger demand between cities based on population,
tourism indices, economic factors, and gravity models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from math import radians, cos, sin, asin, sqrt, exp, log
import warnings

# Suppress scipy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)

@dataclass
class DemandPair:
    """Represents demand between two cities."""
    origin_city: str
    destination_city: str
    distance_km: float
    daily_passengers: int
    commuter_demand: int
    tourism_demand: int
    business_demand: int
    total_annual_demand: int
    demand_density: float  # passengers per km
    revenue_potential: float
    seasonality_factor: float = 1.0
    growth_potential: float = 1.0

@dataclass
class DemandAnalysis:
    """Complete demand analysis results."""
    city_pairs: List[DemandPair] = field(default_factory=list)
    total_network_demand: int = 0
    average_trip_distance: float = 0.0
    peak_corridor_demand: int = 0
    demand_distribution: Dict[str, float] = field(default_factory=dict)
    economic_factors: Dict[str, Any] = field(default_factory=dict)
    
class DemandAnalyzer:
    """Analyzes passenger demand between cities using multiple models."""
    
    def __init__(self):
        """Initialize the demand analyzer with model parameters."""
        # Gravity model parameters
        self.gravity_alpha = 0.8  # Population attraction exponent
        self.gravity_beta = -1.2  # Distance decay exponent
        self.gravity_k = 1000     # Scaling constant
        
        # Trip purpose distributions
        self.trip_purposes = {
            'commuter': 0.4,    # 40% commuting
            'tourism': 0.35,    # 35% tourism/leisure
            'business': 0.25    # 25% business travel
        }
        
        # Distance-based demand modifiers
        self.distance_thresholds = {
            'short': (0, 100),      # < 100km - high commuter demand
            'medium': (100, 300),   # 100-300km - mixed demand
            'long': (300, 800),     # 300-800km - tourism/business
            'very_long': (800, 9999) # > 800km - mainly business/tourism
        }
        
        # Economic parameters
        self.gdp_elasticity = 0.6
        self.tourism_multiplier = 1.5
        self.population_threshold = 50000  # Minimum population for analysis
        
    def analyze_demand(self, cities: List) -> DemandAnalysis:
        """
        Analyze demand between all city pairs.
        
        Args:
            cities: List of CityData objects
            
        Returns:
            DemandAnalysis object with complete results
        """
        try:
            logger.info(f"Starting demand analysis for {len(cities)} cities")
            
            # Filter cities by minimum population
            valid_cities = [city for city in cities if city.population >= self.population_threshold]
            logger.info(f"Analyzing {len(valid_cities)} cities meeting population threshold")
            
            if len(valid_cities) < 2:
                raise ValueError("Need at least 2 cities meeting population threshold for demand analysis")
            
            # Calculate demand for all city pairs
            demand_pairs = []
            total_demand = 0
            
            for i, origin in enumerate(valid_cities):
                for j, destination in enumerate(valid_cities):
                    if i >= j:  # Avoid duplicates and self-pairs
                        continue
                    
                    try:
                        demand_pair = self._calculate_pair_demand(origin, destination)
                        demand_pairs.append(demand_pair)
                        total_demand += demand_pair.total_annual_demand
                        
                    except Exception as e:
                        logger.warning(f"Error calculating demand between {origin.name} and {destination.name}: {e}")
                        continue
            
            # Sort by demand (highest first)
            demand_pairs.sort(key=lambda x: x.total_annual_demand, reverse=True)
            
            # Calculate network statistics
            analysis = self._calculate_network_statistics(demand_pairs, total_demand)
            
            logger.info(f"Demand analysis complete. Total network demand: {total_demand:,} annual passengers")
            logger.info(f"Peak corridor: {demand_pairs[0].origin_city} - {demand_pairs[0].destination_city} "
                       f"({demand_pairs[0].total_annual_demand:,} passengers)")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Demand analysis failed: {e}")
            raise
    
    def _calculate_pair_demand(self, origin, destination) -> DemandPair:
        """Calculate demand between a specific city pair."""
        try:
            # Calculate distance
            distance = self._haversine_distance(
                origin.latitude, origin.longitude,
                destination.latitude, destination.longitude
            )
            
            # Apply gravity model for base demand
            base_demand = self._gravity_model(
                origin.population, destination.population, distance
            )
            
            # Apply distance-based modifiers
            distance_category = self._get_distance_category(distance)
            demand_modifier = self._get_distance_modifier(distance_category)
            
            # Calculate trip purpose breakdown
            commuter_demand = int(base_demand * self.trip_purposes['commuter'] * demand_modifier['commuter'])
            tourism_demand = int(base_demand * self.trip_purposes['tourism'] * demand_modifier['tourism'])
            business_demand = int(base_demand * self.trip_purposes['business'] * demand_modifier['business'])
            
            # Apply tourism multiplier
            tourism_factor = (origin.tourism_index + destination.tourism_index) / 2
            tourism_demand = int(tourism_demand * (1 + tourism_factor * self.tourism_multiplier))
            
            # Apply economic factors
            economic_factor = self._calculate_economic_factor(origin, destination)
            total_daily = int((commuter_demand + tourism_demand + business_demand) * economic_factor)
            
            # Annual demand (accounting for seasonal variations)
            seasonality = self._calculate_seasonality_factor(origin, destination, tourism_factor)
            total_annual = int(total_daily * 365 * seasonality)
            
            # Calculate revenue potential (simplified model)
            revenue_potential = self._calculate_revenue_potential(distance, total_annual, distance_category)
            
            # Calculate growth potential
            growth_potential = self._calculate_growth_potential(origin, destination, distance_category)
            
            return DemandPair(
                origin_city=origin.name,
                destination_city=destination.name,
                distance_km=distance,
                daily_passengers=total_daily,
                commuter_demand=commuter_demand,
                tourism_demand=tourism_demand,
                business_demand=business_demand,
                total_annual_demand=total_annual,
                demand_density=total_annual / distance if distance > 0 else 0,
                revenue_potential=revenue_potential,
                seasonality_factor=seasonality,
                growth_potential=growth_potential
            )
            
        except Exception as e:
            logger.error(f"Error calculating demand pair: {e}")
            raise
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points in kilometers."""
        try:
            # Convert to radians
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            
            # Earth radius in kilometers
            r = 6371
            
            return c * r
            
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return 0.0
    
    def _gravity_model(self, pop1: int, pop2: int, distance: float) -> float:
        """Apply gravity model to calculate base demand."""
        try:
            if distance == 0:
                return 0
            
            # Modified gravity model: Demand = k * (P1^α * P2^α) / (distance^β)
            numerator = (pop1 ** self.gravity_alpha) * (pop2 ** self.gravity_alpha)
            denominator = distance ** abs(self.gravity_beta)
            
            base_demand = self.gravity_k * (numerator / denominator)
            
            # Apply realistic bounds
            return max(0, min(base_demand, 50000))  # Cap at 50k daily passengers
            
        except Exception as e:
            logger.error(f"Error in gravity model: {e}")
            return 0.0
    
    def _get_distance_category(self, distance: float) -> str:
        """Categorize distance for applying appropriate demand modifiers."""
        for category, (min_dist, max_dist) in self.distance_thresholds.items():
            if min_dist <= distance < max_dist:
                return category
        return 'very_long'
    
    def _get_distance_modifier(self, distance_category: str) -> Dict[str, float]:
        """Get demand modifiers based on distance category."""
        modifiers = {
            'short': {
                'commuter': 2.0,   # High commuter demand for short distances
                'tourism': 0.5,    # Low tourism for short trips
                'business': 0.8    # Moderate business travel
            },
            'medium': {
                'commuter': 1.2,   # Some commuter demand
                'tourism': 1.3,    # Good tourism potential
                'business': 1.5    # High business travel
            },
            'long': {
                'commuter': 0.3,   # Very low commuter demand
                'tourism': 1.8,    # High tourism potential
                'business': 1.7    # High business travel
            },
            'very_long': {
                'commuter': 0.1,   # Minimal commuter demand
                'tourism': 1.5,    # Tourism for special destinations
                'business': 2.0    # Primarily business travel
            }
        }
        
        return modifiers.get(distance_category, modifiers['medium'])
    
    def _calculate_economic_factor(self, origin, destination) -> float:
        """Calculate economic factor based on city budgets and economic indicators."""
        try:
            # Use budget as proxy for economic activity
            avg_budget = (origin.budget + destination.budget) / 2
            
            # Normalize budget (assuming budget is in millions USD)
            if avg_budget <= 0:
                return 0.5  # Low economic factor for cities without budget data
            
            # Economic factor based on budget levels
            if avg_budget < 100:
                return 0.6
            elif avg_budget < 500:
                return 0.8
            elif avg_budget < 1000:
                return 1.0
            elif avg_budget < 5000:
                return 1.2
            else:
                return 1.5
                
        except Exception as e:
            logger.warning(f"Error calculating economic factor: {e}")
            return 1.0
    
    def _calculate_seasonality_factor(self, origin, destination, tourism_factor: float) -> float:
        """Calculate seasonality factor based on tourism levels."""
        try:
            # Base seasonality factor
            base_factor = 1.0
            
            # Higher tourism destinations have more seasonal variation
            if tourism_factor > 0.7:
                base_factor = 1.3  # 30% increase due to peak seasons
            elif tourism_factor > 0.4:
                base_factor = 1.15  # 15% increase
            
            # Business corridors have less seasonality
            if origin.population > 1000000 and destination.population > 1000000:
                base_factor *= 0.9  # Reduce seasonality for major business centers
            
            return base_factor
            
        except Exception as e:
            logger.warning(f"Error calculating seasonality factor: {e}")
            return 1.0
    
    def _calculate_revenue_potential(self, distance: float, annual_demand: int, distance_category: str) -> float:
        """Calculate revenue potential based on distance and demand."""
        try:
            # Base fare per km (simplified pricing model)
            fare_per_km = {
                'short': 0.15,      # $0.15 per km for short trips
                'medium': 0.12,     # $0.12 per km for medium trips
                'long': 0.10,       # $0.10 per km for long trips
                'very_long': 0.08   # $0.08 per km for very long trips
            }
            
            base_fare = distance * fare_per_km.get(distance_category, 0.10)
            annual_revenue = base_fare * annual_demand
            
            return annual_revenue
            
        except Exception as e:
            logger.warning(f"Error calculating revenue potential: {e}")
            return 0.0
    
    def _calculate_growth_potential(self, origin, destination, distance_category: str) -> float:
        """Calculate growth potential based on city characteristics."""
        try:
            growth_factor = 1.0
            
            # Population-based growth potential
            total_population = origin.population + destination.population
            if total_population > 2000000:
                growth_factor *= 1.3
            elif total_population > 1000000:
                growth_factor *= 1.2
            elif total_population > 500000:
                growth_factor *= 1.1
            
            # Tourism growth potential
            avg_tourism = (origin.tourism_index + destination.tourism_index) / 2
            if avg_tourism > 0.6:
                growth_factor *= 1.2
            elif avg_tourism > 0.3:
                growth_factor *= 1.1
            
            # Distance-based growth potential
            if distance_category in ['medium', 'long']:
                growth_factor *= 1.15  # Medium and long distances have good growth potential
            
            return min(growth_factor, 2.0)  # Cap at 2x growth potential
            
        except Exception as e:
            logger.warning(f"Error calculating growth potential: {e}")
            return 1.0
    
    def _calculate_network_statistics(self, demand_pairs: List[DemandPair], total_demand: int) -> DemandAnalysis:
        """Calculate network-level statistics from demand pairs."""
        try:
            if not demand_pairs:
                return DemandAnalysis()
            
            # Calculate average trip distance
            total_distance = sum(pair.distance_km * pair.total_annual_demand for pair in demand_pairs)
            avg_distance = total_distance / total_demand if total_demand > 0 else 0
            
            # Find peak corridor demand
            peak_demand = max(pair.total_annual_demand for pair in demand_pairs)
            
            # Calculate demand distribution by distance category
            distribution = {'short': 0, 'medium': 0, 'long': 0, 'very_long': 0}
            for pair in demand_pairs:
                category = self._get_distance_category(pair.distance_km)
                distribution[category] += pair.total_annual_demand
            
            # Normalize distribution
            if total_demand > 0:
                distribution = {k: v / total_demand for k, v in distribution.items()}
            
            # Calculate economic factors
            economic_factors = {
                'total_revenue_potential': sum(pair.revenue_potential for pair in demand_pairs),
                'average_revenue_per_passenger': sum(pair.revenue_potential for pair in demand_pairs) / total_demand if total_demand > 0 else 0,
                'high_demand_corridors': len([p for p in demand_pairs if p.total_annual_demand > 100000]),
                'average_growth_potential': sum(pair.growth_potential for pair in demand_pairs) / len(demand_pairs)
            }
            
            return DemandAnalysis(
                city_pairs=demand_pairs,
                total_network_demand=total_demand,
                average_trip_distance=avg_distance,
                peak_corridor_demand=peak_demand,
                demand_distribution=distribution,
                economic_factors=economic_factors
            )
            
        except Exception as e:
            logger.error(f"Error calculating network statistics: {e}")
            return DemandAnalysis()
    
    def get_top_corridors(self, demand_analysis: DemandAnalysis, top_n: int = 10) -> List[DemandPair]:
        """Get the top N corridors by demand."""
        return demand_analysis.city_pairs[:top_n]
    
    def filter_corridors_by_demand(self, demand_analysis: DemandAnalysis, min_annual_demand: int) -> List[DemandPair]:
        """Filter corridors by minimum annual demand threshold."""
        return [pair for pair in demand_analysis.city_pairs if pair.total_annual_demand >= min_annual_demand]
    
    def get_demand_summary(self, demand_analysis: DemandAnalysis) -> Dict[str, Any]:
        """Generate a comprehensive summary of demand analysis."""
        if not demand_analysis.city_pairs:
            return {'error': 'No demand data available'}
        
        pairs = demand_analysis.city_pairs
        
        return {
            'total_corridors_analyzed': len(pairs),
            'total_network_demand': demand_analysis.total_network_demand,
            'average_trip_distance_km': round(demand_analysis.average_trip_distance, 1),
            'peak_corridor_demand': demand_analysis.peak_corridor_demand,
            'demand_distribution': demand_analysis.demand_distribution,
            'economic_metrics': demand_analysis.economic_factors,
            'top_5_corridors': [
                {
                    'route': f"{pair.origin_city} - {pair.destination_city}",
                    'distance_km': round(pair.distance_km, 1),
                    'annual_demand': pair.total_annual_demand,
                    'daily_demand': pair.daily_passengers,
                    'revenue_potential': round(pair.revenue_potential, 0)
                }
                for pair in pairs[:5]
            ],
            'demand_statistics': {
                'min_corridor_demand': min(pair.total_annual_demand for pair in pairs),
                'max_corridor_demand': max(pair.total_annual_demand for pair in pairs),
                'median_corridor_demand': sorted([pair.total_annual_demand for pair in pairs])[len(pairs) // 2],
                'average_corridor_demand': round(sum(pair.total_annual_demand for pair in pairs) / len(pairs), 0)
            }
        }