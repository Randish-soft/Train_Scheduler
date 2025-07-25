"""
Demand Processing Module
========================

Analyzes passenger demand between cities based on:
- Population sizes and demographics
- Geographic distances
- Economic relationships
- Historical travel patterns

Author: Miguel Ibrahim E
"""

import math
import logging
from typing import List, Dict, Any, Tuple
import numpy as np


class DemandProcessor:
    """Analyzes and processes passenger demand data between cities."""
    
    def __init__(self, cities_data: List[Dict[str, Any]]):
        self.cities = cities_data
        self.logger = logging.getLogger(__name__)
        
    def analyze(self) -> Dict[str, Any]:
        """
        Main analysis method that calculates demand between all city pairs.
        
        Returns:
            Dictionary containing demand analysis results
        """
        self.logger.info("📊 Analyzing passenger demand patterns...")
        
        # Calculate distance matrix
        distance_matrix = self._calculate_distance_matrix()
        
        # Calculate demand matrix
        demand_matrix = self._calculate_demand_matrix(distance_matrix)
        
        # Identify high-demand routes
        high_demand_routes = self._identify_high_demand_routes(demand_matrix)
        
        # Calculate market potential
        market_analysis = self._analyze_market_potential()
        
        # Generate demand clusters
        demand_clusters = self._generate_demand_clusters(demand_matrix)
        
        results = {
            'distance_matrix': distance_matrix,
            'demand_matrix': demand_matrix,
            'high_demand_routes': high_demand_routes,
            'market_analysis': market_analysis,
            'demand_clusters': demand_clusters,
            'total_population': sum(city['population'] for city in self.cities),
            'analysis_summary': self._generate_analysis_summary(demand_matrix)
        }
        
        self.logger.info("✅ Demand analysis completed")
        return results
    
    def _calculate_distance_matrix(self) -> Dict[str, Dict[str, float]]:
        """Calculate distances between all city pairs using Haversine formula."""
        self.logger.info("🗺️ Calculating distance matrix...")
        
        distance_matrix = {}
        
        for i, city1 in enumerate(self.cities):
            city1_name = city1['city_name']
            distance_matrix[city1_name] = {}
            
            for j, city2 in enumerate(self.cities):
                if i == j:
                    distance_matrix[city1_name][city2['city_name']] = 0.0
                else:
                    distance = self._haversine_distance(
                        city1['latitude'], city1['longitude'],
                        city2['latitude'], city2['longitude']
                    )
                    distance_matrix[city1_name][city2['city_name']] = distance
        
        return distance_matrix
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on the earth.
        
        Args:
            lat1, lon1: Latitude and longitude of first point in decimal degrees
            lat2, lon2: Latitude and longitude of second point in decimal degrees
            
        Returns:
            Distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        return c * r
    
    def _calculate_demand_matrix(self, distance_matrix: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        """Calculate passenger demand between city pairs."""
        self.logger.info("👥 Calculating passenger demand matrix...")
        
        demand_matrix = {}
        
        for city1 in self.cities:
            city1_name = city1['city_name']
            demand_matrix[city1_name] = {}
            
            for city2 in self.cities:
                city2_name = city2['city_name']
                
                if city1_name == city2_name:
                    continue
                
                # Calculate demand using gravity model
                demand_data = self._gravity_model_demand(
                    city1, city2, distance_matrix[city1_name][city2_name]
                )
                
                demand_matrix[city1_name][city2_name] = demand_data
        
        return demand_matrix
    
    def _gravity_model_demand(self, city1: Dict, city2: Dict, distance: float) -> Dict[str, Any]:
        """
        Calculate demand using gravity model: Demand ∝ (Pop1 × Pop2) / Distance²
        
        Args:
            city1, city2: City data dictionaries
            distance: Distance between cities in km
            
        Returns:
            Dictionary with demand metrics
        """
        if distance == 0:
            return {'demand_score': 0, 'daily_passengers': 0, 'annual_passengers': 0}
        
        # Basic gravity model
        pop1 = city1['population']
        pop2 = city2['population']
        
        # Gravity model with adjustments
        gravity_base = (pop1 * pop2) / (distance ** 2)
        
        # Apply scaling factors
        scaling_factor = 1e-9  # Adjust for reasonable numbers
        
        # Distance decay function (more realistic than pure inverse square)
        if distance <= 50:
            distance_factor = 1.0  # Short distance, high factor
        elif distance <= 150:
            distance_factor = 0.8  # Medium distance
        elif distance <= 300:
            distance_factor = 0.6  # Long distance
        else:
            distance_factor = 0.4  # Very long distance
        
        # Economic factors
        economic_factor = self._calculate_economic_factor(city1, city2)
        
        # Capital city bonus
        capital_bonus = 1.5 if (city1.get('is_capital') or city2.get('is_capital')) else 1.0
        
        # Calculate base demand score
        demand_score = gravity_base * scaling_factor * distance_factor * economic_factor * capital_bonus
        
        # Convert to passenger estimates
        daily_passengers = max(1, int(demand_score * 1000))
        annual_passengers = daily_passengers * 365
        
        # Calculate revenue potential (rough estimate)
        avg_ticket_price = self._estimate_ticket_price(distance)
        annual_revenue = annual_passengers * avg_ticket_price
        
        return {
            'demand_score': demand_score,
            'daily_passengers': daily_passengers,
            'annual_passengers': annual_passengers,
            'distance_km': distance,
            'population_served': pop1 + pop2,
            'economic_factor': economic_factor,
            'estimated_annual_revenue': annual_revenue,
            'avg_ticket_price': avg_ticket_price,
            'route_priority': self._calculate_route_priority(demand_score, distance, pop1 + pop2)
        }
    
    def _calculate_economic_factor(self, city1: Dict, city2: Dict) -> float:
        """Calculate economic relationship factor between cities."""
        # Simple economic factor based on population sizes
        # In a real implementation, this would use GDP, economic data, etc.
        
        pop1 = city1['population']
        pop2 = city2['population']
        
        # Larger cities have stronger economic relationships
        if pop1 > 1000000 and pop2 > 1000000:
            return 1.8  # Major metropolitan areas
        elif pop1 > 500000 or pop2 > 500000:
            return 1.4  # At least one major city
        elif pop1 > 100000 and pop2 > 100000:
            return 1.2  # Both medium cities
        else:
            return 1.0  # Smaller cities
    
    def _estimate_ticket_price(self, distance: float) -> float:
        """Estimate ticket price based on distance."""
        # Simple price model: base price + distance-based component
        base_price = 5.0  # Base price in USD
        price_per_km = 0.08  # Price per km in USD
        
        return base_price + (distance * price_per_km)
    
    def _calculate_route_priority(self, demand_score: float, distance: float, total_population: int) -> str:
        """Calculate route priority level."""
        # Normalize factors for priority calculation
        demand_factor = min(demand_score * 1000, 1.0)  # Normalize demand
        distance_factor = 1.0 if distance <= 300 else 0.5  # Favor shorter routes
        population_factor = min(total_population / 2000000, 1.0)  # Normalize population
        
        priority_score = (demand_factor * 0.5) + (distance_factor * 0.3) + (population_factor * 0.2)
        
        if priority_score >= 0.7:
            return "High"
        elif priority_score >= 0.4:
            return "Medium"
        else:
            return "Low"
    
    def _identify_high_demand_routes(self, demand_matrix: Dict) -> List[Dict[str, Any]]:
        """Identify and rank high-demand routes."""
        self.logger.info("🚀 Identifying high-demand routes...")
        
        all_routes = []
        
        # Collect all routes with their demand data
        for city1_name, destinations in demand_matrix.items():
            for city2_name, demand_data in destinations.items():
                route_info = demand_data.copy()
                route_info['route'] = f"{city1_name} - {city2_name}"
                route_info['origin'] = city1_name
                route_info['destination'] = city2_name
                all_routes.append(route_info)
        
        # Sort by demand score
        high_demand_routes = sorted(
            all_routes, 
            key=lambda x: x['demand_score'], 
            reverse=True
        )
        
        # Return top routes
        return high_demand_routes[:15]
    
    def _analyze_market_potential(self) -> Dict[str, Any]:
        """Analyze overall market potential."""
        self.logger.info("💰 Analyzing market potential...")
        
        total_population = sum(city['population'] for city in self.cities)
        
        # Calculate potential passenger trips per year
        # Assumption: average person takes 2-3 train trips per year in a developed network
        trips_per_person_per_year = 2.5
        potential_annual_trips = int(total_population * trips_per_person_per_year)
        
        # Calculate market segments
        major_cities = [city for city in self.cities if city['population'] > 500000]
        medium_cities = [city for city in self.cities if 100000 <= city['population'] <= 500000]
        smaller_cities = [city for city in self.cities if city['population'] < 100000]
        
        # Estimate total market value
        avg_trip_price = 25.0  # USD
        total_market_value = potential_annual_trips * avg_trip_price
        
        return {
            'total_population': total_population,
            'potential_annual_trips': potential_annual_trips,
            'total_market_value_usd': total_market_value,
            'major_cities_count': len(major_cities),
            'medium_cities_count': len(medium_cities),
            'smaller_cities_count': len(smaller_cities),
            'market_segments': {
                'intercity_premium': total_market_value * 0.4,  # 40% premium routes
                'regional_standard': total_market_value * 0.45,  # 45% standard routes
                'local_economy': total_market_value * 0.15  # 15% economy routes
            }
        }
    
    def _generate_demand_clusters(self, demand_matrix: Dict) -> Dict[str, Any]:
        """Generate demand-based city clusters for network planning."""
        self.logger.info("🌐 Generating demand clusters...")
        
        # Simple clustering based on geographic proximity and demand
        clusters = []
        used_cities = set()
        
        # Sort cities by population (largest first)
        sorted_cities = sorted(self.cities, key=lambda x: x['population'], reverse=True)
        
        for anchor_city in sorted_cities:
            if anchor_city['city_name'] in used_cities:
                continue
                
            cluster = {
                'anchor_city': anchor_city['city_name'],
                'cities': [anchor_city['city_name']],
                'total_population': anchor_city['population'],
                'cluster_type': self._determine_cluster_type(anchor_city['population'])
            }
            
            used_cities.add(anchor_city['city_name'])
            
            # Find nearby cities with good connectivity
            for other_city in sorted_cities:
                if other_city['city_name'] in used_cities:
                    continue
                    
                # Check if cities should be in same cluster
                if self._should_cluster_cities(anchor_city, other_city, demand_matrix):
                    cluster['cities'].append(other_city['city_name'])
                    cluster['total_population'] += other_city['population']
                    used_cities.add(other_city['city_name'])
            
            clusters.append(cluster)
            
            # Limit number of clusters
            if len(clusters) >= 5:
                break
        
        return {
            'clusters': clusters,
            'total_clusters': len(clusters),
            'cluster_connectivity': self._analyze_cluster_connectivity(clusters, demand_matrix)
        }
    
    def _determine_cluster_type(self, population: int) -> str:
        """Determine cluster type based on anchor city population."""
        if population > 2000000:
            return "Metropolitan"
        elif population > 500000:
            return "Major Urban"
        elif population > 100000:
            return "Regional"
        else:
            return "Local"
    
    def _should_cluster_cities(self, anchor_city: Dict, other_city: Dict, demand_matrix: Dict) -> bool:
        """Determine if two cities should be in the same cluster."""
        anchor_name = anchor_city['city_name']
        other_name = other_city['city_name']
        
        # Check if demand data exists
        if (anchor_name in demand_matrix and 
            other_name in demand_matrix[anchor_name]):
            
            demand_data = demand_matrix[anchor_name][other_name]
            distance = demand_data.get('distance_km', float('inf'))
            demand_score = demand_data.get('demand_score', 0)
            
            # Cluster criteria: close distance and reasonable demand
            return distance <= 150 and demand_score > 0.1
        
        return False
    
    def _analyze_cluster_connectivity(self, clusters: List[Dict], demand_matrix: Dict) -> Dict[str, Any]:
        """Analyze connectivity between clusters."""
        inter_cluster_routes = []
        
        for i, cluster1 in enumerate(clusters):
            for j, cluster2 in enumerate(clusters[i+1:], i+1):
                # Find best connection between clusters
                best_route = None
                best_demand = 0
                
                for city1 in cluster1['cities']:
                    for city2 in cluster2['cities']:
                        if (city1 in demand_matrix and 
                            city2 in demand_matrix[city1]):
                            
                            demand_data = demand_matrix[city1][city2]
                            demand_score = demand_data.get('demand_score', 0)
                            
                            if demand_score > best_demand:
                                best_demand = demand_score
                                best_route = {
                                    'route': f"{city1} - {city2}",
                                    'cluster1': cluster1['anchor_city'],
                                    'cluster2': cluster2['anchor_city'],
                                    'demand_score': demand_score,
                                    'distance_km': demand_data.get('distance_km', 0)
                                }
                
                if best_route:
                    inter_cluster_routes.append(best_route)
        
        return {
            'inter_cluster_routes': sorted(inter_cluster_routes, 
                                         key=lambda x: x['demand_score'], 
                                         reverse=True),
            'total_connections': len(inter_cluster_routes)
        }
    
    def _generate_analysis_summary(self, demand_matrix: Dict) -> Dict[str, Any]:
        """Generate summary statistics for the demand analysis."""
        all_demands = []
        all_distances = []
        all_revenues = []
        
        for city1_routes in demand_matrix.values():
            for route_data in city1_routes.values():
                all_demands.append(route_data.get('demand_score', 0))
                all_distances.append(route_data.get('distance_km', 0))
                all_revenues.append(route_data.get('estimated_annual_revenue', 0))
        
        if not all_demands:
            return {'error': 'No demand data available'}
        
        return {
            'total_routes_analyzed': len(all_demands),
            'average_demand_score': sum(all_demands) / len(all_demands),
            'max_demand_score': max(all_demands),
            'average_route_distance': sum(all_distances) / len(all_distances),
            'total_potential_revenue': sum(all_revenues),
            'high_priority_routes': len([d for d in all_demands if d > 0.5]),
            'medium_priority_routes': len([d for d in all_demands if 0.2 <= d <= 0.5]),
            'low_priority_routes': len([d for d in all_demands if d < 0.2])
        }