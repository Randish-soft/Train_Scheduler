import logging
import numpy as np
from typing import Dict, Any, List, Tuple

class DemandAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze transportation demand across the country"""
        self.logger.info("Analyzing transportation demand")
        
        country_data = context['country_data']
        country_features = context['country_features']
        user_priorities = context['user_priorities']
        
        # Analyze population distribution
        population_analysis = self._analyze_population_distribution(country_data)
        
        # Analyze travel patterns
        travel_patterns = self._analyze_travel_patterns(country_data, population_analysis)
        
        # Calculate demand scores for different corridors
        demand_corridors = self._identify_demand_corridors(population_analysis, travel_patterns, user_priorities)
        
        context['demand_data'] = {
            'population_analysis': population_analysis,
            'travel_patterns': travel_patterns,
            'demand_corridors': demand_corridors,
            'priority_routes': self._prioritize_routes(demand_corridors, user_priorities)
        }
        
        self.logger.info(f"Identified {len(demand_corridors)} demand corridors")
        return context
    
    def _analyze_population_distribution(self, country_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how population is distributed across the country"""
        cities = country_data.get('cities', [])
        total_population = country_data['population']
        
        if not cities:
            # Generate synthetic cities if none provided
            cities = self._generate_synthetic_cities(total_population, country_data['area'])
        
        # Calculate population metrics
        city_populations = [city.get('population', 0) for city in cities]
        urban_population = sum(city_populations)
        rural_population = max(0, total_population - urban_population)
        
        # Identify major urban centers
        major_cities = [city for city in cities if city.get('population', 0) > total_population * 0.05]
        
        return {
            'cities': cities,
            'total_population': total_population,
            'urban_population': urban_population,
            'rural_population': rural_population,
            'urbanization_rate': urban_population / total_population if total_population > 0 else 0,
            'major_cities': major_cities,
            'population_density_map': self._create_population_density_map(cities, country_data['area'])
        }
    
    def _generate_synthetic_cities(self, total_population: float, area: float) -> List[Dict[str, Any]]:
        """Generate synthetic city data if no real data provided"""
        num_cities = max(5, int(total_population / 500000))  # Rough heuristic
        cities = []
        
        # Capital city (largest)
        capital_pop = total_population * 0.2
        cities.append({
            'name': 'Capital City',
            'population': capital_pop,
            'is_capital': True,
            'location': {'x': 0.5, 'y': 0.5}  # Center of country
        })
        
        # Other major cities
        remaining_pop = total_population - capital_pop
        for i in range(num_cities - 1):
            pop_share = np.random.beta(2, 5)  # Most cities are smaller
            city_pop = remaining_pop * pop_share * 0.8  # Leave some for rural
            cities.append({
                'name': f'City {i+1}',
                'population': city_pop,
                'is_capital': False,
                'location': {
                    'x': np.random.uniform(0.1, 0.9),
                    'y': np.random.uniform(0.1, 0.9)
                }
            })
        
        return cities
    
    def _create_population_density_map(self, cities: List[Dict[str, Any]], area: float) -> Dict[str, Any]:
        """Create a simplified population density map"""
        # This would typically use GIS data in real implementation
        grid_size = 10
        density_grid = np.zeros((grid_size, grid_size))
        
        for city in cities:
            location = city.get('location', {'x': 0.5, 'y': 0.5})
            pop = city.get('population', 0)
            
            # Convert location to grid coordinates
            grid_x = int(location['x'] * (grid_size - 1))
            grid_y = int(location['y'] * (grid_size - 1))
            
            # Distribute population to surrounding cells (simplified)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    x = max(0, min(grid_size - 1, grid_x + dx))
                    y = max(0, min(grid_size - 1, grid_y + dy))
                    weight = 0.5 if abs(dx) + abs(dy) > 0 else 1.0
                    density_grid[y, x] += pop * weight / 9  # Distribute across 9 cells
        
        return {
            'grid': density_grid.tolist(),
            'max_density': np.max(density_grid),
            'min_density': np.min(density_grid[density_grid > 0]) if np.any(density_grid > 0) else 0
        }
    
    def _analyze_travel_patterns(self, country_data: Dict[str, Any], population_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze typical travel patterns and distances"""
        cities = population_analysis['cities']
        
        # Calculate inter-city travel volumes (simplified)
        travel_matrix = {}
        for i, city1 in enumerate(cities):
            for j, city2 in enumerate(cities):
                if i != j:
                    key = f"{city1['name']}-{city2['name']}"
                    # Simple gravity model: travel ~ (pop1 * pop2) / distance^2
                    pop1 = city1['population']
                    pop2 = city2['population']
                    
                    # Estimate distance from locations
                    loc1 = city1.get('location', {'x': 0, 'y': 0})
                    loc2 = city2.get('location', {'x': 0, 'y': 0})
                    distance = np.sqrt((loc1['x'] - loc2['x'])**2 + (loc1['y'] - loc2['y'])**2) * 500  # Scale to km
                    
                    travel_volume = (pop1 * pop2) / (distance ** 2 + 1)
                    travel_matrix[key] = {
                        'volume': travel_volume,
                        'distance_km': distance,
                        'cities': [city1['name'], city2['name']]
                    }
        
        # Analyze commute patterns
        commute_patterns = self._analyze_commute_patterns(cities, population_analysis)
        
        return {
            'inter_city_travel': travel_matrix,
            'commute_patterns': commute_patterns,
            'average_trip_length': self._calculate_average_trip_length(travel_matrix),
            'peak_demand_periods': ['morning_commute', 'evening_commute', 'weekend_travel']
        }
    
    def _analyze_commute_patterns(self, cities: List[Dict[str, Any]], population_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze daily commute patterns"""
        major_cities = population_analysis['major_cities']
        commute_data = {}
        
        for city in major_cities:
            # Estimate commute flows into major city
            inbound_commuters = city['population'] * 0.3  # 30% of population commutes in
            commute_data[city['name']] = {
                'inbound_commuters': inbound_commuters,
                'commute_radius_km': min(100, city['population'] / 10000),  # Rough heuristic
                'peak_hours': ['07:00-09:00', '17:00-19:00']
            }
        
        return commute_data
    
    def _calculate_average_trip_length(self, travel_matrix: Dict[str, Any]) -> float:
        """Calculate average trip length from travel matrix"""
        if not travel_matrix:
            return 50.0  # Default average
        
        total_volume = sum(entry['volume'] for entry in travel_matrix.values())
        weighted_distance = sum(entry['volume'] * entry['distance_km'] for entry in travel_matrix.values())
        
        return weighted_distance / total_volume if total_volume > 0 else 50.0
    
    def _identify_demand_corridors(self, population_analysis: Dict[str, Any], 
                                 travel_patterns: Dict[str, Any], 
                                 user_priorities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify high-demand transportation corridors"""
        travel_matrix = travel_patterns['inter_city_travel']
        major_cities = population_analysis['major_cities']
        
        corridors = []
        
        # Find highest volume city pairs
        sorted_routes = sorted(travel_matrix.items(), key=lambda x: x[1]['volume'], reverse=True)
        
        for route_key, route_data in sorted_routes[:10]:  # Top 10 routes
            volume = route_data['volume']
            distance = route_data['distance_km']
            cities = route_data['cities']
            
            # Calculate demand score
            demand_score = self._calculate_demand_score(volume, distance, cities, population_analysis)
            
            corridor = {
                'route': route_key,
                'cities': cities,
                'distance_km': distance,
                'volume': volume,
                'demand_score': demand_score,
                'priority': self._assign_priority(demand_score, user_priorities),
                'estimated_ridership': self._estimate_ridership(volume, distance),
                'suggested_service_type': self._suggest_service_type(distance, volume)
            }
            
            corridors.append(corridor)
        
        return corridors
    
    def _calculate_demand_score(self, volume: float, distance: float, cities: List[str], 
                              population_analysis: Dict[str, Any]) -> float:
        """Calculate normalized demand score (0-1) for a corridor"""
        # Normalize volume (0-1 scale)
        max_volume = max(entry['volume'] for entry in population_analysis['travel_patterns']['inter_city_travel'].values()) if population_analysis['travel_patterns']['inter_city_travel'] else 1
        normalized_volume = volume / max_volume if max_volume > 0 else 0
        
        # Distance factor (shorter distances generally have higher relative demand)
        distance_factor = 1.0 / (1.0 + distance / 100)  # Decay with distance
        
        # City importance factor
        city_importance = 1.0
        for city_name in cities:
            city = next((c for c in population_analysis['cities'] if c['name'] == city_name), None)
            if city and city.get('is_capital', False):
                city_importance *= 1.5
            elif city and city['population'] > population_analysis['total_population'] * 0.1:
                city_importance *= 1.2
        
        return min(1.0, normalized_volume * distance_factor * city_importance)
    
    def _assign_priority(self, demand_score: float, user_priorities: Dict[str, Any]) -> str:
        """Assign priority level based on demand score and user preferences"""
        priority_areas = user_priorities.get('priority_areas', [])
        
        if demand_score > 0.8:
            return 'critical'
        elif demand_score > 0.6:
            return 'high'
        elif demand_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_ridership(self, volume: float, distance: float) -> Dict[str, float]:
        """Estimate daily ridership for a corridor"""
        # Convert gravity model volume to actual ridership estimates
        base_ridership = volume * 0.01  # Conversion factor
        
        return {
            'daily_riders': base_ridership,
            'annual_riders': base_ridership * 365,
            'peak_hour_riders': base_ridership * 0.1  # 10% travel in peak hour
        }
    
    def _suggest_service_type(self, distance: float, volume: float) -> str:
        """Suggest appropriate service type based on distance and volume"""
        if distance > 300 and volume > 1000000:
            return 'high_speed'
        elif distance > 100 and volume > 500000:
            return 'regional_express'
        elif distance > 50:
            return 'regional'
        else:
            return 'commuter'
    
    def _prioritize_routes(self, demand_corridors: List[Dict[str, Any]], user_priorities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize routes based on demand and user preferences"""
        priority_areas = user_priorities.get('priority_areas', [])
        
        # Filter and sort based on priorities
        if priority_areas:
            # Boost priority for routes connecting priority areas
            for corridor in demand_corridors:
                for city in corridor['cities']:
                    if any(priority.lower() in city.lower() for priority in priority_areas):
                        corridor['demand_score'] *= 1.3  # Boost score
        
        # Sort by demand score
        prioritized = sorted(demand_corridors, key=lambda x: x['demand_score'], reverse=True)
        
        return prioritized[:5]  # Return top 5 priority routes