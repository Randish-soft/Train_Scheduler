"""
BCPC Pipeline - Station Placement Optimization Module
===================================================

This module optimizes railway station placement for both inter-city and intra-city scenarios,
considering population density, employment centers, accessibility, and travel demand patterns.

Features:
- Multi-modal station placement (terminus, intermediate, local)
- Population density-based optimization
- Employment center integration
- Accessibility analysis
- Demand modeling for intra-city stations
- Integration with OpenStreetMap and demographic data

Author: BCPC Pipeline Team
License: Open Source
"""

import json
import logging
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union, Set
from enum import Enum
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, MultiPoint
from shapely.ops import unary_union
from scipy.spatial import distance_matrix, Voronoi
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import osmnx as ox
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StationType(Enum):
    """Types of railway stations"""
    TERMINUS = "terminus"          # End stations (major hubs)
    INTERMEDIATE = "intermediate"   # Stations between cities
    LOCAL = "local"                # Intra-city stations
    JUNCTION = "junction"          # Connection points for multiple lines
    DEPOT = "depot"               # Maintenance and storage facilities

class AccessibilityMode(Enum):
    """Transportation modes for station access"""
    WALKING = "walking"
    CYCLING = "cycling"
    BUS = "bus"
    METRO = "metro"
    CAR = "car"
    FEEDER_BUS = "feeder_bus"

@dataclass
class PopulationCenter:
    """Represents a population concentration area"""
    location: Point
    population: int
    density_per_km2: float
    area_km2: float
    center_type: str  # "residential", "commercial", "mixed", "industrial"
    employment_count: Optional[int] = None
    poi_count: int = 0  # Points of interest (hospitals, schools, etc.)

@dataclass
class EmploymentCenter:
    """Represents an employment/business district"""
    location: Point
    job_count: int
    business_type: str  # "cbd", "industrial", "tech_park", "university", "hospital"
    area_km2: float
    daily_workers: int
    peak_hour_factor: float = 2.5  # Peak vs average demand

@dataclass
class StationCandidate:
    """Potential station location with attributes"""
    location: Point
    station_type: StationType
    population_catchment: int
    employment_catchment: int
    accessibility_score: float
    construction_feasibility: float  # 0-1 score based on terrain, existing infrastructure
    estimated_daily_passengers: int
    cost_factor: float = 1.0  # Relative construction cost multiplier
    existing_infrastructure: bool = False  # Near existing transport hubs
    
@dataclass
class StationPlacement:
    """Final optimized station placement"""
    station_id: str
    location: Point
    station_type: StationType
    name: str
    population_served: int
    employment_served: int
    daily_passengers_estimate: int
    catchment_area: Polygon
    access_modes: List[AccessibilityMode]
    platform_count: int = 2
    estimated_cost: float = 0.0

@dataclass
class CityStationNetwork:
    """Complete station network for a city"""
    city_name: str
    city_boundary: Polygon
    stations: List[StationPlacement]
    total_population_served: int
    coverage_percentage: float  # % of population within reasonable access
    network_efficiency_score: float

class StationPlacementOptimizer:
    """
    Main engine for optimizing station placement
    """
    
    def __init__(self, 
                 walking_distance_km: float = 0.8,
                 cycling_distance_km: float = 3.0,
                 feeder_bus_distance_km: float = 8.0,
                 min_station_spacing_km: float = 1.5,
                 population_threshold: int = 10000):
        """
        Initialize the station placement optimizer
        
        Args:
            walking_distance_km: Maximum comfortable walking distance to station
            cycling_distance_km: Maximum cycling distance to station
            feeder_bus_distance_km: Maximum feeder bus service distance
            min_station_spacing_km: Minimum distance between stations
            population_threshold: Minimum population to justify a local station
        """
        self.walking_distance = walking_distance_km
        self.cycling_distance = cycling_distance_km
        self.feeder_distance = feeder_bus_distance_km
        self.min_spacing = min_station_spacing_km
        self.pop_threshold = population_threshold
        
        # Demand modeling parameters
        self.trip_rate_residential = 2.1  # trips per person per day
        self.trip_rate_employment = 1.8   # trips per job per day
        self.modal_split_rail = 0.35      # % of trips that use rail
        
    def optimize_city_stations(self, 
                             city_boundary: Polygon,
                             population_centers: List[PopulationCenter],
                             employment_centers: List[EmploymentCenter],
                             route_line: LineString,
                             city_name: str = "Unknown",
                             max_stations: int = 8) -> CityStationNetwork:
        """
        Optimize station placement for a single city
        
        Args:
            city_boundary: City administrative boundary
            population_centers: List of population concentration areas
            employment_centers: List of employment/business districts
            route_line: Proposed railway route through the city
            city_name: Name of the city
            max_stations: Maximum number of stations allowed
            
        Returns:
            Optimized station network for the city
        """
        logger.info(f"Optimizing station placement for {city_name}")
        
        # Generate candidate station locations
        candidates = self._generate_station_candidates(
            city_boundary, population_centers, employment_centers, route_line
        )
        
        # Score candidates based on multiple criteria
        scored_candidates = self._score_candidates(
            candidates, population_centers, employment_centers
        )
        
        # Select optimal station set
        selected_stations = self._select_optimal_stations(
            scored_candidates, max_stations, route_line
        )
        
        # Create final station placements
        final_stations = self._create_station_placements(
            selected_stations, population_centers, employment_centers, city_name
        )
        
        # Calculate network metrics
        network_metrics = self._calculate_network_metrics(
            final_stations, population_centers, city_boundary
        )
        
        return CityStationNetwork(
            city_name=city_name,
            city_boundary=city_boundary,
            stations=final_stations,
            total_population_served=network_metrics['population_served'],
            coverage_percentage=network_metrics['coverage_percentage'],
            network_efficiency_score=network_metrics['efficiency_score']
        )
    
    def optimize_multi_city_corridor(self,
                                   cities_data: List[Dict],
                                   corridor_route: LineString,
                                   total_budget: Optional[float] = None) -> List[CityStationNetwork]:
        """
        Optimize station placement for multiple cities along a corridor
        
        Args:
            cities_data: List of city data dictionaries with boundaries, population, employment
            corridor_route: Complete railway route line
            total_budget: Optional budget constraint for all stations
            
        Returns:
            List of optimized station networks for each city
        """
        logger.info(f"Optimizing station placement for {len(cities_data)} cities along corridor")
        
        city_networks = []
        
        for city_data in cities_data:
            # Extract route segment within city
            city_segment = self._extract_city_route_segment(
                corridor_route, city_data['boundary']
            )
            
            if city_segment.length > 0:  # Route passes through city
                # Determine station allocation based on city size
                max_stations = self._calculate_station_allocation(
                    city_data, total_budget, len(cities_data)
                )
                
                # Optimize stations for this city
                city_network = self.optimize_city_stations(
                    city_boundary=city_data['boundary'],
                    population_centers=city_data['population_centers'],
                    employment_centers=city_data['employment_centers'],
                    route_line=city_segment,
                    city_name=city_data['name'],
                    max_stations=max_stations
                )
                
                city_networks.append(city_network)
        
        return city_networks
    
    def _generate_station_candidates(self,
                                   city_boundary: Polygon,
                                   population_centers: List[PopulationCenter],
                                   employment_centers: List[EmploymentCenter],
                                   route_line: LineString) -> List[StationCandidate]:
        """Generate potential station locations"""
        
        candidates = []
        
        # 1. Terminus stations at route endpoints within city
        route_coords = list(route_line.coords)
        
        for endpoint in [Point(route_coords[0]), Point(route_coords[-1])]:
            if city_boundary.contains(endpoint):
                candidates.append(StationCandidate(
                    location=endpoint,
                    station_type=StationType.TERMINUS,
                    population_catchment=0,  # Will be calculated later
                    employment_catchment=0,
                    accessibility_score=0.0,
                    construction_feasibility=0.8,  # Endpoints often easier
                    estimated_daily_passengers=0
                ))
        
        # 2. Population center-based candidates
        for pop_center in population_centers:
            if pop_center.population >= self.pop_threshold:
                # Find nearest point on route to population center
                nearest_point = self._nearest_point_on_line(route_line, pop_center.location)
                
                candidates.append(StationCandidate(
                    location=nearest_point,
                    station_type=StationType.LOCAL,
                    population_catchment=pop_center.population,
                    employment_catchment=0,
                    accessibility_score=0.0,
                    construction_feasibility=0.6,
                    estimated_daily_passengers=0
                ))
        
        # 3. Employment center-based candidates
        for emp_center in employment_centers:
            if emp_center.job_count >= 5000:  # Significant employment center
                nearest_point = self._nearest_point_on_line(route_line, emp_center.location)
                
                candidates.append(StationCandidate(
                    location=nearest_point,
                    station_type=StationType.LOCAL,
                    population_catchment=0,
                    employment_catchment=emp_center.job_count,
                    accessibility_score=0.0,
                    construction_feasibility=0.7,  # Usually better infrastructure
                    estimated_daily_passengers=0
                ))
        
        # 4. Strategic intermediate points along route
        # Place candidates at regular intervals for coverage
        route_length = route_line.length
        num_intervals = max(2, int(route_length / (self.min_spacing * 1000)))  # Convert to meters
        
        for i in range(1, num_intervals):
            distance_along = (i / num_intervals) * route_length
            point = route_line.interpolate(distance_along)
            
            if city_boundary.contains(point):
                candidates.append(StationCandidate(
                    location=point,
                    station_type=StationType.INTERMEDIATE,
                    population_catchment=0,
                    employment_catchment=0,
                    accessibility_score=0.0,
                    construction_feasibility=0.5,
                    estimated_daily_passengers=0
                ))
        
        # Remove candidates that are too close together
        candidates = self._filter_close_candidates(candidates)
        
        logger.info(f"Generated {len(candidates)} station candidates")
        return candidates
    
    def _score_candidates(self,
                         candidates: List[StationCandidate],
                         population_centers: List[PopulationCenter],
                         employment_centers: List[EmploymentCenter]) -> List[StationCandidate]:
        """Score station candidates based on multiple criteria"""
        
        for candidate in candidates:
            # Population accessibility score
            pop_score = self._calculate_population_accessibility(
                candidate, population_centers
            )
            
            # Employment accessibility score
            emp_score = self._calculate_employment_accessibility(
                candidate, employment_centers
            )
            
            # Infrastructure feasibility score
            infra_score = self._assess_infrastructure_feasibility(candidate)
            
            # Calculate total catchment and demand
            candidate.population_catchment = self._calculate_population_catchment(
                candidate, population_centers
            )
            candidate.employment_catchment = self._calculate_employment_catchment(
                candidate, employment_centers
            )
            
            # Estimate daily passengers
            candidate.estimated_daily_passengers = self._estimate_daily_demand(candidate)
            
            # Combined accessibility score (weighted)
            candidate.accessibility_score = (
                0.4 * pop_score +           # Population access weight
                0.3 * emp_score +           # Employment access weight
                0.2 * infra_score +         # Infrastructure weight
                0.1 * (1.0 if candidate.station_type == StationType.TERMINUS else 0.5)  # Type bonus
            )
            
        return candidates
    
    def _select_optimal_stations(self,
                               candidates: List[StationCandidate],
                               max_stations: int,
                               route_line: LineString) -> List[StationCandidate]:
        """Select optimal set of stations using optimization algorithm"""
        
        # Sort candidates by accessibility score
        candidates_sorted = sorted(candidates, key=lambda x: x.accessibility_score, reverse=True)
        
        # Always include terminus stations if available
        selected = [c for c in candidates_sorted if c.station_type == StationType.TERMINUS]
        remaining_slots = max_stations - len(selected)
        
        if remaining_slots <= 0:
            return selected[:max_stations]
        
        # Filter out terminus stations from further selection
        non_terminus = [c for c in candidates_sorted if c.station_type != StationType.TERMINUS]
        
        # Use greedy algorithm with spacing constraints
        for candidate in non_terminus:
            if len(selected) >= max_stations:
                break
                
            # Check minimum spacing constraint
            too_close = False
            for existing in selected:
                if candidate.location.distance(existing.location) < self.min_spacing * 1000:  # Convert to meters
                    too_close = True
                    break
            
            if not too_close:
                selected.append(candidate)
        
        logger.info(f"Selected {len(selected)} stations from {len(candidates)} candidates")
        return selected
    
    def _create_station_placements(self,
                                 selected_candidates: List[StationCandidate],
                                 population_centers: List[PopulationCenter],
                                 employment_centers: List[EmploymentCenter],
                                 city_name: str) -> List[StationPlacement]:
        """Create final station placements with detailed attributes"""
        
        stations = []
        
        for i, candidate in enumerate(selected_candidates):
            # Generate station name
            station_name = self._generate_station_name(
                candidate, city_name, i, population_centers, employment_centers
            )
            
            # Calculate catchment area
            catchment_area = self._calculate_catchment_area(candidate)
            
            # Determine access modes
            access_modes = self._determine_access_modes(candidate, population_centers, employment_centers)
            
            # Calculate platform requirements
            platform_count = self._calculate_platform_requirements(candidate)
            
            stations.append(StationPlacement(
                station_id=f"{city_name.lower().replace(' ', '_')}_station_{i+1}",
                location=candidate.location,
                station_type=candidate.station_type,
                name=station_name,
                population_served=candidate.population_catchment,
                employment_served=candidate.employment_catchment,
                daily_passengers_estimate=candidate.estimated_daily_passengers,
                catchment_area=catchment_area,
                access_modes=access_modes,
                platform_count=platform_count,
                estimated_cost=self._estimate_station_cost(candidate)
            ))
        
        return stations
    
    def _calculate_population_accessibility(self,
                                          candidate: StationCandidate,
                                          population_centers: List[PopulationCenter]) -> float:
        """Calculate how well a station serves population centers"""
        
        total_weighted_pop = 0
        total_pop = sum(pc.population for pc in population_centers)
        
        if total_pop == 0:
            return 0.0
        
        for pop_center in population_centers:
            distance_km = candidate.location.distance(pop_center.location) / 1000
            
            # Distance decay function
            if distance_km <= self.walking_distance:
                weight = 1.0
            elif distance_km <= self.cycling_distance:
                weight = 0.7
            elif distance_km <= self.feeder_distance:
                weight = 0.3
            else:
                weight = 0.0
            
            total_weighted_pop += pop_center.population * weight
        
        return total_weighted_pop / total_pop
    
    def _calculate_employment_accessibility(self,
                                          candidate: StationCandidate,
                                          employment_centers: List[EmploymentCenter]) -> float:
        """Calculate how well a station serves employment centers"""
        
        total_weighted_jobs = 0
        total_jobs = sum(ec.job_count for ec in employment_centers)
        
        if total_jobs == 0:
            return 0.0
        
        for emp_center in employment_centers:
            distance_km = candidate.location.distance(emp_center.location) / 1000
            
            # Higher weight for employment centers (work trips are more predictable)
            if distance_km <= self.walking_distance:
                weight = 1.0
            elif distance_km <= self.cycling_distance:
                weight = 0.8
            elif distance_km <= self.feeder_distance:
                weight = 0.5
            else:
                weight = 0.0
            
            total_weighted_jobs += emp_center.job_count * weight
        
        return total_weighted_jobs / total_jobs
    
    def _assess_infrastructure_feasibility(self, candidate: StationCandidate) -> float:
        """Assess construction feasibility based on location characteristics"""
        
        # This would integrate with terrain analysis and existing infrastructure data
        # For now, use simplified rules based on station type and location
        
        base_score = candidate.construction_feasibility
        
        # Bonus for existing infrastructure (would be determined from OSM data)
        if candidate.existing_infrastructure:
            base_score += 0.2
        
        # Penalty for difficult terrain (would come from terrain module)
        # This is a simplified placeholder
        base_score *= candidate.cost_factor
        
        return min(1.0, base_score)
    
    def _calculate_population_catchment(self,
                                      candidate: StationCandidate,
                                      population_centers: List[PopulationCenter]) -> int:
        """Calculate total population served by a station"""
        
        total_population = 0
        
        for pop_center in population_centers:
            distance_km = candidate.location.distance(pop_center.location) / 1000
            
            if distance_km <= self.feeder_distance:
                # Apply distance decay to population served
                if distance_km <= self.walking_distance:
                    served_fraction = 1.0
                elif distance_km <= self.cycling_distance:
                    served_fraction = 0.7
                else:  # Within feeder bus range
                    served_fraction = 0.3
                
                total_population += int(pop_center.population * served_fraction)
        
        return total_population
    
    def _calculate_employment_catchment(self,
                                      candidate: StationCandidate,
                                      employment_centers: List[EmploymentCenter]) -> int:
        """Calculate total employment served by a station"""
        
        total_employment = 0
        
        for emp_center in employment_centers:
            distance_km = candidate.location.distance(emp_center.location) / 1000
            
            if distance_km <= self.feeder_distance:
                if distance_km <= self.walking_distance:
                    served_fraction = 1.0
                elif distance_km <= self.cycling_distance:
                    served_fraction = 0.8
                else:
                    served_fraction = 0.5
                
                total_employment += int(emp_center.job_count * served_fraction)
        
        return total_employment
    
    def _estimate_daily_demand(self, candidate: StationCandidate) -> int:
        """Estimate daily passenger demand for a station"""
        
        # Base demand from residential trips
        residential_demand = candidate.population_catchment * self.trip_rate_residential * self.modal_split_rail
        
        # Employment-based demand (commuter trips)
        employment_demand = candidate.employment_catchment * self.trip_rate_employment * self.modal_split_rail
        
        # Terminus stations get bonus for long-distance trips
        if candidate.station_type == StationType.TERMINUS:
            terminus_bonus = residential_demand * 0.3
        else:
            terminus_bonus = 0
        
        total_demand = residential_demand + employment_demand + terminus_bonus
        
        return int(total_demand)
    
    def _filter_close_candidates(self, candidates: List[StationCandidate]) -> List[StationCandidate]:
        """Remove candidates that are too close to each other"""
        
        filtered = []
        
        for candidate in candidates:
            too_close = False
            for existing in filtered:
                if candidate.location.distance(existing.location) < (self.min_spacing * 1000 * 0.8):
                    # Keep the better candidate
                    if candidate.accessibility_score > existing.accessibility_score:
                        filtered.remove(existing)
                    else:
                        too_close = True
                    break
            
            if not too_close:
                filtered.append(candidate)
        
        return filtered
    
    def _nearest_point_on_line(self, line: LineString, point: Point) -> Point:
        """Find nearest point on line to given point"""
        return line.interpolate(line.project(point))
    
    def _extract_city_route_segment(self, corridor_route: LineString, city_boundary: Polygon) -> LineString:
        """Extract the portion of route that passes through a city"""
        
        intersection = corridor_route.intersection(city_boundary)
        
        if isinstance(intersection, LineString):
            return intersection
        elif hasattr(intersection, 'geoms'):
            # Multiple segments, return the longest
            segments = [geom for geom in intersection.geoms if isinstance(geom, LineString)]
            if segments:
                return max(segments, key=lambda x: x.length)
        
        return LineString()  # Empty line if no intersection
    
    def _calculate_station_allocation(self, city_data: Dict, total_budget: Optional[float], num_cities: int) -> int:
        """Calculate how many stations to allocate to a city"""
        
        # Base allocation on population and employment
        population = sum(pc.population for pc in city_data['population_centers'])
        employment = sum(ec.job_count for ec in city_data['employment_centers'])
        
        # Size-based allocation
        if population > 1_000_000:
            base_stations = 6
        elif population > 500_000:
            base_stations = 4
        elif population > 100_000:
            base_stations = 3
        elif population > 50_000:
            base_stations = 2
        else:
            base_stations = 1
        
        # Adjust for employment density
        if employment > population * 0.6:  # High employment center
            base_stations += 1
        
        return min(8, base_stations)  # Cap at 8 stations per city
    
    def _generate_station_name(self,
                             candidate: StationCandidate,
                             city_name: str,
                             station_index: int,
                             population_centers: List[PopulationCenter],
                             employment_centers: List[EmploymentCenter]) -> str:
        """Generate appropriate station name"""
        
        if candidate.station_type == StationType.TERMINUS:
            # Check if it's at city center or edge
            city_center_distance = min(
                candidate.location.distance(pc.location) for pc in population_centers
            ) / 1000
            
            if city_center_distance < 2.0:
                return f"{city_name} Central"
            else:
                return f"{city_name} Terminal"
        
        # Find closest population or employment center
        closest_pop = min(
            population_centers,
            key=lambda pc: candidate.location.distance(pc.location),
            default=None
        )
        
        closest_emp = min(
            employment_centers,
            key=lambda ec: candidate.location.distance(ec.location),
            default=None
        )
        
        # Use closest significant feature for naming
        if closest_pop and candidate.location.distance(closest_pop.location) < 2000:  # Within 2km
            if closest_pop.center_type == "commercial":
                return f"{city_name} Commercial"
            elif "university" in closest_pop.center_type.lower():
                return f"{city_name} University"
            else:
                return f"{city_name} {closest_pop.center_type.title()}"
        
        if closest_emp and candidate.location.distance(closest_emp.location) < 2000:
            if closest_emp.business_type == "cbd":
                return f"{city_name} CBD"
            elif closest_emp.business_type == "industrial":
                return f"{city_name} Industrial"
            else:
                return f"{city_name} {closest_emp.business_type.title()}"
        
        # Default naming
        return f"{city_name} Station {station_index + 1}"
    
    def _calculate_catchment_area(self, candidate: StationCandidate) -> Polygon:
        """Calculate the catchment area polygon for a station"""
        
        # Use maximum access distance to create catchment
        max_distance = self.feeder_distance * 1000  # Convert to meters
        
        # Create circular catchment (could be more sophisticated with road network)
        catchment = candidate.location.buffer(max_distance)
        
        return catchment
    
    def _determine_access_modes(self,
                               candidate: StationCandidate,
                               population_centers: List[PopulationCenter],
                               employment_centers: List[EmploymentCenter]) -> List[AccessibilityMode]:
        """Determine appropriate access modes for a station"""
        
        access_modes = [AccessibilityMode.WALKING]  # All stations have walking access
        
        # Check distances to major population/employment centers
        max_distance = 0
        for pc in population_centers:
            distance = candidate.location.distance(pc.location) / 1000
            max_distance = max(max_distance, distance)
        
        for ec in employment_centers:
            distance = candidate.location.distance(ec.location) / 1000
            max_distance = max(max_distance, distance)
        
        # Add access modes based on maximum catchment distance
        if max_distance > self.walking_distance:
            access_modes.append(AccessibilityMode.CYCLING)
        
        if max_distance > self.cycling_distance:
            access_modes.append(AccessibilityMode.FEEDER_BUS)
        
        # Terminus stations typically have more access modes
        if candidate.station_type == StationType.TERMINUS:
            access_modes.extend([AccessibilityMode.CAR, AccessibilityMode.BUS])
        
        # High-demand stations get metro connections
        if candidate.estimated_daily_passengers > 20000:
            access_modes.append(AccessibilityMode.METRO)
        
        return list(set(access_modes))  # Remove duplicates
    
    def _calculate_platform_requirements(self, candidate: StationCandidate) -> int:
        """Calculate number of platforms needed"""
        
        if candidate.station_type == StationType.TERMINUS:
            return 4  # Terminus needs more platforms
        elif candidate.estimated_daily_passengers > 15000:
            return 3  # High-volume stations
        else:
            return 2  # Standard stations
    
    def _estimate_station_cost(self, candidate: StationCandidate) -> float:
        """Estimate construction cost for a station"""
        
        # Base costs from cost_analysis module
        if candidate.station_type == StationType.TERMINUS:
            base_cost = 10_000_000  # EUR
        else:
            base_cost = 2_000_000   # EUR
        
        # Adjust for demand and complexity
        if candidate.estimated_daily_passengers > 20000:
            base_cost *= 1.5
        elif candidate.estimated_daily_passengers > 10000:
            base_cost *= 1.2
        
        # Apply cost factor for construction difficulty
        return base_cost * candidate.cost_factor
    
    def _calculate_network_metrics(self,
                                 stations: List[StationPlacement],
                                 population_centers: List[PopulationCenter],
                                 city_boundary: Polygon) -> Dict:
        """Calculate network performance metrics"""
        
        total_city_population = sum(pc.population for pc in population_centers)
        population_served = sum(station.population_served for station in stations)
        
        # Avoid double-counting population served by multiple stations
        unique_served = min(population_served, total_city_population)
        
        coverage_percentage = (unique_served / total_city_population * 100) if total_city_population > 0 else 0
        
        # Simple efficiency score based on coverage and number of stations
        efficiency_score = coverage_percentage / len(stations) if stations else 0
        
        return {
            'population_served': unique_served,
            'coverage_percentage': coverage_percentage,
            'efficiency_score': efficiency_score
        }

# Data loading and integration functions

def load_population_data_from_osm(city_boundary: Polygon, city_name: str) -> List[PopulationCenter]:
    """Load population data from OpenStreetMap and census sources"""
    
    logger.info(f"Loading population data for {city_name}")
    
    try:
        # Get city graph for analysis
        graph = ox.graph_from_polygon(city_boundary, network_type='all')
        
        # Get points of interest that indicate population centers
        pois = ox.features_from_polygon(city_boundary, tags={
            'amenity': ['school', 'hospital', 'shopping_mall', 'university'],
            'landuse': ['residential', 'commercial', 'retail'],
            'place': ['suburb', 'neighbourhood', 'quarter']
        })
        
        population_centers = []
        
        # Extract residential areas and estimate population
        if not pois.empty:
            for idx, poi in pois.iterrows():
                if poi.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                    centroid = poi.geometry.centroid
                    area_km2 = poi.geometry.area / 1_000_000  # Convert to km²
                    
                    # Estimate population based on land use type
                    if poi.get('landuse') == 'residential':
                        density = 3000  # people per km² (adjustable)
                        population = int(area_km2 * density)
                        center_type = 'residential'
                    elif poi.get('landuse') in ['commercial', 'retail']:
                        density = 1000  # Lower residential density in commercial areas
                        population = int(area_km2 * density)
                        center_type = 'commercial'
                    elif poi.get('amenity') == 'university':
                        population = 15000  # Typical university population
                        center_type = 'university'
                    elif poi.get('place') in ['suburb', 'neighbourhood']:
                        density = 2500
                        population = int(area_km2 * density)
                        center_type = 'residential'
                    else:
                        continue
                    
                    if population > 1000:  # Minimum threshold
                        population_centers.append(PopulationCenter(
                            location=centroid,
                            population=population,
                            density_per_km2=density,
                            area_km2=area_km2,
                            center_type=center_type
                        ))
        
        # If no detailed data available, create grid-based estimates
        if len(population_centers) == 0:
            population_centers = _create_grid_population_estimates(city_boundary, city_name)
            
        logger.info(f"Found {len(population_centers)} population centers")
        return population_centers
        
    except Exception as e:
        logger.warning(f"Error loading OSM data for {city_name}: {e}")
        return _create_grid_population_estimates(city_boundary, city_name)

def load_employment_data_from_osm(city_boundary: Polygon, city_name: str) -> List[EmploymentCenter]:
    """Load employment center data from OpenStreetMap"""
    
    logger.info(f"Loading employment data for {city_name}")
    
    try:
        # Get employment-related POIs
        employment_pois = ox.features_from_polygon(city_boundary, tags={
            'landuse': ['commercial', 'industrial', 'retail'],
            'amenity': ['hospital', 'university', 'government'],
            'office': True,
            'shop': True
        })
        
        employment_centers = []
        
        if not employment_pois.empty:
            for idx, poi in employment_pois.iterrows():
                if poi.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                    centroid = poi.geometry.centroid
                    area_km2 = poi.geometry.area / 1_000_000
                    
                    # Estimate employment based on land use
                    if poi.get('landuse') == 'commercial':
                        jobs_per_km2 = 2000
                        business_type = 'cbd'
                    elif poi.get('landuse') == 'industrial':
                        jobs_per_km2 = 800
                        business_type = 'industrial'
                    elif poi.get('amenity') == 'hospital':
                        jobs_per_km2 = 3000
                        business_type = 'hospital'
                    elif poi.get('amenity') == 'university':
                        jobs_per_km2 = 1500
                        business_type = 'university'
                    elif poi.get('office'):
                        jobs_per_km2 = 2500
                        business_type = 'cbd'
                    else:
                        jobs_per_km2 = 1000
                        business_type = 'mixed'
                    
                    job_count = int(area_km2 * jobs_per_km2)
                    
                    if job_count > 500:  # Minimum threshold
                        employment_centers.append(EmploymentCenter(
                            location=centroid,
                            job_count=job_count,
                            business_type=business_type,
                            area_km2=area_km2,
                            daily_workers=int(job_count * 1.2)  # Include visitors/customers
                        ))
        
        logger.info(f"Found {len(employment_centers)} employment centers")
        return employment_centers
        
    except Exception as e:
        logger.warning(f"Error loading employment data for {city_name}: {e}")
        return []

def _create_grid_population_estimates(city_boundary: Polygon, city_name: str) -> List[PopulationCenter]:
    """Create grid-based population estimates when detailed data unavailable"""
    
    # Get bounding box
    minx, miny, maxx, maxy = city_boundary.bounds
    
    # Create grid cells (1km x 1km)
    grid_size = 0.01  # Approximately 1km at mid-latitudes
    
    population_centers = []
    x = minx
    cell_id = 1
    
    while x < maxx:
        y = miny
        while y < maxy:
            # Create grid cell
            cell = Polygon([
                (x, y), (x + grid_size, y),
                (x + grid_size, y + grid_size), (x, y + grid_size)
            ])
            
            # Check if cell intersects city boundary
            if city_boundary.intersects(cell):
                intersection = city_boundary.intersection(cell)
                if intersection.area > 0:
                    centroid = intersection.centroid
                    area_km2 = intersection.area * 111_000 * 111_000 / 1_000_000  # Rough conversion
                    
                    # Estimate population (varies by distance from center)
                    city_centroid = city_boundary.centroid
                    distance_from_center = centroid.distance(city_centroid)
                    
                    # Higher density near center
                    if distance_from_center < 0.02:  # ~2km from center
                        density = 4000
                    elif distance_from_center < 0.05:  # ~5km from center
                        density = 2500
                    else:
                        density = 1000
                    
                    population = int(area_km2 * density)
                    
                    if population > 500:
                        population_centers.append(PopulationCenter(
                            location=centroid,
                            population=population,
                            density_per_km2=density,
                            area_km2=area_km2,
                            center_type='residential'
                        ))
            
            y += grid_size
        x += grid_size
    
    return population_centers

def create_example_city_data() -> Dict:
    """Create example city data for testing"""
    
    # Example: Brussels-like city
    city_boundary = Point(4.3517, 50.8466).buffer(0.1)  # ~10km radius around Brussels center
    
    population_centers = [
        PopulationCenter(
            location=Point(4.3517, 50.8466),  # City center
            population=180000,
            density_per_km2=5000,
            area_km2=36,
            center_type='mixed'
        ),
        PopulationCenter(
            location=Point(4.3200, 50.8300),  # West residential
            population=120000,
            density_per_km2=3500,
            area_km2=34,
            center_type='residential'
        ),
        PopulationCenter(
            location=Point(4.3800, 50.8600),  # North residential
            population=95000,
            density_per_km2=3200,
            area_km2=30,
            center_type='residential'
        ),
        PopulationCenter(
            location=Point(4.3400, 50.8200),  # University area
            population=45000,
            density_per_km2=2800,
            area_km2=16,
            center_type='university'
        )
    ]
    
    employment_centers = [
        EmploymentCenter(
            location=Point(4.3517, 50.8466),  # CBD
            job_count=150000,
            business_type='cbd',
            area_km2=5,
            daily_workers=180000
        ),
        EmploymentCenter(
            location=Point(4.3200, 50.8100),  # Industrial zone
            job_count=35000,
            business_type='industrial',
            area_km2=8,
            daily_workers=40000
        ),
        EmploymentCenter(
            location=Point(4.3700, 50.8700),  # Tech park
            job_count=25000,
            business_type='tech_park',
            area_km2=3,
            daily_workers=28000
        ),
        EmploymentCenter(
            location=Point(4.3400, 50.8200),  # University
            job_count=8000,
            business_type='university',
            area_km2=2,
            daily_workers=12000
        )
    ]
    
    return {
        'name': 'Brussels',
        'boundary': city_boundary,
        'population_centers': population_centers,
        'employment_centers': employment_centers
    }

# Export functions

def export_station_network_geojson(city_network: CityStationNetwork, output_path: str) -> None:
    """Export station network to GeoJSON for visualization"""
    
    features = []
    
    for station in city_network.stations:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [station.location.x, station.location.y]
            },
            "properties": {
                "station_id": station.station_id,
                "name": station.name,
                "type": station.station_type.value,
                "population_served": station.population_served,
                "employment_served": station.employment_served,
                "daily_passengers": station.daily_passengers_estimate,
                "platform_count": station.platform_count,
                "estimated_cost": station.estimated_cost,
                "access_modes": [mode.value for mode in station.access_modes]
            }
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "city_name": city_network.city_name,
            "total_population_served": city_network.total_population_served,
            "coverage_percentage": city_network.coverage_percentage,
            "efficiency_score": city_network.network_efficiency_score,
            "station_count": len(city_network.stations)
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Station network exported to {output_path}")

def generate_station_report(city_network: CityStationNetwork) -> str:
    """Generate detailed station placement report"""
    
    report = []
    report.append("=" * 70)
    report.append(f"STATION PLACEMENT REPORT - {city_network.city_name.upper()}")
    report.append("=" * 70)
    report.append("")
    
    # Network summary
    report.append("NETWORK SUMMARY:")
    report.append(f"  Total Stations: {len(city_network.stations)}")
    report.append(f"  Population Served: {city_network.total_population_served:,}")
    report.append(f"  Coverage: {city_network.coverage_percentage:.1f}%")
    report.append(f"  Network Efficiency Score: {city_network.network_efficiency_score:.1f}")
    report.append("")
    
    # Station details
    report.append("STATION DETAILS:")
    report.append("-" * 70)
    
    total_cost = 0
    total_daily_passengers = 0
    
    for i, station in enumerate(city_network.stations, 1):
        report.append(f"{i}. {station.name}")
        report.append(f"   Type: {station.station_type.value.title()}")
        report.append(f"   Location: {station.location.x:.6f}, {station.location.y:.6f}")
        report.append(f"   Population Served: {station.population_served:,}")
        report.append(f"   Employment Served: {station.employment_served:,}")
        report.append(f"   Daily Passengers: {station.daily_passengers_estimate:,}")
        report.append(f"   Platforms: {station.platform_count}")
        report.append(f"   Estimated Cost: €{station.estimated_cost:,.0f}")
        report.append(f"   Access Modes: {', '.join(mode.value for mode in station.access_modes)}")
        report.append("")
        
        total_cost += station.estimated_cost
        total_daily_passengers += station.daily_passengers_estimate
    
    # Cost summary
    report.append("COST SUMMARY:")
    report.append(f"  Total Station Construction Cost: €{total_cost:,.0f}")
    report.append(f"  Average Cost per Station: €{total_cost/len(city_network.stations):,.0f}")
    report.append(f"  Cost per Daily Passenger: €{total_cost/total_daily_passengers:.0f}")
    report.append("")
    
    # Demand summary
    report.append("DEMAND SUMMARY:")
    report.append(f"  Total Daily Passengers: {total_daily_passengers:,}")
    report.append(f"  Average per Station: {total_daily_passengers/len(city_network.stations):,.0f}")
    report.append("")
    
    # Station type breakdown
    type_counts = {}
    for station in city_network.stations:
        station_type = station.station_type.value
        type_counts[station_type] = type_counts.get(station_type, 0) + 1
    
    report.append("STATION TYPE BREAKDOWN:")
    for station_type, count in type_counts.items():
        report.append(f"  {station_type.title()}: {count}")
    report.append("")
    
    report.append("=" * 70)
    report.append("Report generated by BCPC Station Placement Module")
    report.append("=" * 70)
    
    return "\n".join(report)

# Example usage and testing
if __name__ == "__main__":
    # Create optimizer
    optimizer = StationPlacementOptimizer(
        walking_distance_km=0.8,
        cycling_distance_km=3.0,
        feeder_bus_distance_km=8.0,
        min_station_spacing_km=1.5
    )
    
    # Create example city data
    city_data = create_example_city_data()
    
    # Create example route through the city
    route_line = LineString([
        (4.3200, 50.8100),  # Southwest entry
        (4.3400, 50.8300),  # Through residential
        (4.3517, 50.8466),  # City center
        (4.3600, 50.8550),  # North residential
        (4.3800, 50.8700)   # Northeast exit
    ])
    
    # Optimize station placement
    logger.info("Starting station placement optimization...")
    
    city_network = optimizer.optimize_city_stations(
        city_boundary=city_data['boundary'],
        population_centers=city_data['population_centers'],
        employment_centers=city_data['employment_centers'],
        route_line=route_line,
        city_name=city_data['name'],
        max_stations=6
    )
    
    # Generate and print report
    report = generate_station_report(city_network)
    print(report)
    
    # Export results
    export_station_network_geojson(city_network, "brussels_stations.geojson")
    
    logger.info("Station placement optimization completed!")
    
    # Example of multi-city corridor optimization
    logger.info("Testing multi-city corridor optimization...")
    
    # Create example corridor with multiple cities
    cities_data = [city_data]  # Add more cities as needed
    corridor_route = LineString([
        (4.2000, 50.7500),  # Start before Brussels
        (4.3200, 50.8100),
        (4.3517, 50.8466),  # Through Brussels
        (4.3800, 50.8700),
        (4.5000, 50.9000)   # Continue after Brussels
    ])
    
    corridor_networks = optimizer.optimize_multi_city_corridor(
        cities_data=cities_data,
        corridor_route=corridor_route,
        total_budget=100_000_000  # €100M budget
    )
    
    for network in corridor_networks:
        print(f"\nSummary for {network.city_name}:")
        print(f"  Stations: {len(network.stations)}")
        print(f"  Population served: {network.total_population_served:,}")
        print(f"  Coverage: {network.coverage_percentage:.1f}%")
    
    logger.info("Multi-city optimization completed!")