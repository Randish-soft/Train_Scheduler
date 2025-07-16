# File: railway_ai/generate.py
"""
Route Generation Module
Generates optimized railway route plans using learned intelligence patterns.
"""
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import time
import math
from datetime import datetime

from railway_ai.config import RailwayConfig
from railway_ai.intelligence.station_patterns import StationPatternAnalyzer, StationPlacement
from railway_ai.intelligence.track_intelligence import TrackIntelligence, RouteOption, TrackSegment
from railway_ai.intelligence.train_classifier import TrainClassifier, RouteAnalysis, TrainSpecification
from railway_ai.intelligence.railyard_optimizer import RailwayardOptimizer, RailyardCandidate
from railway_ai.extractors.terrain_analysis import TerrainAnalyzer
from railway_ai.utils.geo import GeoUtils, haversine_distance, BoundingBox
from railway_ai.utils.ml import MLPipeline, RailwayMLUtils

@dataclass
class GeneratedStation:
    """Generated station with all specifications"""
    name: str
    lat: float
    lon: float
    station_type: str
    platform_count: int
    estimated_daily_passengers: int
    construction_cost: float
    accessibility_score: float
    transfer_connections: List[str]
    services: List[str]
    reasons: List[str]

@dataclass
class GeneratedRoute:
    """Complete generated route specification"""
    name: str
    stations: List[GeneratedStation]
    track_segments: List[TrackSegment]
    total_length_km: float
    total_cost: float
    construction_time_months: int
    train_specifications: List[TrainSpecification]
    railyards: List[RailyardCandidate]
    environmental_impact_score: float
    feasibility_score: float
    ridership_potential: int
    operational_metrics: Dict[str, Any]
    generation_metadata: Dict[str, Any]

class RouteGenerator:
    """Main route generation engine"""
    
    def __init__(self, config: RailwayConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Example: Generate Brussels to Amsterdam route
    cities = [
        {'name': 'Brussels', 'lat': 50.8503, 'lon': 4.3517, 'population': 1200000},
        {'name': 'Antwerp', 'lat': 51.2194, 'lon': 4.4025, 'population': 530000},
        {'name': 'Rotterdam', 'lat': 51.9225, 'lon': 4.4792, 'population': 650000},
        {'name': 'Amsterdam', 'lat': 52.3676, 'lon': 4.9041, 'population': 870000}
    ]
    
    print("🚄 Generating Brussels-Amsterdam route...")
    
    try:
        route = generator.create_plan(
            input_data=cities,
            country="BE",
            optimization_targets=["cost", "ridership"],
            route_name="Brussels_Amsterdam_Express"
        )
        
        print(f"✅ Generated route: {route.name}")
        print(f"📏 Length: {route.total_length_km:.1f} km")
        print(f"🚉 Stations: {len(route.stations)}")
        print(f"💰 Cost: €{route.total_cost/1_000_000:.1f}M")
        print(f"🚄 Train types: {len(route.train_specifications)}")
        print(f"⏱️ Construction time: {route.construction_time_months} months")
        
        # Save the plan
        generator.save_plan(route, "example_route_plan.json")
        print("📁 Plan saved to example_route_plan.json")
        
        # Show station details
        print("\n🚉 Stations:")
        for station in route.stations:
            print(f"  • {station.name} ({station.station_type})")
            print(f"    {station.platform_count} platforms, {station.estimated_daily_passengers:,} daily passengers")
            print(f"    Cost: €{station.construction_cost/1_000_000:.1f}M")
        
        # Show train specifications
        print("\n🚄 Train Services:")
        for train_spec in route.train_specifications:
            print(f"  • {train_spec.category.value}: {train_spec.max_speed_kmh}km/h max, {train_spec.capacity_passengers} passengers")
            print(f"    Frequency: {train_spec.frequency_peak_min}min peak, {train_spec.frequency_offpeak_min}min off-peak")
        
        # Show railyards
        if route.railyards:
            print("\n🏭 Railyards:")
            for railyard in route.railyards:
                print(f"  • {railyard.maintenance_type.title()} at ({railyard.lat:.3f}, {railyard.lon:.3f})")
                print(f"    Capacity: {railyard.capacity_estimate} trains, Score: {railyard.score:.2f}")
        
        print(f"\n📊 Performance Metrics:")
        print(f"  • Environmental impact: {route.environmental_impact_score:.2f}/1.0")
        print(f"  • Feasibility score: {route.feasibility_score:.2f}/1.0") 
        print(f"  • Annual ridership: {route.ridership_potential:,} passengers")
        print(f"  • Cost per km: €{route.operational_metrics.get('cost_per_km', 0)/1_000_000:.1f}M")
        
    except Exception as e:
        print(f"❌ Route generation failed: {e}")
        import traceback
        traceback.print_exc() 
        self.station_analyzer = StationPatternAnalyzer()
        self.track_intelligence = TrackIntelligence()
        self.train_classifier = TrainClassifier()
        self.railyard_optimizer = RailwayardOptimizer()
        self.terrain_analyzer = TerrainAnalyzer()
        self.ml_pipeline = MLPipeline()
        
        # Load trained models if available
        self._load_trained_models()
        
        self.generation_stats = {
            'routes_generated': 0,
            'total_generation_time': 0,
            'average_generation_time': 0
        }
    
    def _load_trained_models(self):
        """Load previously trained ML models"""
        models_dir = self.config.paths.models_dir
        if models_dir.exists():
            try:
                # Load ML pipeline models
                self.ml_pipeline.load_models(str(models_dir))
                
                # Load intelligence module models (simplified - would be more complex in practice)
                self.logger.info(f"📚 Loaded trained models from {models_dir}")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load some models: {e}")
                self.logger.info("🔄 Will use heuristic methods as fallback")
    
    def create_plan(self,
                   input_data: Union[str, List[str], List[Dict]],
                   country: Optional[str] = None,
                   optimization_targets: List[str] = None,
                   constraints: Optional[Dict] = None,
                   route_name: Optional[str] = None) -> GeneratedRoute:
        """Generate complete optimized route plan"""
        
        start_time = time.time()
        self.logger.info("🚀 Starting route generation...")
        
        # Parse input data
        cities = self._parse_input_data(input_data)
        self.logger.info(f"📍 Processing {len(cities)} cities")
        
        # Set defaults
        if optimization_targets is None:
            optimization_targets = ["cost", "ridership"]
        if country is None:
            country = self.config.country_default
        if route_name is None:
            route_name = f"Generated_Route_{int(time.time())}"
        
        # Generate route plan
        try:
            route = self._generate_complete_route(
                cities=cities,
                country=country,
                optimization_targets=optimization_targets,
                constraints=constraints or {},
                route_name=route_name
            )
            
            # Record generation statistics
            generation_time = time.time() - start_time
            self.generation_stats['routes_generated'] += 1
            self.generation_stats['total_generation_time'] += generation_time
            self.generation_stats['average_generation_time'] = (
                self.generation_stats['total_generation_time'] / 
                self.generation_stats['routes_generated']
            )
            
            self.logger.info(f"✅ Route generation completed in {generation_time:.1f}s")
            return route
            
        except Exception as e:
            self.logger.error(f"❌ Route generation failed: {e}")
            raise
    
    def _parse_input_data(self, input_data: Union[str, List[str], List[Dict]]) -> List[Dict]:
        """Parse various input formats into standardized city data"""
        
        if isinstance(input_data, str):
            # Check if it's a file path or comma-separated cities
            input_path = Path(input_data)
            if input_path.exists():
                return self._load_cities_from_file(input_path)
            else:
                # Treat as comma-separated city names
                city_names = [city.strip() for city in input_data.split(",")]
                return self._geocode_cities(city_names)
        
        elif isinstance(input_data, list):
            if all(isinstance(item, str) for item in input_data):
                # List of city names
                return self._geocode_cities(input_data)
            elif all(isinstance(item, dict) for item in input_data):
                # Already formatted city data
                return input_data
        
        raise ValueError("Invalid input data format")
    
    def _load_cities_from_file(self, file_path: Path) -> List[Dict]:
        """Load cities from CSV file"""
        cities = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to detect if it's CSV
            sample = f.read(1024)
            f.seek(0)
            
            if ',' in sample or ';' in sample:
                # CSV format
                reader = csv.DictReader(f)
                for row in reader:
                    city = {
                        'name': row.get('name', row.get('city', '')),
                        'lat': float(row.get('lat', row.get('latitude', 0))),
                        'lon': float(row.get('lon', row.get('longitude', 0))),
                        'population': int(row.get('population', 50000)),
                        'country': row.get('country', ''),
                    }
                    cities.append(city)
            else:
                # Plain text, one city per line
                for line in f:
                    city_name = line.strip()
                    if city_name:
                        cities.extend(self._geocode_cities([city_name]))
        
        return cities
    
    def _geocode_cities(self, city_names: List[str]) -> List[Dict]:
        """Convert city names to coordinates (simplified geocoding)"""
        # Simplified geocoding using known European cities
        known_cities = {
            'brussels': {'lat': 50.8503, 'lon': 4.3517, 'population': 1200000},
            'amsterdam': {'lat': 52.3676, 'lon': 4.9041, 'population': 870000},
            'paris': {'lat': 48.8566, 'lon': 2.3522, 'population': 2100000},
            'berlin': {'lat': 52.5200, 'lon': 13.4050, 'population': 3700000},
            'cologne': {'lat': 50.9375, 'lon': 6.9603, 'population': 1100000},
            'frankfurt': {'lat': 50.1109, 'lon': 8.6821, 'population': 750000},
            'munich': {'lat': 48.1351, 'lon': 11.5820, 'population': 1500000},
            'vienna': {'lat': 48.2082, 'lon': 16.3738, 'population': 1900000},
            'zurich': {'lat': 47.3769, 'lon': 8.5417, 'population': 430000},
            'geneva': {'lat': 46.2044, 'lon': 6.1432, 'population': 200000},
            'antwerp': {'lat': 51.2194, 'lon': 4.4025, 'population': 530000},
            'ghent': {'lat': 51.0500, 'lon': 3.7303, 'population': 260000},
            'rotterdam': {'lat': 51.9225, 'lon': 4.4792, 'population': 650000},
            'the hague': {'lat': 52.0705, 'lon': 4.3007, 'population': 540000},
            'luxembourg': {'lat': 49.6116, 'lon': 6.1319, 'population': 125000},
        }
        
        cities = []
        for city_name in city_names:
            city_key = city_name.lower().strip()
            if city_key in known_cities:
                city_data = known_cities[city_key].copy()
                city_data['name'] = city_name.title()
                cities.append(city_data)
            else:
                # Fallback: place near Brussels with estimated data
                self.logger.warning(f"⚠️ Unknown city '{city_name}', using estimated location")
                cities.append({
                    'name': city_name.title(),
                    'lat': 50.8 + (len(cities) * 0.1),  # Spread cities out
                    'lon': 4.3 + (len(cities) * 0.1),
                    'population': 100000  # Default population
                })
        
        return cities
    
    def _generate_complete_route(self,
                               cities: List[Dict],
                               country: str,
                               optimization_targets: List[str],
                               constraints: Dict,
                               route_name: str) -> GeneratedRoute:
        """Generate complete route with all components"""
        
        # Step 1: Generate base route path
        self.logger.info("🗺️ Generating base route path...")
        route_points = self._generate_route_path(cities, optimization_targets)
        
        # Step 2: Analyze terrain along route
        self.logger.info("🏔️ Analyzing terrain...")
        elevation_data = self._analyze_route_terrain(route_points)
        
        # Step 3: Optimize track routing
        self.logger.info("🛤️ Optimizing track routing...")
        track_options = self._optimize_track_routing(route_points, elevation_data, constraints)
        best_track_option = track_options[0] if track_options else None
        
        if not best_track_option:
            raise ValueError("Could not generate viable track routing")
        
        # Step 4: Optimize station placement
        self.logger.info("🚉 Optimizing station placement...")
        stations = self._optimize_station_placement(route_points, cities, constraints)
        
        # Step 5: Select optimal train types
        self.logger.info("🚄 Selecting train types...")
        train_specs = self._select_train_types(best_track_option, stations, constraints)
        
        # Step 6: Plan railyard locations
        self.logger.info("🏭 Planning railyard locations...")
        railyards = self._plan_railyards(stations, best_track_option, constraints)
        
        # Step 7: Calculate comprehensive metrics
        self.logger.info("📊 Calculating metrics...")
        metrics = self._calculate_route_metrics(
            stations, best_track_option, train_specs, railyards, optimization_targets
        )
        
        # Step 8: Assemble final route
        route = GeneratedRoute(
            name=route_name,
            stations=stations,
            track_segments=best_track_option.segments,
            total_length_km=best_track_option.total_length_km,
            total_cost=metrics['total_cost'],
            construction_time_months=best_track_option.construction_time_months,
            train_specifications=train_specs,
            railyards=railyards,
            environmental_impact_score=metrics['environmental_score'],
            feasibility_score=metrics['feasibility_score'],
            ridership_potential=metrics['ridership_potential'],
            operational_metrics=metrics['operational_metrics'],
            generation_metadata={
                'generation_time': time.time(),
                'country': country,
                'optimization_targets': optimization_targets,
                'constraints_applied': list(constraints.keys()),
                'model_versions': self._get_model_versions()
            }
        )
        
        return route
    
    def _generate_route_path(self, cities: List[Dict], optimization_targets: List[str]) -> List[Tuple[float, float]]:
        """Generate optimal route path connecting cities"""
        
        if len(cities) < 2:
            raise ValueError("Need at least 2 cities to generate a route")
        
        # Extract coordinates
        city_coords = [(city['lat'], city['lon']) for city in cities]
        
        # For simplicity, use TSP-like approach to order cities optimally
        if len(cities) > 2:
            ordered_coords = self._solve_city_ordering(city_coords, optimization_targets)
        else:
            ordered_coords = city_coords
        
        # Generate detailed route points between cities
        detailed_route = []
        for i in range(len(ordered_coords) - 1):
            start = ordered_coords[i]
            end = ordered_coords[i + 1]
            
            # Interpolate points between cities (every ~5km)
            distance = haversine_distance(start[0], start[1], end[0], end[1])
            num_points = max(3, int(distance / 5))  # Point every 5km
            
            segment_points = GeoUtils.interpolate_route(start, end, num_points)
            
            # Add to route (skip first point of subsequent segments to avoid duplicates)
            if i == 0:
                detailed_route.extend(segment_points)
            else:
                detailed_route.extend(segment_points[1:])
        
        return detailed_route
    
    def _solve_city_ordering(self, city_coords: List[Tuple[float, float]], 
                           optimization_targets: List[str]) -> List[Tuple[float, float]]:
        """Solve city ordering problem (simplified TSP)"""
        
        if len(city_coords) <= 3:
            return city_coords
        
        # Use nearest neighbor heuristic
        unvisited = city_coords[1:]  # Start from first city
        route = [city_coords[0]]
        
        while unvisited:
            current = route[-1]
            
            # Find nearest unvisited city
            nearest_idx = 0
            min_distance = float('inf')
            
            for i, candidate in enumerate(unvisited):
                distance = haversine_distance(current[0], current[1], candidate[0], candidate[1])
                
                # Apply optimization target weighting
                if "cost" in optimization_targets:
                    # Prefer shorter distances to reduce cost
                    score = distance
                elif "ridership" in optimization_targets:
                    # Consider population density (simplified)
                    score = distance * 0.8  # Slightly prefer connecting populated areas
                else:
                    score = distance
                
                if score < min_distance:
                    min_distance = score
                    nearest_idx = i
            
            # Add nearest city to route
            route.append(unvisited.pop(nearest_idx))
        
        return route
    
    def _analyze_route_terrain(self, route_points: List[Tuple[float, float]]) -> List[Dict]:
        """Analyze terrain along the route"""
        
        # Get elevation profile
        elevation_profile = self.terrain_analyzer.get_elevation_profile(route_points)
        
        # Calculate terrain metrics
        if elevation_profile:
            terrain_metrics = self.terrain_analyzer.calculate_route_metrics(elevation_profile)
            self.logger.info(f"🏔️ Terrain analysis: max elevation {terrain_metrics.get('max_elevation_m', 0):.0f}m, "
                           f"max gradient {terrain_metrics.get('max_slope_percent', 0):.1f}%")
        
        return elevation_profile
    
    def _optimize_track_routing(self, route_points: List[Tuple[float, float]], 
                              elevation_data: List[Dict], 
                              constraints: Dict) -> List[RouteOption]:
        """Optimize track routing considering terrain and constraints"""
        
        if len(route_points) < 2:
            return []
        
        # Generate route options between start and end
        start_point = route_points[0]
        end_point = route_points[-1]
        
        route_options = self.track_intelligence.optimize_track_routing(
            start_point=start_point,
            end_point=end_point,
            elevation_data=elevation_data,
            constraints=constraints
        )
        
        if route_options:
            self.logger.info(f"🛤️ Generated {len(route_options)} track routing options")
            self.logger.info(f"Best option: {route_options[0].total_length_km:.1f}km, "
                           f"€{route_options[0].total_cost/1_000_000:.1f}M cost")
        
        return route_options
    
    def _optimize_station_placement(self, route_points: List[Tuple[float, float]], 
                                  cities: List[Dict], 
                                  constraints: Dict) -> List[GeneratedStation]:
        """Optimize station placement along route"""
        
        # Create route analysis for station planning
        total_distance = GeoUtils.calculate_route_length(route_points)
        
        # Estimate demand and station metrics
        population_data = cities  # Use cities as population centers
        
        # Use station pattern analyzer
        station_placements = self.station_analyzer.optimize_station_placement(
            route_points=route_points,
            population_data=population_data,
            constraints=constraints
        )
        
        # Convert to GeneratedStation objects
        stations = []
        for i, placement in enumerate(station_placements):
            # Find nearest city for naming
            nearest_city = min(cities, key=lambda c: haversine_distance(
                placement.lat, placement.lon, c['lat'], c['lon']
            ))
            
            # Generate station name
            distance_to_city = haversine_distance(
                placement.lat, placement.lon, nearest_city['lat'], nearest_city['lon']
            )
            
            if distance_to_city < 2:  # Within 2km of city
                station_name = f"{nearest_city['name']} Central"
            else:
                station_name = f"{nearest_city['name']} {placement.station_type.title()}"
            
            # Estimate station specifications
            platform_count = self._estimate_platform_count(placement.station_type, placement.population_served)
            construction_cost = self._estimate_station_cost(placement.station_type, platform_count)
            
            station = GeneratedStation(
                name=station_name,
                lat=placement.lat,
                lon=placement.lon,
                station_type=placement.station_type,
                platform_count=platform_count,
                estimated_daily_passengers=placement.population_served // 10,  # Rough estimate
                construction_cost=construction_cost,
                accessibility_score=placement.accessibility_score,
                transfer_connections=[],  # Would be populated based on existing networks
                services=self._determine_station_services(placement.station_type),
                reasons=placement.reasons
            )
            stations.append(station)
        
        self.logger.info(f"🚉 Planned {len(stations)} stations")
        return stations
    
    def _select_train_types(self, track_option: RouteOption, 
                          stations: List[GeneratedStation], 
                          constraints: Dict) -> List[TrainSpecification]:
        """Select optimal train types for the route"""
        
        # Create route analysis
        route_analysis = RouteAnalysis(
            total_distance_km=track_option.total_length_km,
            max_gradient_percent=track_option.max_gradient,
            avg_gradient_percent=track_option.avg_gradient,
            min_curve_radius_m=2000,  # Estimated from track design
            station_count=len(stations),
            avg_station_spacing_km=track_option.total_length_km / max(1, len(stations) - 1),
            urban_percentage=0.4,  # Estimated
            international_route=self._is_international_route(stations),
            expected_demand_passengers_per_hour=sum(s.estimated_daily_passengers for s in stations) // 16,  # Peak hour
            stops_per_100km=(len(stations) / track_option.total_length_km) * 100
        )
        
        # Use train classifier
        train_options = self.train_classifier.classify_optimal_train_type(route_analysis)
        
        # Optimize service parameters for each train type
        optimized_specs = []
        for train_spec in train_options:
            optimized_spec = self.train_classifier.optimize_service_parameters(train_spec, route_analysis)
            optimized_specs.append(optimized_spec)
        
        if optimized_specs:
            self.logger.info(f"🚄 Selected {len(optimized_specs)} train types: " + 
                           ", ".join([spec.category.value for spec in optimized_specs]))
        
        return optimized_specs
    
    def _plan_railyards(self, stations: List[GeneratedStation], 
                       track_option: RouteOption, 
                       constraints: Dict) -> List[RailyardCandidate]:
        """Plan railyard locations for maintenance and operations"""
        
        # Convert stations to format expected by railyard optimizer
        station_data = [
            {
                'lat': s.lat,
                'lon': s.lon,
                'name': s.name,
                'category': s.station_type,
                'platforms': s.platform_count
            }
            for s in stations
        ]
        
        # Mock terrain data (in practice would use real terrain analysis)
        terrain_data = {'terrain_type': 'moderate'}
        
        # Mock network demand (in practice would use ridership projections)
        network_demand = {'freight_score': 0.3, 'passenger_score': 0.7}
        
        # Use railyard optimizer
        railyard_candidates = self.railyard_optimizer.optimize_railyard_locations(
            stations=station_data,
            terrain_data=terrain_data,
            network_demand=network_demand,
            budget_constraints=constraints.get('budget')
        )
        
        if railyard_candidates:
            self.logger.info(f"🏭 Planned {len(railyard_candidates)} railyards")
        
        return railyard_candidates
    
    def _calculate_route_metrics(self, stations: List[GeneratedStation], 
                               track_option: RouteOption, 
                               train_specs: List[TrainSpecification], 
                               railyards: List[RailyardCandidate],
                               optimization_targets: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive route performance metrics"""
        
        # Calculate total costs
        station_costs = sum(s.construction_cost for s in stations)
        track_costs = track_option.total_cost
        railyard_costs = sum(r.land_cost_factor * 50_000_000 for r in railyards)  # €50M base cost
        total_cost = station_costs + track_costs + railyard_costs
        
        # Calculate ridership potential
        ridership_potential = sum(s.estimated_daily_passengers for s in stations) * 365
        
        # Calculate environmental impact
        environmental_score = 1.0 - track_option.environmental_score  # Convert to positive score
        
        # Calculate feasibility score
        feasibility_score = 0.8  # Simplified - would involve detailed engineering analysis
        
        # Calculate operational metrics
        if train_specs:
            primary_train = train_specs[0]
            operational_metrics = self.train_classifier.calculate_operational_metrics(
                primary_train, 
                RouteAnalysis(
                    total_distance_km=track_option.total_length_km,
                    max_gradient_percent=track_option.max_gradient,
                    avg_gradient_percent=track_option.avg_gradient,
                    min_curve_radius_m=2000,
                    station_count=len(stations),
                    avg_station_spacing_km=track_option.total_length_km / max(1, len(stations) - 1),
                    urban_percentage=0.4,
                    international_route=False,
                    expected_demand_passengers_per_hour=ridership_potential // (365 * 16),
                    stops_per_100km=(len(stations) / track_option.total_length_km) * 100
                )
            )
        else:
            operational_metrics = {}
        
        return {
            'total_cost': total_cost,
            'station_costs': station_costs,
            'track_costs': track_costs,
            'railyard_costs': railyard_costs,
            'ridership_potential': ridership_potential,
            'environmental_score': environmental_score,
            'feasibility_score': feasibility_score,
            'operational_metrics': operational_metrics,
            'cost_per_km': total_cost / track_option.total_length_km,
            'cost_per_passenger': total_cost / max(1, ridership_potential),
        }
    
    # Helper methods
    
    def _estimate_platform_count(self, station_type: str, population_served: int) -> int:
        """Estimate required platform count"""
        if station_type == 'intercity':
            return min(8, max(4, population_served // 50000))
        elif station_type == 'regional':
            return min(6, max(2, population_served // 25000))
        else:  # local, suburban
            return min(4, max(2, population_served // 10000))
    
    def _estimate_station_cost(self, station_type: str, platform_count: int) -> float:
        """Estimate station construction cost"""
        base_cost = self.config.costs.station_base_cost
        complexity_multiplier = self.config.costs.station_complexity_multipliers.get(station_type, 1.0)
        platform_cost = platform_count * 150 * self.config.costs.platform_cost_per_meter  # 150m platforms
        
        return base_cost * complexity_multiplier + platform_cost
    
    def _determine_station_services(self, station_type: str) -> List[str]:
        """Determine services available at station based on type"""
        base_services = ['ticketing', 'waiting_area']
        
        if station_type in ['intercity', 'major']:
            base_services.extend(['restaurants', 'shops', 'wifi', 'parking', 'car_rental'])
        elif station_type == 'regional':
            base_services.extend(['cafe', 'wifi', 'parking'])
        else:  # local, suburban
            base_services.extend(['parking'])
        
        return base_services
    
    def _is_international_route(self, stations: List[GeneratedStation]) -> bool:
        """Determine if route crosses international borders"""
        # Simplified: check if stations span large distance (>300km suggests international)
        if len(stations) < 2:
            return False
        
        max_distance = 0
        for i in range(len(stations)):
            for j in range(i + 1, len(stations)):
                distance = haversine_distance(
                    stations[i].lat, stations[i].lon,
                    stations[j].lat, stations[j].lon
                )
                max_distance = max(max_distance, distance)
        
        return max_distance > 300  # >300km suggests international route
    
    def _get_model_versions(self) -> Dict[str, str]:
        """Get versions of trained models used"""
        return {
            'station_analyzer': 'v1.0',
            'track_intelligence': 'v1.0',
            'train_classifier': 'v1.0',
            'railyard_optimizer': 'v1.0',
            'generation_time': datetime.now().isoformat()
        }
    
    def save_plan(self, route: GeneratedRoute, file_path: str):
        """Save generated route plan to file"""
        
        # Convert route to serializable format
        route_dict = self._route_to_dict(route)
        
        # Save to JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(route_dict, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"💾 Route plan saved to {file_path}")
    
    def _route_to_dict(self, route: GeneratedRoute) -> Dict[str, Any]:
        """Convert route to dictionary for serialization"""
        
        def convert_dataclass(obj):
            if hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    if hasattr(value, '__dict__'):
                        result[key] = convert_dataclass(value)
                    elif isinstance(value, list):
                        result[key] = [convert_dataclass(item) if hasattr(item, '__dict__') else item for item in value]
                    elif isinstance(value, dict):
                        result[key] = {k: convert_dataclass(v) if hasattr(v, '__dict__') else v for k, v in value.items()}
                    else:
                        result[key] = value
                return result
            else:
                return obj
        
        return convert_dataclass(route)
    
    def load_plan(self, file_path: str) -> Dict[str, Any]:
        """Load route plan from file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get route generation statistics"""
        return self.generation_stats.copy()

# Example usage
if __name__ == "__main__":
    # Example route generation
    from railway_ai.config import create_development_config
    
    config = create_development_config()
    generator = RouteGenerator(config)
    
    