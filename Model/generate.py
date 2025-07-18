# File: Model/generate.py
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

try:
    from .config import RailwayConfig
    from .intelligence.station_patterns import StationPatternAnalyzer, StationPlacement
    from .intelligence.track_intelligence import TrackIntelligence, RouteOption, TrackSegment
    from .intelligence.train_classifier import TrainClassifier, RouteAnalysis, TrainSpecification
    from .intelligence.railyard_optimizer import RailyardOptimizer, RailyardCandidate
    from .extractors.terrain_analysis import TerrainAnalyzer
    from .utils.geo import GeoUtils, haversine_distance, BoundingBox
    from .utils.ml import MLPipeline, RailwayMLUtils
except ImportError:
    # Fallback for missing modules
    print("Warning: Some modules not available, using simplified implementations")

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
    track_segments: List[Any]  # Changed from TrackSegment to Any for compatibility
    total_length_km: float
    total_cost: float
    construction_time_months: int
    train_specifications: List[Any]  # Changed from TrainSpecification to Any
    railyards: List[Any]  # Changed from RailyardCandidate to Any
    environmental_impact_score: float
    feasibility_score: float
    ridership_potential: int
    operational_metrics: Dict[str, Any]
    generation_metadata: Dict[str, Any]

class RouteGenerator:
    """Main route generation engine"""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components (with fallbacks for missing modules)
        try:
            self.station_analyzer = StationPatternAnalyzer()
            self.track_intelligence = TrackIntelligence()
            self.train_classifier = TrainClassifier()
            self.railyard_optimizer = RailyardOptimizer()
            self.terrain_analyzer = TerrainAnalyzer()
            self.ml_pipeline = MLPipeline()
        except:
            self.logger.warning("Some intelligence modules not available, using simplified approach")
            self.station_analyzer = None
            self.track_intelligence = None
            self.train_classifier = None
            self.railyard_optimizer = None
            self.terrain_analyzer = None
            self.ml_pipeline = None
        
        # Load trained models if available
        self._load_trained_models()
        
        self.generation_stats = {
            'routes_generated': 0,
            'total_generation_time': 0,
            'average_generation_time': 0
        }
    
    def _load_trained_models(self):
        """Load previously trained ML models"""
        try:
            if self.config and hasattr(self.config, 'paths'):
                models_dir = self.config.paths.models_dir
                if models_dir and models_dir.exists():
                    if self.ml_pipeline:
                        self.ml_pipeline.load_models(str(models_dir))
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
            country = "lebanon"  # Default for this example
        if route_name is None:
            route_name = f"Generated_Route_{int(time.time())}"
        
        # Generate route plan (simplified version for Lebanon)
        try:
            route = self._generate_lebanon_route(
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
        
        return cities
    
    def _geocode_cities(self, city_names: List[str]) -> List[Dict]:
        """Convert city names to coordinates (simplified geocoding)"""
        # Lebanon cities coordinates
        lebanon_cities = {
            'tyre': {'lat': 33.2704, 'lon': 35.2038, 'population': 200000},
            'tripoli': {'lat': 34.4359, 'lon': 35.8492, 'population': 730000},
            'beirut': {'lat': 33.8886, 'lon': 35.4955, 'population': 2400000},
            'jounieh': {'lat': 33.9811, 'lon': 35.6178, 'population': 150000},
            'sidon': {'lat': 33.5614, 'lon': 35.3712, 'population': 260000},
            'baalbek': {'lat': 34.0067, 'lon': 36.2117, 'population': 82000},
            'zahle': {'lat': 33.8462, 'lon': 35.9018, 'population': 120000},
        }
        
        cities = []
        for city_name in city_names:
            city_key = city_name.lower().strip()
            if city_key in lebanon_cities:
                city_data = lebanon_cities[city_key].copy()
                city_data['name'] = city_name.title()
                cities.append(city_data)
            else:
                # Fallback: place near Beirut with estimated data
                self.logger.warning(f"⚠️ Unknown city '{city_name}', using estimated location")
                cities.append({
                    'name': city_name.title(),
                    'lat': 33.8 + (len(cities) * 0.1),
                    'lon': 35.4 + (len(cities) * 0.1),
                    'population': 100000
                })
        
        return cities
    
    def _generate_lebanon_route(self,
                               cities: List[Dict],
                               country: str,
                               optimization_targets: List[str],
                               constraints: Dict,
                               route_name: str) -> GeneratedRoute:
        """Generate Lebanon-specific route (simplified implementation)"""
        
        if len(cities) < 2:
            raise ValueError("Need at least 2 cities to generate a route")
        
        # Calculate total distance
        total_distance = 0
        for i in range(len(cities) - 1):
            distance = self._haversine_distance(
                cities[i]['lat'], cities[i]['lon'],
                cities[i+1]['lat'], cities[i+1]['lon']
            )
            total_distance += distance
        
        # Generate stations
        stations = []
        for i, city in enumerate(cities):
            station = GeneratedStation(
                name=f"{city['name']} Central",
                lat=city['lat'],
                lon=city['lon'],
                station_type='intercity' if city['population'] > 500000 else 'regional',
                platform_count=min(6, max(2, city['population'] // 100000)),
                estimated_daily_passengers=city['population'] // 50,
                construction_cost=50_000_000 + (city['population'] // 10000) * 1_000_000,
                accessibility_score=0.8,
                transfer_connections=[],
                services=['ticketing', 'waiting_area', 'wifi', 'parking'],
                reasons=[f"Major station serving {city['name']} metropolitan area"]
            )
            stations.append(station)
        
        # Calculate costs (French technology premium)
        base_cost_per_km = 15_000_000  # €15M per km (French high-speed standard)
        french_technology_premium = 1.3  # 30% premium for French tech
        track_cost = total_distance * base_cost_per_km * french_technology_premium
        station_cost = sum(s.construction_cost for s in stations)
        total_cost = track_cost + station_cost
        
        # Create route
        route = GeneratedRoute(
            name=route_name,
            stations=stations,
            track_segments=[],  # Simplified
            total_length_km=total_distance,
            total_cost=total_cost,
            construction_time_months=int(total_distance / 10) + 24,  # Rough estimate
            train_specifications=[],  # Simplified
            railyards=[],  # Simplified
            environmental_impact_score=0.7,  # Good score for modern French tech
            feasibility_score=0.85,  # High feasibility
            ridership_potential=sum(s.estimated_daily_passengers for s in stations) * 365,
            operational_metrics={
                'cost_per_km': total_cost / total_distance,
                'max_speed_kmh': 200,  # French high-speed capability
                'travel_time_minutes': (total_distance / 200) * 60,  # At max speed
                'french_technology': True,
                'electrification': '25kV AC',
                'gauge': '1435mm standard'
            },
            generation_metadata={
                'generation_time': time.time(),
                'country': country,
                'optimization_targets': optimization_targets,
                'constraints_applied': list(constraints.keys()),
                'french_grant_integration': True,
                'budget_eur': 10_000_000_000
            }
        )
        
        return route
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in km"""
        R = 6371  # Earth's radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    def save_plan(self, route: GeneratedRoute, file_path: str):
        """Save generated route plan to file"""
        
        # Convert route to serializable format
        route_dict = self._route_to_dict(route)
        
        # Ensure output directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(route_dict, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"💾 Route plan saved to {file_path}")
    
    def _route_to_dict(self, route: GeneratedRoute) -> Dict[str, Any]:
        """Convert route to dictionary for serialization"""
        
        def convert_obj(obj):
            if hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    if hasattr(value, '__dict__'):
                        result[key] = convert_obj(value)
                    elif isinstance(value, list):
                        result[key] = [convert_obj(item) if hasattr(item, '__dict__') else item for item in value]
                    elif isinstance(value, dict):
                        result[key] = {k: convert_obj(v) if hasattr(v, '__dict__') else v for k, v in value.items()}
                    else:
                        result[key] = value
                return result
            else:
                return obj
        
        return convert_obj(route)
    
    def load_plan(self, file_path: str) -> Dict[str, Any]:
        """Load route plan from file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get route generation statistics"""
        return self.generation_stats.copy()