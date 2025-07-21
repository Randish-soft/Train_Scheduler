# File: Model/learn.py
"""
Railway Learning Module
Learns patterns and intelligence from existing railway networks.
"""
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from .config import RailwayConfig
from .extractors.osm_railway import OSMRailwayExtractor
from .extractors.terrain_analysis import TerrainAnalyzer
from .extractors.network_parser import RailwayNetworkParser
from .intelligence.station_patterns import StationPatternAnalyzer
from .intelligence.track_intelligence import TrackIntelligence
from .intelligence.train_classifier import TrainClassifier
from .intelligence.railyard_optimizer import RailyardOptimizer
from .utils.geo import get_country_bounds, haversine_distance
from .utils.ml import MLPipeline, RailwayMLUtils

@dataclass
class LearningResults:
    """Results from railway learning process"""
    country: str
    learning_time: float
    stations_analyzed: int = 0
    track_segments: int = 0
    railyards_found: int = 0
    train_services_analyzed: int = 0
    
    # Pattern statistics
    station_patterns: Dict[str, Any] = field(default_factory=dict)
    track_patterns: Dict[str, Any] = field(default_factory=dict)
    operational_patterns: Dict[str, Any] = field(default_factory=dict)
    cost_patterns: Dict[str, Any] = field(default_factory=dict)
    
    # Model performance
    model_accuracies: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Quality metrics
    data_quality_score: float = 0.0
    coverage_completeness: float = 0.0
    learning_confidence: float = 0.0
    
    # Learned insights
    key_insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    learning_date: datetime = field(default_factory=datetime.now)
    data_sources: List[str] = field(default_factory=list)
    focus_areas: List[str] = field(default_factory=list)

class RailwayLearner:
    """Main learning engine that extracts intelligence from existing railways"""
    
    def __init__(self, country: str, train_types: List[str], config: RailwayConfig):
        self.country = country.upper()
        self.train_types = train_types
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize extractors
        self.osm_extractor = OSMRailwayExtractor()
        self.terrain_analyzer = TerrainAnalyzer()
        self.network_parser = RailwayNetworkParser()
        
        # Initialize intelligence modules
        self.station_analyzer = StationPatternAnalyzer()
        self.track_intelligence = TrackIntelligence()
        self.train_classifier = TrainClassifier()
        self.railyard_optimizer = RailyardOptimizer()
        
        # Initialize ML pipeline
        self.ml_pipeline = MLPipeline(random_state=self.config.ml.random_state)
        
        # Learning cache
        self.extracted_data = {}
        self.processed_data = {}
        
    def execute(self, 
               focus: Optional[List[str]] = None,
               data_sources: Optional[str] = None) -> LearningResults:
        """Execute the complete learning process"""
        
        start_time = time.time()
        self.logger.info(f"🧠 Starting learning from {self.country} railway network...")
        
        # Initialize results
        results = LearningResults(
            country=self.country,
            learning_time=0,
            focus_areas=focus or [],
            data_sources=data_sources.split(',') if data_sources else []
        )
        
        try:
            # Phase 1: Data Extraction
            self.logger.info("📡 Phase 1: Extracting railway data...")
            extraction_results = self._extract_railway_data()
            results.stations_analyzed = extraction_results.get('stations_count', 0)
            results.track_segments = extraction_results.get('tracks_count', 0)
            
            # Phase 2: Data Processing and Analysis
            self.logger.info("🔍 Phase 2: Processing and analyzing data...")
            analysis_results = self._analyze_extracted_data(extraction_results)
            
            # Phase 3: Pattern Learning
            self.logger.info("🎓 Phase 3: Learning operational patterns...")
            pattern_results = self._learn_patterns(analysis_results, focus)
            results.station_patterns = pattern_results.get('station_patterns', {})
            results.track_patterns = pattern_results.get('track_patterns', {})
            results.operational_patterns = pattern_results.get('operational_patterns', {})
            
            # Phase 4: ML Model Training
            self.logger.info("🤖 Phase 4: Training ML models...")
            ml_results = self._train_ml_models(analysis_results, focus)
            results.model_accuracies = ml_results.get('model_accuracies', {})
            results.feature_importance = ml_results.get('feature_importance', {})
            
            # Phase 5: Quality Assessment
            self.logger.info("📊 Phase 5: Assessing learning quality...")
            quality_results = self._assess_learning_quality(extraction_results, analysis_results)
            results.data_quality_score = quality_results.get('data_quality', 0.0)
            results.coverage_completeness = quality_results.get('coverage', 0.0)
            results.learning_confidence = quality_results.get('confidence', 0.0)
            
            # Phase 6: Generate Insights
            self.logger.info("💡 Phase 6: Generating insights and recommendations...")
            insights = self._generate_insights(results)
            results.key_insights = insights.get('insights', [])
            results.recommendations = insights.get('recommendations', [])
            
            # Record timing
            results.learning_time = time.time() - start_time
            
            self.logger.info(f"✅ Learning completed successfully in {results.learning_time:.1f}s")
            self.logger.info(f"📊 Analyzed {results.stations_analyzed} stations, {results.track_segments} track segments")
            self.logger.info(f"🎯 Learning confidence: {results.learning_confidence:.1%}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Learning failed: {e}")
            results.learning_time = time.time() - start_time
            raise
    
    def _extract_railway_data(self) -> Dict[str, Any]:
        """Extract raw railway data from various sources"""
        
        extraction_results = {
            'stations': [],
            'tracks': [],
            'network_graph': None,
            'terrain_data': {},
            'operational_data': []
        }
        
        # Get country bounding box
        country_bounds = get_country_bounds(self.country)
        if not country_bounds:
            raise ValueError(f"Unknown country code: {self.country}")
        
        self.logger.info(f"🗺️ Extracting data for {self.country} within bounds {country_bounds}")
        
        try:
            # Extract railway infrastructure from OSM
            self.logger.info("🚉 Extracting stations from OpenStreetMap...")
            stations = self.osm_extractor.extract_train_stations(self.country)
            extraction_results['stations'] = stations
            extraction_results['stations_count'] = len(stations)
            
            self.logger.info(f"📍 Found {len(stations)} stations")
            
            # Extract track network
            self.logger.info("🛤️ Extracting track network...")
            tracks = self.osm_extractor.extract_railway_tracks(self.country)
            extraction_results['tracks'] = tracks
            extraction_results['tracks_count'] = len(tracks)
            
            self.logger.info(f"🚂 Found {len(tracks)} track segments")
            
            # Build network graph
            if stations and tracks:
                self.logger.info("🕸️ Building network graph...")
                osm_data = {
                    'elements': []
                }
                
                # Convert stations to OSM format
                for station in stations:
                    osm_element = {
                        'type': 'node',
                        'id': station['id'],
                        'lat': station['lat'],
                        'lon': station['lon'],
                        'tags': {
                            'railway': 'station',
                            'name': station['name'],
                            'operator': station.get('operator', ''),
                            'platforms': str(station.get('platforms', 1))
                        }
                    }
                    osm_data['elements'].append(osm_element)
                
                # Convert tracks to OSM format
                for track in tracks:
                    osm_element = {
                        'type': 'way',
                        'id': track['id'],
                        'geometry': track['geometry'],
                        'tags': {
                            'railway': 'rail',
                            'maxspeed': str(track.get('maxspeed', 100)),
                            'electrified': 'yes' if track.get('electrified') else 'no',
                            'usage': track.get('usage', 'main')
                        }
                    }
                    osm_data['elements'].append(osm_element)
                
                # Build network graph
                network_graph = self.network_parser.parse_osm_to_network(osm_data)
                extraction_results['network_graph'] = network_graph
                
                # Analyze network patterns
                network_analysis = self.network_parser.analyze_network_patterns()
                extraction_results['network_analysis'] = network_analysis
                
                self.logger.info(f"🔗 Built network with {network_graph.number_of_nodes()} nodes, "
                               f"{network_graph.number_of_edges()} edges")
            
            # Extract terrain data for key routes
            if stations and len(stations) >= 2:
                self.logger.info("🏔️ Analyzing terrain along major routes...")
                terrain_data = self._extract_terrain_for_routes(stations[:10])  # Sample first 10 stations
                extraction_results['terrain_data'] = terrain_data
            
            # Cache extracted data
            self.extracted_data = extraction_results
            
        except Exception as e:
            self.logger.error(f"❌ Data extraction failed: {e}")
            # Continue with partial data if possible
            if not extraction_results['stations']:
                raise
        
        return extraction_results
    
    def _extract_terrain_for_routes(self, stations: List[Dict]) -> Dict[str, Any]:
        """Extract terrain data for routes between major stations"""
        
        terrain_data = {
            'route_profiles': [],
            'elevation_statistics': {},
            'gradient_analysis': {}
        }
        
        # Analyze terrain between consecutive station pairs
        for i in range(min(5, len(stations) - 1)):  # Limit to 5 routes to avoid API limits
            start_station = stations[i]
            end_station = stations[i + 1]
            
            try:
                # Create route points
                route_points = [
                    (start_station['lat'], start_station['lon']),
                    (end_station['lat'], end_station['lon'])
                ]
                
                # Get elevation profile
                elevation_profile = self.terrain_analyzer.get_elevation_profile(
                    route_points, sample_rate=20  # Reduced sample rate
                )
                
                if elevation_profile:
                    # Calculate route metrics
                    route_metrics = self.terrain_analyzer.calculate_route_metrics(elevation_profile)
                    
                    terrain_data['route_profiles'].append({
                        'start_station': start_station['name'],
                        'end_station': end_station['name'],
                        'elevation_profile': elevation_profile[:10],  # Store first 10 points
                        'metrics': route_metrics
                    })
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to analyze terrain for {start_station['name']} - {end_station['name']}: {e}")
        
        return terrain_data
    
    def _analyze_extracted_data(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process and analyze extracted railway data"""
        
        analysis_results = {
            'station_analysis': {},
            'track_analysis': {},
            'service_analysis': {},
            'network_topology': {},
            'operational_metrics': {}
        }
        
        stations = extraction_results.get('stations', [])
        tracks = extraction_results.get('tracks', [])
        network_graph = extraction_results.get('network_graph')
        
        # Analyze station characteristics
        if stations:
            self.logger.info("🚉 Analyzing station characteristics...")
            station_analysis = self._analyze_stations(stations)
            analysis_results['station_analysis'] = station_analysis
        
        # Analyze track characteristics
        if tracks:
            self.logger.info("🛤️ Analyzing track characteristics...")
            track_analysis = self._analyze_tracks(tracks)
            analysis_results['track_analysis'] = track_analysis
        
        # Analyze service patterns
        if stations and tracks:
            self.logger.info("🚄 Analyzing service patterns...")
            service_analysis = self._analyze_services(stations, tracks)
            analysis_results['service_analysis'] = service_analysis
        
        # Analyze network topology
        if network_graph:
            self.logger.info("🕸️ Analyzing network topology...")
            topology_analysis = self._analyze_network_topology(network_graph)
            analysis_results['network_topology'] = topology_analysis
        
        return analysis_results
    
    def _analyze_stations(self, stations: List[Dict]) -> Dict[str, Any]:
        """Analyze station characteristics and patterns"""
        
        analysis = {
            'total_count': len(stations),
            'categories': {},
            'operators': {},
            'platform_distribution': {},
            'geographic_distribution': {},
            'spacing_analysis': {}
        }
        
        # Categorize stations
        categories = {}
        operators = {}
        platform_counts = []
        
        for station in stations:
            # Station category
            category = station.get('level', 'regional')
            categories[category] = categories.get(category, 0) + 1
            
            # Operator
            operator = station.get('operator', 'unknown')
            operators[operator] = operators.get(operator, 0) + 1
            
            # Platform count
            platforms = station.get('platforms', 1)
            platform_counts.append(platforms)
        
        analysis['categories'] = categories
        analysis['operators'] = operators
        analysis['platform_distribution'] = {
            'mean': np.mean([int(x) if x.isdigit() else 0 for x in platform_counts]),
            'median': np.median(platform_counts),
            'max': np.max(platform_counts),
            'distribution': dict(zip(*np.unique(platform_counts, return_counts=True)))
        }
        
        # Geographic distribution
        if len(stations) > 1:
            lats = [s['lat'] for s in stations]
            lons = [s['lon'] for s in stations]
            
            analysis['geographic_distribution'] = {
                'lat_range': max(lats) - min(lats),
                'lon_range': max(lons) - min(lons),
                'center_lat': np.mean(lats),
                'center_lon': np.mean(lons)
            }
            
            # Station spacing analysis
            spacings = []
            for i in range(len(stations) - 1):
                for j in range(i + 1, len(stations)):
                    distance = haversine_distance(
                        stations[i]['lat'], stations[i]['lon'],
                        stations[j]['lat'], stations[j]['lon']
                    )
                    spacings.append(distance)
            
            if spacings:
                analysis['spacing_analysis'] = {
                    'mean_spacing': np.mean(spacings),
                    'median_spacing': np.median(spacings),
                    'min_spacing': np.min(spacings),
                    'max_spacing': np.max(spacings)
                }
        
        return analysis
    
    def _analyze_tracks(self, tracks: List[Dict]) -> Dict[str, Any]:
        """Analyze track characteristics and infrastructure"""
        
        analysis = {
            'total_count': len(tracks),
            'electrification': {},
            'speed_distribution': {},
            'track_types': {},
            'gauge_analysis': {},
            'infrastructure_quality': {}
        }
        
        # Analyze electrification
        electrified_count = sum(1 for track in tracks if track.get('electrified'))
        analysis['electrification'] = {
            'electrified_count': electrified_count,
            'electrification_ratio': electrified_count / len(tracks),
            'non_electrified_count': len(tracks) - electrified_count
        }
        
        # Speed distribution
        speeds = [track.get('maxspeed', 100) for track in tracks]
        analysis['speed_distribution'] = {
            'mean_speed': np.mean(speeds),
            'median_speed': np.median(speeds),
            'max_speed': np.max(speeds),
            'speed_categories': {
                'high_speed_300plus': sum(1 for s in speeds if s >= 300),
                'fast_200_300': sum(1 for s in speeds if 200 <= s < 300),
                'medium_120_200': sum(1 for s in speeds if 120 <= s < 200),
                'low_below_120': sum(1 for s in speeds if s < 120)
            }
        }
        
        # Track usage patterns
        usage_types = {}
        for track in tracks:
            usage = track.get('usage', 'main')
            usage_types[usage] = usage_types.get(usage, 0) + 1
        
        analysis['track_types'] = usage_types
        
        # Gauge analysis
        gauges = [track.get('gauge', '1435') for track in tracks]
        gauge_distribution = {}
        for gauge in gauges:
            gauge_distribution[gauge] = gauge_distribution.get(gauge, 0) + 1
        
        analysis['gauge_analysis'] = gauge_distribution
        
        return analysis
    
    def _analyze_services(self, stations: List[Dict], tracks: List[Dict]) -> Dict[str, Any]:
        """Analyze service patterns and train operations"""
        
        analysis = {
            'service_types': {},
            'frequency_patterns': {},
            'capacity_analysis': {},
            'route_characteristics': {}
        }
        
        # Infer service types from station and track characteristics
        service_types = {}
        
        for station in stations:
            platforms = station.get('platforms', 1)
            
            # Classify service type based on platform count and location
            if platforms >= 6:
                service_type = 'intercity'
            elif platforms >= 4:
                service_type = 'regional'
            elif platforms >= 2:
                service_type = 'suburban'
            else:
                service_type = 'local'
            
            service_types[service_type] = service_types.get(service_type, 0) + 1
        
        analysis['service_types'] = service_types
        
        # Estimate frequency patterns based on infrastructure
        high_frequency_indicators = 0
        total_electrified = sum(1 for track in tracks if track.get('electrified'))
        
        if total_electrified / len(tracks) > 0.8:  # High electrification
            high_frequency_indicators += 1
        
        if len([s for s in stations if s.get('platforms', 1) >= 4]) > len(stations) * 0.3:
            high_frequency_indicators += 1
        
        frequency_category = 'high' if high_frequency_indicators >= 2 else 'medium' if high_frequency_indicators == 1 else 'low'
        
        analysis['frequency_patterns'] = {
            'frequency_category': frequency_category,
            'electrification_ratio': total_electrified / len(tracks),
            'major_stations_ratio': len([s for s in stations if s.get('platforms', 1) >= 4]) / len(stations)
        }
        
        return analysis
    
    def _analyze_network_topology(self, network_graph) -> Dict[str, Any]:
        """Analyze network topology and connectivity patterns"""
        
        if not network_graph or network_graph.number_of_nodes() == 0:
            return {}
        
        analysis = {
            'basic_metrics': {},
            'connectivity': {},
            'centrality': {},
            'efficiency': {}
        }
        
        # Basic network metrics
        analysis['basic_metrics'] = {
            'nodes': network_graph.number_of_nodes(),
            'edges': network_graph.number_of_edges(),
            'density': network_graph.number_of_edges() / (network_graph.number_of_nodes() * (network_graph.number_of_nodes() - 1)) if network_graph.number_of_nodes() > 1 else 0,
            'components': len(list(network_graph.subgraph(c) for c in network_graph.connected_components()))
        }
        
        # Degree distribution
        degrees = [d for n, d in network_graph.degree()]
        if degrees:
            analysis['connectivity'] = {
                'avg_degree': np.mean(degrees),
                'max_degree': np.max(degrees),
                'degree_std': np.std(degrees),
                'hub_count': sum(1 for d in degrees if d >= 4)  # Stations with 4+ connections
            }
        
        return analysis
    
    def _learn_patterns(self, analysis_results: Dict[str, Any], focus: Optional[List[str]]) -> Dict[str, Any]:
        """Learn operational and design patterns from analyzed data"""
        
        pattern_results = {
            'station_patterns': {},
            'track_patterns': {},
            'operational_patterns': {},
            'design_principles': {}
        }
        
        stations_data = self.extracted_data.get('stations', [])
        tracks_data = self.extracted_data.get('tracks', [])
        
        # Learn station patterns
        if stations_data and (not focus or 'stations' in focus):
            self.logger.info("🚉 Learning station placement patterns...")
            try:
                self.station_analyzer.learn_from_existing_stations(stations_data)
                station_patterns = self.station_analyzer.get_learned_patterns()
                pattern_results['station_patterns'] = station_patterns
            except Exception as e:
                self.logger.warning(f"⚠️ Station pattern learning failed: {e}")
        
        # Learn track intelligence
        if tracks_data and (not focus or 'tracks' in focus):
            self.logger.info("🛤️ Learning track routing intelligence...")
            try:
                # Convert tracks data to format expected by track intelligence
                track_learning_data = []
                terrain_data = self.extracted_data.get('terrain_data', {})
                
                for track in tracks_data[:20]:  # Limit for performance
                    track_info = {
                        'id': track['id'],
                        'maxspeed': track.get('maxspeed', 100),
                        'electrified': track.get('electrified', False),
                        'usage': track.get('usage', 'main'),
                        'engineering_type': 'surface',  # Simplified
                        'elevation_profile': []  # Would need terrain data
                    }
                    track_learning_data.append(track_info)
                
                if track_learning_data:
                    self.track_intelligence.learn_from_existing_tracks(track_learning_data, [])
                    track_patterns = self.track_intelligence.get_learned_patterns()
                    pattern_results['track_patterns'] = track_patterns
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Track intelligence learning failed: {e}")
        
        # Learn train service patterns
        if stations_data and tracks_data and (not focus or 'services' in focus):
            self.logger.info("🚄 Learning train service patterns...")
            try:
                # Create service data from infrastructure analysis
                service_data = self._create_service_data(analysis_results)
                
                if service_data:
                    self.train_classifier.learn_from_existing_services(service_data)
                    service_patterns = self.train_classifier.get_learned_patterns()
                    pattern_results['operational_patterns'] = service_patterns
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Service pattern learning failed: {e}")
        
        # Learn railyard patterns
        if stations_data and (not focus or 'railyards' in focus):
            self.logger.info("🏭 Learning railyard placement patterns...")
            try:
                # Create railyard data from station analysis
                railyard_data = self._infer_railyard_data(stations_data)
                
                if railyard_data:
                    self.railyard_optimizer.learn_from_existing_railyards(railyard_data)
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Railyard pattern learning failed: {e}")
        
        return pattern_results
    
    def _create_service_data(self, analysis_results: Dict[str, Any]) -> List[Dict]:
        """Create service data from infrastructure analysis"""
        
        service_data = []
        station_analysis = analysis_results.get('station_analysis', {})
        track_analysis = analysis_results.get('track_analysis', {})
        
        # Infer service characteristics from infrastructure
        avg_spacing = station_analysis.get('spacing_analysis', {}).get('mean_spacing', 15)
        electrification_ratio = track_analysis.get('electrification', {}).get('electrification_ratio', 0.5)
        avg_speed = track_analysis.get('speed_distribution', {}).get('mean_speed', 120)
        
        # Create representative service entries
        service_types = ['ICE', 'IC', 'RE', 'S']
        
        for service_type in service_types:
            # Estimate service characteristics based on infrastructure
            if service_type == 'ICE':
                route_distance = avg_spacing * 8  # ICE connects distant cities
                max_gradient = 2.5
                avg_commercial_speed = min(avg_speed * 0.8, 200)
                stops_per_100km = 2
            elif service_type == 'IC':
                route_distance = avg_spacing * 4
                max_gradient = 3.0
                avg_commercial_speed = min(avg_speed * 0.7, 150)
                stops_per_100km = 4
            elif service_type == 'RE':
                route_distance = avg_spacing * 2
                max_gradient = 3.5
                avg_commercial_speed = min(avg_speed * 0.6, 120)
                stops_per_100km = 8
            else:  # S
                route_distance = avg_spacing
                max_gradient = 4.0
                avg_commercial_speed = min(avg_speed * 0.5, 100)
                stops_per_100km = 15
            
            service_entry = {
                'train_type': service_type,
                'route_distance_km': route_distance,
                'max_gradient_percent': max_gradient,
                'avg_gradient_percent': max_gradient * 0.4,
                'min_curve_radius_m': 1000 if service_type in ['ICE', 'IC'] else 500,
                'station_count': max(2, int(route_distance / avg_spacing)),
                'avg_station_spacing_km': avg_spacing,
                'urban_percentage': 0.6 if service_type == 'S' else 0.3,
                'crosses_border': service_type in ['ICE', 'IC'],
                'daily_passengers': 5000 * (4 if service_type == 'S' else 2 if service_type == 'RE' else 1),
                'stops_per_100km': stops_per_100km,
                'electrified_percentage': electrification_ratio,
                'avg_commercial_speed_kmh': avg_commercial_speed
            }
            
            service_data.append(service_entry)
        
        return service_data
    
    def _infer_railyard_data(self, stations_data: List[Dict]) -> List[Dict]:
        """Infer railyard characteristics from station data"""
        
        railyard_data = []
        
        # Find potential railyard locations (large stations, end-of-line stations)
        major_stations = [s for s in stations_data if s.get('platforms', 1) >= 6]
        
        for station in major_stations[:5]:  # Limit to 5 major stations
            # Estimate railyard characteristics
            railyard_entry = {
                'distance_to_city_center': 5.0,  # Estimated
                'distance_to_major_station': 2.0,
                'elevation': 100,  # Default
                'land_area_hectares': 25,
                'track_connections': station.get('platforms', 4),
                'industrial_proximity': 0.4,
                'highway_access': 0.7,
                'population_density_1km': 2000,
                'freight_demand_score': 0.5,
                'terrain_flatness': 0.8,
                'water_access': 0.2,
                'power_grid_distance': 1.0,
                'utilization_rate': 0.75,
                'on_time_performance': 0.85,
                'cost_efficiency': 0.7,
                'congestion_incidents': 3,
                'maintenance_issues': 2,
                'capacity': 30
            }
            
            railyard_data.append(railyard_entry)
        
        return railyard_data
    
    def _train_ml_models(self, analysis_results: Dict[str, Any], focus: Optional[List[str]]) -> Dict[str, Any]:
        """Train ML models on the analyzed data"""
        
        ml_results = {
            'model_accuracies': {},
            'feature_importance': {},
            'training_statistics': {}
        }
        
        stations_data = self.extracted_data.get('stations', [])
        
        if not stations_data or len(stations_data) < 10:
            self.logger.warning("⚠️ Insufficient data for ML training")
            return ml_results
        
        try:
            # Train demand prediction model
            if not focus or 'demand' in focus:
                self.logger.info("🤖 Training demand prediction model...")
                demand_accuracy = self._train_demand_model(stations_data, analysis_results)
                ml_results['model_accuracies']['demand_prediction'] = demand_accuracy
            
            # Train cost estimation model
            if not focus or 'cost' in focus:
                self.logger.info("🤖 Training cost estimation model...")
                cost_accuracy = self._train_cost_model(stations_data, analysis_results)
                ml_results['model_accuracies']['cost_estimation'] = cost_accuracy
            
            # Train station classification model
            if not focus or 'classification' in focus:
                self.logger.info("🤖 Training station classification model...")
                classification_accuracy = self._train_classification_model(stations_data)
                ml_results['model_accuracies']['station_classification'] = classification_accuracy
            
            # Get feature importance for trained models
            ml_results['feature_importance'] = self._extract_feature_importance()
            
        except Exception as e:
            self.logger.warning(f"⚠️ ML model training failed: {e}")
        
        return ml_results
    
    def _train_demand_model(self, stations_data: List[Dict], analysis_results: Dict[str, Any]) -> float:
        """Train passenger demand prediction model"""
        
        # Create training data from station characteristics
        training_data = []
        
        for station in stations_data:
            # Estimate demand based on station characteristics
            platforms = station.get('platforms', 1)
            estimated_demand = platforms * 2000  # Rough estimate: 2000 passengers per platform per day
            
            # Add noise based on station type
            if station.get('level') == 'intercity':
                estimated_demand *= 2.5
            elif station.get('level') == 'regional':
                estimated_demand *= 1.5
            
            # Create feature vector
            features = RailwayMLUtils.create_demand_features([station], [])
            if len(features) > 0 and len(features[0]) > 0:
                feature_dict = {
                    'population_1km': features[0][0],
                    'population_5km': features[0][1],
                    'commercial_density': features[0][2],
                    'transport_connections': features[0][4],
                    'employment_centers': features[0][6],
                    'daily_passengers': estimated_demand
                }
                training_data.append(feature_dict)
        
        if len(training_data) < 5:
            return 0.0
        
        # Train the model
        X, y = self.ml_pipeline.prepare_features(training_data, target_column='daily_passengers')
        performance = self.ml_pipeline.train_regressor(X, y, "demand_predictor", "random_forest")
        
        return performance.r2_score if performance.r2_score else 0.0
    
    def _train_cost_model(self, stations_data: List[Dict], analysis_results: Dict[str, Any]) -> float:
        """Train construction cost estimation model"""
        
        # Create training data from station and infrastructure characteristics
        training_data = []
        
        for station in stations_data[:20]:  # Limit for performance
            platforms = station.get('platforms', 1)
            
            # Estimate construction cost based on platforms and complexity
            base_cost = 5_000_000  # €5M base cost
            platform_cost = platforms * 1_000_000  # €1M per platform
            
            # Complexity multiplier based on station type
            if station.get('level') == 'intercity':
                complexity_multiplier = 2.5
            elif station.get('level') == 'regional':
                complexity_multiplier = 1.5
            else:
                complexity_multiplier = 1.0
            
            estimated_cost = (base_cost + platform_cost) * complexity_multiplier
            
            # Create synthetic route segment for cost training
            segment_data = {
                'length_km': 1.0,  # 1km segment
                'max_gradient_percent': 2.0,
                'avg_gradient_percent': 1.0,
                'elevation_change_m': 20,
                'min_curve_radius_m': 1000,
                'urban_percentage': 0.5,
                'water_crossings': 0,
                'protected_areas': 0,
                'soil_stability': 0.8,
                'seismic_risk': 0.1,
                'construction_cost': estimated_cost
            }
            training_data.append(segment_data)
        
        if len(training_data) < 5:
            return 0.0
        
        # Train the model
        X, y = self.ml_pipeline.prepare_features(training_data, target_column='construction_cost')
        performance = self.ml_pipeline.train_regressor(X, y, "cost_predictor", "random_forest")
        
        return performance.r2_score if performance.r2_score else 0.0
    
    def _train_classification_model(self, stations_data: List[Dict]) -> float:
        """Train station type classification model"""
        
        # Create training data from station characteristics
        training_data = []
        
        for station in stations_data:
            platforms = station.get('platforms', 1)
            station_type = station.get('level', 'regional')
            
            # Create feature vector
            feature_dict = {
                'platforms': platforms,
                'operator_type': 1 if station.get('operator') else 0,
                'services_count': len(station.get('services', [])),
                'station_type': station_type
            }
            training_data.append(feature_dict)
        
        if len(training_data) < 5:
            return 0.0
        
        # Train the model
        X, y = self.ml_pipeline.prepare_features(training_data, target_column='station_type')
        performance = self.ml_pipeline.train_classifier(X, y, "station_classifier", "random_forest")
        
        return performance.accuracy if performance.accuracy else 0.0
    
    def _extract_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Extract feature importance from trained models"""
        
        feature_importance = {}
        
        # Get feature importance for each trained model
        for model_name in self.ml_pipeline.models.keys():
            try:
                importance = self.ml_pipeline.get_feature_importance(model_name)
                if importance:
                    feature_importance[model_name] = {
                        feat.feature_name: feat.importance 
                        for feat in importance[:5]  # Top 5 features
                    }
            except Exception as e:
                self.logger.debug(f"Could not extract feature importance for {model_name}: {e}")
        
        return feature_importance
    
    def _assess_learning_quality(self, extraction_results: Dict[str, Any], 
                                analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """Assess the quality and completeness of the learning process"""
        
        quality_metrics = {
            'data_quality': 0.0,
            'coverage': 0.0,
            'confidence': 0.0
        }
        
        # Data quality assessment
        stations_count = extraction_results.get('stations_count', 0)
        tracks_count = extraction_results.get('tracks_count', 0)
        
        data_quality_score = 0.0
        
        # Station data quality
        if stations_count > 0:
            stations = extraction_results.get('stations', [])
            complete_stations = sum(1 for s in stations 
                                  if s.get('name') and s.get('lat') and s.get('lon'))
            station_completeness = complete_stations / stations_count
            data_quality_score += station_completeness * 0.5
        
        # Track data quality
        if tracks_count > 0:
            tracks = extraction_results.get('tracks', [])
            complete_tracks = sum(1 for t in tracks 
                                if t.get('maxspeed') and 'electrified' in t)
            track_completeness = complete_tracks / tracks_count
            data_quality_score += track_completeness * 0.5
        
        quality_metrics['data_quality'] = min(1.0, data_quality_score)
        
        # Coverage assessment
        network_analysis = extraction_results.get('network_analysis', {})
        topology = network_analysis.get('topology', {})
        
        coverage_score = 0.0
        
        # Network connectivity coverage
        if topology.get('connected_components', 1) == 1:
            coverage_score += 0.4  # Single connected component
        elif topology.get('connected_components', 0) <= 3:
            coverage_score += 0.2  # Few components
        
        # Station distribution coverage
        if stations_count >= 10:
            coverage_score += 0.3
        elif stations_count >= 5:
            coverage_score += 0.2
        
        # Track coverage
        if tracks_count >= 20:
            coverage_score += 0.3
        elif tracks_count >= 10:
            coverage_score += 0.2
        
        quality_metrics['coverage'] = min(1.0, coverage_score)
        
        # Learning confidence
        model_count = len(self.ml_pipeline.models)
        pattern_count = len(self.station_analyzer.learned_patterns) + len(self.track_intelligence.learned_patterns)
        
        confidence_score = 0.0
        
        # Model training success
        if model_count >= 3:
            confidence_score += 0.4
        elif model_count >= 1:
            confidence_score += 0.2
        
        # Pattern learning success
        if pattern_count >= 5:
            confidence_score += 0.3
        elif pattern_count >= 2:
            confidence_score += 0.15
        
        # Data sufficiency
        if stations_count >= 20 and tracks_count >= 50:
            confidence_score += 0.3
        elif stations_count >= 10 and tracks_count >= 20:
            confidence_score += 0.2
        
        quality_metrics['confidence'] = min(1.0, confidence_score)
        
        return quality_metrics
    
    def _generate_insights(self, results: LearningResults) -> Dict[str, List[str]]:
        """Generate key insights and recommendations from learning results"""
        
        insights = []
        recommendations = []
        
        # Data quality insights
        if results.data_quality_score > 0.8:
            insights.append(f"High quality data extracted from {self.country} railway network")
        elif results.data_quality_score < 0.5:
            insights.append(f"Limited data quality may affect learning accuracy")
            recommendations.append("Consider supplementing with additional data sources")
        
        # Network coverage insights
        if results.coverage_completeness > 0.7:
            insights.append(f"Comprehensive network coverage achieved")
        else:
            insights.append(f"Partial network coverage - some areas may lack representation")
            recommendations.append("Focus on key missing network segments for future learning")
        
        # Station pattern insights
        station_patterns = results.station_patterns
        if station_patterns.get('density_analysis'):
            avg_density = station_patterns['density_analysis'].get('avg_density', 0)
            if avg_density > 0.1:
                insights.append("High station density indicates urban-focused network")
            else:
                insights.append("Low station density suggests intercity-focused network")
        
        # Track infrastructure insights
        track_patterns = results.track_patterns
        if track_patterns.get('cost_analysis'):
            cost_data = track_patterns['cost_analysis'].get('cost_per_km_by_type', {})
            if 'tunnel' in cost_data:
                insights.append("Network includes significant tunneling infrastructure")
                recommendations.append("Apply tunnel cost models for mountainous terrain")
        
        # Operational insights
        operational_patterns = results.operational_patterns
        if operational_patterns.get('service_patterns'):
            speed_data = operational_patterns['service_patterns'].get('speed_vs_distance', [])
            if speed_data:
                high_speed_services = sum(1 for s in speed_data if s.get('speed', 0) > 200)
                if high_speed_services > 0:
                    insights.append("High-speed rail services identified in network")
                    recommendations.append("Consider high-speed infrastructure for long-distance routes")
        
        # Learning confidence insights
        if results.learning_confidence > 0.8:
            insights.append("High confidence in learned patterns - suitable for route generation")
        elif results.learning_confidence < 0.5:
            insights.append("Lower confidence in patterns - recommend additional learning")
            recommendations.append("Expand training data or focus on specific network segments")
        
        # Model performance insights
        if results.model_accuracies:
            best_model = max(results.model_accuracies.items(), key=lambda x: x[1])
            if best_model[1] > 0.8:
                insights.append(f"Strong ML model performance - {best_model[0]} accuracy: {best_model[1]:.2f}")
            
            worst_model = min(results.model_accuracies.items(), key=lambda x: x[1])
            if worst_model[1] < 0.6:
                recommendations.append(f"Improve {worst_model[0]} model with additional training data")
        
        # Country-specific insights
        if self.country in ['DE', 'FR', 'CH']:
            insights.append("Learning from advanced European railway network with high standards")
            recommendations.append("Apply strict engineering standards for gradient and curve compliance")
        
        # Feature importance insights
        if results.feature_importance:
            for model_name, features in results.feature_importance.items():
                if features:
                    top_feature = max(features.items(), key=lambda x: x[1])
                    insights.append(f"Key factor for {model_name}: {top_feature[0]} (importance: {top_feature[1]:.3f})")
        
        return {
            'insights': insights[:10],  # Limit to top 10 insights
            'recommendations': recommendations[:8]  # Limit to top 8 recommendations
        }
    
    def save_models(self, directory: str):
        """Save all trained models and learned patterns"""
        
        models_dir = Path(directory)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save ML pipeline models
            self.ml_pipeline.save_models(str(models_dir))
            
            # Save learned patterns
            patterns_file = models_dir / "learned_patterns.json"
            patterns_data = {
                'station_patterns': self.station_analyzer.learned_patterns,
                'track_patterns': self.track_intelligence.learned_patterns,
                'train_patterns': self.train_classifier.learned_patterns,
                'country': self.country,
                'train_types': self.train_types,
                'learning_date': datetime.now().isoformat()
            }
            
            with open(patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2, default=str)
            
            # Save extracted data cache
            cache_file = models_dir / f"extracted_data_{self.country.lower()}.json"
            with open(cache_file, 'w') as f:
                json.dump(self.extracted_data, f, indent=2, default=str)
            
            self.logger.info(f"💾 Models and patterns saved to {models_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save models: {e}")
            raise
    
    def load_models(self, directory: str):
        """Load previously trained models and patterns"""
        
        models_dir = Path(directory)
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {directory}")
        
        try:
            # Load ML pipeline models
            self.ml_pipeline.load_models(str(models_dir))
            
            # Load learned patterns
            patterns_file = models_dir / "learned_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                
                self.station_analyzer.learned_patterns = patterns_data.get('station_patterns', {})
                self.track_intelligence.learned_patterns = patterns_data.get('track_patterns', {})
                self.train_classifier.learned_patterns = patterns_data.get('train_patterns', {})
            
            # Load extracted data cache
            cache_file = models_dir / f"extracted_data_{self.country.lower()}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    self.extracted_data = json.load(f)
            
            self.logger.info(f"📚 Models and patterns loaded from {models_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load models: {e}")
            raise
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of current learning state"""
        
        return {
            'country': self.country,
            'train_types': self.train_types,
            'models_trained': list(self.ml_pipeline.models.keys()),
            'patterns_learned': {
                'station_patterns': len(self.station_analyzer.learned_patterns),
                'track_patterns': len(self.track_intelligence.learned_patterns),
                'train_patterns': len(self.train_classifier.learned_patterns)
            },
            'data_extracted': {
                'stations': len(self.extracted_data.get('stations', [])),
                'tracks': len(self.extracted_data.get('tracks', [])),
                'has_network': bool(self.extracted_data.get('network_graph'))
            }
        }

# Example usage and testing
if __name__ == "__main__":
    from Model.config import create_development_config
    
    config = create_development_config()
    
    print("🧠 Testing Railway Learning System...")
    
    # Initialize learner for German railway network
    learner = RailwayLearner(
        country="DE",
        train_types=["ICE", "IC", "RE", "S"],
        config=config
    )
    
    try:
        # Execute learning process
        results = learner.execute(
            focus=["stations", "tracks", "services"],
            data_sources="osm,terrain"
        )
        
        print(f"\n✅ Learning Results for {results.country}")
        print(f"⏱️  Learning Time: {results.learning_time:.1f} seconds")
        print(f"📊 Data Quality: {results.data_quality_score:.1%}")
        print(f"📡 Coverage: {results.coverage_completeness:.1%}")
        print(f"🎯 Confidence: {results.learning_confidence:.1%}")
        
        print(f"\n📈 Infrastructure Analyzed:")
        print(f"  🚉 Stations: {results.stations_analyzed}")
        print(f"  🛤️  Track Segments: {results.track_segments}")
        print(f"  🏭 Railyards: {results.railyards_found}")
        
        if results.model_accuracies:
            print(f"\n🤖 Model Performance:")
            for model, accuracy in results.model_accuracies.items():
                print(f"  • {model}: {accuracy:.3f}")
        
        if results.key_insights:
            print(f"\n💡 Key Insights:")
            for insight in results.key_insights[:5]:
                print(f"  • {insight}")
        
        if results.recommendations:
            print(f"\n🎯 Recommendations:")
            for rec in results.recommendations[:3]:
                print(f"  • {rec}")
        
        # Save learned models
        learner.save_models("data/models")
        print(f"\n💾 Models saved to data/models/")
        
        # Get learning summary
        summary = learner.get_learning_summary()
        print(f"\n📋 Learning Summary:")
        print(f"  Models trained: {len(summary['models_trained'])}")
        print(f"  Patterns learned: {sum(summary['patterns_learned'].values())}")
        print(f"  Data points: {summary['data_extracted']['stations']} stations, {summary['data_extracted']['tracks']} tracks")
        
    except Exception as e:
        print(f"❌ Learning failed: {e}")
        import traceback
        traceback.print_exc()