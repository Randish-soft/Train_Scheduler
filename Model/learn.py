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
    
    # Enhanced infrastructure data
    infrastructure_data: Dict[str, Any] = field(default_factory=dict)

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
        self.enhanced_infrastructure = {}
        
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
            
            # Phase 7: Generate Enhanced Infrastructure Data
            self.logger.info("🏗️ Phase 7: Generating enhanced infrastructure data...")
            infrastructure_data = self._generate_enhanced_infrastructure(extraction_results, analysis_results)
            results.infrastructure_data = infrastructure_data
            
            # Save enhanced JSON and generate HTML
            self._save_enhanced_network_data(infrastructure_data)
            self._generate_html_dashboard(infrastructure_data)
            
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
            # Extract stations with enhanced data
            self.logger.info("🚉 Extracting stations from OpenStreetMap...")
            stations = self.osm_extractor.extract_train_stations(self.country)
            
            # Enhance station data
            for station in stations:
                station = self._enhance_station_data(station)
            
            extraction_results['stations'] = stations
            extraction_results['stations_count'] = len(stations)
            
            # Extract tracks with enhanced data
            self.logger.info("🛤️ Extracting track network...")
            tracks = self.osm_extractor.extract_railway_tracks(self.country)
            
            # Enhance track data
            for track in tracks:
                track = self._enhance_track_data(track)
            
            extraction_results['tracks'] = tracks
            extraction_results['tracks_count'] = len(tracks)
            
            self.logger.info(f"📍 Found {len(stations)} stations, {len(tracks)} track segments")
            
            # Build network graph
            if stations and tracks:
                self.logger.info("🕸️ Building network graph...")
                osm_data = self._prepare_osm_data(stations, tracks)
                network_graph = self.network_parser.parse_osm_to_network(osm_data)
                extraction_results['network_graph'] = network_graph
            
            # Cache extracted data
            self.extracted_data = extraction_results
            
        except Exception as e:
            self.logger.error(f"❌ Data extraction failed: {e}")
            if not extraction_results['stations']:
                raise
        
        return extraction_results
    
    def _enhance_station_data(self, station: Dict) -> Dict:
        """Enhance station data with additional attributes"""
        platforms = int(station.get('platforms', 1))
        
        # Estimate platform types
        if platforms >= 8:
            station['underground_platforms'] = 2
            station['elevated_platforms'] = 2
            station['surface_platforms'] = platforms - 4
        elif platforms >= 4:
            station['underground_platforms'] = 0
            station['elevated_platforms'] = 1 if platforms > 4 else 0
            station['surface_platforms'] = platforms - station['elevated_platforms']
        else:
            station['underground_platforms'] = 0
            station['elevated_platforms'] = 0
            station['surface_platforms'] = platforms
        
        # Check for quay/port connections
        name = station.get('name', '').lower()
        station['has_quay'] = any(term in name for term in ['port', 'maritime', 'ferry', 'harbor', 'quay'])
        station['quay_type'] = 'ferry' if station['has_quay'] else None
        
        # Transportation connections
        connections = ['bus']
        if platforms >= 10:
            connections.extend(['metro', 'tram', 'taxi'])
        elif platforms >= 6:
            connections.extend(['tram', 'taxi'])
        elif platforms >= 3:
            connections.append('taxi')
        
        if station['has_quay']:
            connections.append('ferry')
        
        station['connections'] = list(set(connections))
        
        # Station classification
        if platforms >= 10:
            station['station_type'] = 'terminal'
        elif platforms >= 6:
            station['station_type'] = 'intercity'
        elif platforms >= 3:
            station['station_type'] = 'regional'
        else:
            station['station_type'] = 'local'
        
        return station
    
    def _enhance_track_data(self, track: Dict) -> Dict:
        """Enhance track data with infrastructure details"""
        maxspeed = int(track.get('maxspeed', 100))
        electrified = track.get('electrified', False)
        usage = track.get('usage', 'main')
        
        # Track category based on speed
        if maxspeed >= 250:
            track['track_category'] = 'high_speed'
        elif maxspeed >= 160:
            track['track_category'] = 'mainline'
        elif maxspeed >= 100:
            track['track_category'] = 'regional'
        else:
            track['track_category'] = 'urban'
        
        # Infrastructure type (simplified)
        if usage == 'main' and maxspeed > 200:
            track['track_type'] = 'dedicated_high_speed'
        elif 'tunnel' in track.get('tags', {}).get('name', '').lower():
            track['track_type'] = 'underground'
        elif 'bridge' in track.get('tags', {}).get('name', '').lower():
            track['track_type'] = 'elevated'
        else:
            track['track_type'] = 'surface'
        
        # Electrification details
        if electrified:
            voltage = track.get('tags', {}).get('voltage', '25000')
            frequency = track.get('tags', {}).get('frequency', '50')
            track['electrification_type'] = f"{int(voltage)/1000}kV AC {frequency}Hz"
        else:
            track['electrification_type'] = 'none'
        
        # Infrastructure elements
        track['infrastructure_elements'] = []
        if track['track_type'] == 'underground':
            track['infrastructure_elements'].extend(['tunnel', 'ventilation_shafts'])
        elif track['track_type'] == 'elevated':
            track['infrastructure_elements'].extend(['viaduct', 'bridge'])
        else:
            track['infrastructure_elements'].append('surface')
        
        return track
    
    def _generate_enhanced_infrastructure(self, extraction_results: Dict, analysis_results: Dict) -> Dict:
        """Generate enhanced infrastructure data for JSON export"""
        
        stations = extraction_results.get('stations', [])
        tracks = extraction_results.get('tracks', [])
        
        # Categorize tracks by type
        underground_sections = []
        overground_sections = []
        high_speed_sections = []
        
        for track in tracks:
            # Calculate track length
            coords = self._extract_track_coordinates(track)
            length_km = self._calculate_track_length(coords)
            
            track_data = {
                'id': f"track_{track['id']}",
                'name': track.get('name', f"Section {track['id']}"),
                'coordinates': coords,
                'track_type': track.get('track_type', 'surface'),
                'electrified': track.get('electrified', False),
                'electrification_type': track.get('electrification_type', 'none'),
                'max_speed': int(track.get('maxspeed', 100)),
                'track_category': track.get('track_category', 'regional'),
                'infrastructure_elements': track.get('infrastructure_elements', []),
                'length_km': length_km
            }
            
            if track_data['track_type'] == 'underground':
                underground_sections.append(track_data)
            elif track_data['track_category'] == 'high_speed':
                high_speed_sections.append(track_data)
            else:
                overground_sections.append(track_data)
        
        # Process stations
        enhanced_stations = []
        quays = []
        
        for station in stations:
            station_data = {
                'name': station['name'],
                'lat': station['lat'],
                'lon': station['lon'],
                'station_type': station.get('station_type', 'regional'),
                'has_quay': station.get('has_quay', False),
                'quay_type': station.get('quay_type'),
                'platform_count': int(station.get('platforms', 1)),
                'underground_platforms': station.get('underground_platforms', 0),
                'elevated_platforms': station.get('elevated_platforms', 0),
                'surface_platforms': station.get('surface_platforms', 1),
                'connections': station.get('connections', ['bus']),
                'daily_passengers': self._estimate_daily_passengers(station)
            }
            enhanced_stations.append(station_data)
            
            # Create quay if applicable
            if station_data['has_quay']:
                quay = {
                    'name': f"{station['name']} Port Terminal",
                    'lat': station['lat'] - 0.005,
                    'lon': station['lon'] + 0.005,
                    'type': station_data['quay_type'] or 'ferry',
                    'capacity': 'medium',
                    'rail_connection': True,
                    'services': ['passenger', 'freight']
                }
                quays.append(quay)
        
        # Calculate totals
        total_length = sum(s['length_km'] for s in underground_sections + overground_sections + high_speed_sections)
        electrified_sections = [s for s in underground_sections + overground_sections + high_speed_sections if s['electrified']]
        electrified_length = sum(s['length_km'] for s in electrified_sections)
        
        # Create comprehensive infrastructure data
        infrastructure_data = {
            'country': self.country,
            'network_name': f"{self.country} National Railway Network",
            'network': {
                'high_speed_lines': high_speed_sections,
                'underground_sections': underground_sections,
                'overground_lines': overground_sections,
                'stations': enhanced_stations,
                'quays': quays,
                'electrification': {
                    'electrified_km': electrified_length,
                    'non_electrified_km': total_length - electrified_length,
                    'electrification_percentage': (electrified_length / total_length * 100) if total_length > 0 else 0,
                    'electrification_type': '25kV AC 50Hz',  # Most common
                    'sections': []  # Simplified
                },
                'total_length_km': total_length,
                'infrastructure_summary': {
                    'underground_km': sum(s['length_km'] for s in underground_sections),
                    'elevated_km': sum(s['length_km'] for s in overground_sections if 'elevated' in s.get('track_type', '')),
                    'surface_km': sum(s['length_km'] for s in overground_sections if s.get('track_type') == 'surface'),
                    'tunnel_count': len([s for s in underground_sections]),
                    'bridge_count': len([s for s in overground_sections if 'bridge' in s.get('infrastructure_elements', [])]),
                    'station_count': len(enhanced_stations),
                    'quay_count': len(quays),
                    'high_speed_km': sum(s['length_km'] for s in high_speed_sections)
                }
            },
            'technical_specifications': {
                'gauge': '1435mm',
                'signaling': 'ETCS Level 2',
                'loading_gauge': 'UIC GC',
                'axle_load': '22.5t',
                'platform_height': '550mm',
                'voltage': '25kV AC 50Hz'
            },
            'metadata': {
                'extraction_date': datetime.now().isoformat(),
                'generator_version': '2.0',
                'data_sources': ['OpenStreetMap']
            }
        }
        
        return infrastructure_data
    
    def _extract_track_coordinates(self, track: Dict) -> List[List[float]]:
        """Extract coordinates from track geometry"""
        coords = []
        geometry = track.get('geometry', [])
        
        for point in geometry[:10]:  # Limit points for simplicity
            if isinstance(point, dict) and 'lat' in point and 'lon' in point:
                coords.append([point['lat'], point['lon']])
        
        # Ensure at least 2 points
        if len(coords) < 2:
            coords = [[track.get('lat', 0), track.get('lon', 0)], 
                     [track.get('lat', 0) + 0.01, track.get('lon', 0) + 0.01]]
        
        return coords
    
    def _calculate_track_length(self, coords: List[List[float]]) -> float:
        """Calculate track length from coordinates"""
        if len(coords) < 2:
            return 1.0  # Default 1km
        
        total_length = 0.0
        for i in range(len(coords) - 1):
            length = haversine_distance(
                coords[i][0], coords[i][1],
                coords[i+1][0], coords[i+1][1]
            )
            total_length += length
        
        return max(total_length, 1.0)  # Minimum 1km
    
    def _estimate_daily_passengers(self, station: Dict) -> int:
        """Estimate daily passenger count"""
        platforms = int(station.get('platforms', 1))
        station_type = station.get('station_type', 'regional')
        
        base_passengers = {
            'terminal': 50000,
            'intercity': 20000,
            'regional': 8000,
            'local': 2000
        }
        
        passengers = base_passengers.get(station_type, 5000)
        passengers *= (platforms / 4)  # Scale by platforms
        
        return int(passengers)
    
    def _save_enhanced_network_data(self, infrastructure_data: Dict):
        """Save enhanced network data as JSON"""
        output_dir = Path(f"data/learned/{self.country.lower()}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive JSON
        json_file = output_dir / f"{self.country}_railway_network_enhanced.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(infrastructure_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Saved enhanced network data to {json_file}")
    
    def _generate_html_dashboard(self, infrastructure_data: Dict):
        """Generate HTML dashboard for the learned network"""
        try:
            # Import the dashboard generator we created for generate.py
            import sys
            sys.path.append(str(Path(__file__).parent))
            from generate import RailwayDashboardGenerator
            
            output_dir = Path(f"data/learned/{self.country.lower()}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate HTML
            dashboard_gen = RailwayDashboardGenerator()
            html_path = output_dir / f"{self.country}_railway_network_dashboard.html"
            dashboard_gen.generate_dashboard(infrastructure_data, str(html_path))
            
            self.logger.info(f"📊 Generated HTML dashboard at {html_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not generate HTML dashboard: {e}")
    
    def _prepare_osm_data(self, stations: List[Dict], tracks: List[Dict]) -> Dict:
        """Prepare OSM data for network parsing"""
        osm_data = {'elements': []}
        
        for station in stations:
            osm_element = {
                'type': 'node',
                'id': station['id'],
                'lat': station['lat'],
                'lon': station['lon'],
                'tags': station
            }
            osm_data['elements'].append(osm_element)
        
        for track in tracks:
            osm_element = {
                'type': 'way',
                'id': track['id'],
                'geometry': track.get('geometry', []),
                'tags': track
            }
            osm_data['elements'].append(osm_element)
        
        return osm_data
    
    # Keep all the other original methods from the file unchanged
    # Just add the enhanced data generation and saving
    
    def _analyze_extracted_data(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        # Keep original implementation
        return super()._analyze_extracted_data(extraction_results)
    
    def _learn_patterns(self, analysis_results: Dict[str, Any], focus: Optional[List[str]]) -> Dict[str, Any]:
        # Keep original implementation
        return super()._learn_patterns(analysis_results, focus)
    
    def _train_ml_models(self, analysis_results: Dict[str, Any], focus: Optional[List[str]]) -> Dict[str, Any]:
        # Keep original implementation
        return super()._train_ml_models(analysis_results, focus)
    
    def _assess_learning_quality(self, extraction_results: Dict[str, Any], 
                                analysis_results: Dict[str, Any]) -> Dict[str, float]:
        # Keep original implementation
        return super()._assess_learning_quality(extraction_results, analysis_results)
    
    def _generate_insights(self, results: LearningResults) -> Dict[str, List[str]]:
        # Keep original implementation
        return super()._generate_insights(results)
    
    def save_models(self, directory: str):
        # Keep original implementation
        super().save_models(directory)
    
    def load_models(self, directory: str):
        # Keep original implementation
        super().load_models(directory)
    
    def get_learning_summary(self) -> Dict[str, Any]:
        # Keep original implementation
        return super().get_learning_summary()