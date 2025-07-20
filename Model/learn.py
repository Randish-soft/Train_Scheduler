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
            'operational_data': [],
            'infrastructure_elements': {
                'tunnels': [],
                'bridges': [],
                'level_crossings': [],
                'electrification': {}
            }
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
            
            # Enhance station data
            enhanced_stations = []
            for station in stations:
                enhanced_station = self._enhance_station_data(station)
                enhanced_stations.append(enhanced_station)
            
            extraction_results['stations'] = enhanced_stations
            extraction_results['stations_count'] = len(enhanced_stations)
            
            self.logger.info(f"📍 Found {len(enhanced_stations)} stations")
            
            # Extract track network with infrastructure details
            self.logger.info("🛤️ Extracting track network...")
            tracks = self.osm_extractor.extract_railway_tracks(self.country)
            
            # Enhance track data
            enhanced_tracks = []
            for track in tracks:
                enhanced_track = self._enhance_track_data(track)
                enhanced_tracks.append(enhanced_track)
            
            extraction_results['tracks'] = enhanced_tracks
            extraction_results['tracks_count'] = len(enhanced_tracks)
            
            self.logger.info(f"🚂 Found {len(enhanced_tracks)} track segments")
            
            # Extract infrastructure elements
            self.logger.info("🏗️ Extracting infrastructure elements...")
            infrastructure = self._extract_infrastructure_elements(country_bounds)
            extraction_results['infrastructure_elements'] = infrastructure
            
            # Build network graph
            if enhanced_stations and enhanced_tracks:
                self.logger.info("🕸️ Building network graph...")
                osm_data = self._prepare_osm_data(enhanced_stations, enhanced_tracks)
                network_graph = self.network_parser.parse_osm_to_network(osm_data)
                extraction_results['network_graph'] = network_graph
                
                # Analyze network patterns
                network_analysis = self.network_parser.analyze_network_patterns()
                extraction_results['network_analysis'] = network_analysis
                
                self.logger.info(f"🔗 Built network with {network_graph.number_of_nodes()} nodes, "
                               f"{network_graph.number_of_edges()} edges")
            
            # Extract terrain data for key routes
            if enhanced_stations and len(enhanced_stations) >= 2:
                self.logger.info("🏔️ Analyzing terrain along major routes...")
                terrain_data = self._extract_terrain_for_routes(enhanced_stations[:10])
                extraction_results['terrain_data'] = terrain_data
            
            # Cache extracted data
            self.extracted_data = extraction_results
            
        except Exception as e:
            self.logger.error(f"❌ Data extraction failed: {e}")
            # Continue with partial data if possible
            if not extraction_results['stations']:
                raise
        
        return extraction_results
    
    def _enhance_station_data(self, station: Dict) -> Dict:
        """Enhance station data with additional attributes"""
        enhanced = station.copy()
        
        # Determine station infrastructure
        platforms = int(station.get('platforms', 1))
        
        # Estimate platform types
        if platforms >= 8:
            enhanced['underground_platforms'] = 2
            enhanced['elevated_platforms'] = 2
            enhanced['surface_platforms'] = platforms - 4
        elif platforms >= 4:
            enhanced['underground_platforms'] = 0
            enhanced['elevated_platforms'] = 1 if platforms > 4 else 0
            enhanced['surface_platforms'] = platforms - enhanced['elevated_platforms']
        else:
            enhanced['underground_platforms'] = 0
            enhanced['elevated_platforms'] = 0
            enhanced['surface_platforms'] = platforms
        
        # Check for quay/port connections
        enhanced['has_quay'] = self._check_for_quay(station)
        enhanced['quay_type'] = 'ferry' if enhanced['has_quay'] else None
        
        # Transportation connections
        enhanced['connections'] = self._get_station_connections(station)
        
        # Station classification
        if platforms >= 10:
            enhanced['station_type'] = 'terminal'
        elif platforms >= 6:
            enhanced['station_type'] = 'intercity'
        elif platforms >= 3:
            enhanced['station_type'] = 'regional'
        else:
            enhanced['station_type'] = 'local'
        
        return enhanced
    
    def _enhance_track_data(self, track: Dict) -> Dict:
        """Enhance track data with infrastructure details"""
        enhanced = track.copy()
        
        # Determine track type and category
        maxspeed = int(track.get('maxspeed', 100))
        electrified = track.get('electrified', False)
        usage = track.get('usage', 'main')
        
        # Track category based on speed
        if maxspeed >= 250:
            enhanced['track_category'] = 'high_speed'
        elif maxspeed >= 160:
            enhanced['track_category'] = 'mainline'
        elif maxspeed >= 100:
            enhanced['track_category'] = 'regional'
        else:
            enhanced['track_category'] = 'urban'
        
        # Infrastructure type (simplified - would need elevation data for accuracy)
        if usage == 'main' and maxspeed > 200:
            enhanced['track_type'] = 'dedicated_high_speed'
        elif 'tunnel' in track.get('tags', {}).get('name', '').lower():
            enhanced['track_type'] = 'underground'
        elif 'bridge' in track.get('tags', {}).get('name', '').lower():
            enhanced['track_type'] = 'elevated'
        else:
            enhanced['track_type'] = 'surface'
        
        # Electrification details
        if electrified:
            voltage = track.get('tags', {}).get('voltage', '25000')
            frequency = track.get('tags', {}).get('frequency', '50')
            enhanced['electrification_type'] = f"{int(voltage)/1000}kV AC {frequency}Hz"
        else:
            enhanced['electrification_type'] = 'none'
        
        # Infrastructure elements
        enhanced['infrastructure_elements'] = []
        if enhanced['track_type'] == 'underground':
            enhanced['infrastructure_elements'].extend(['tunnel', 'ventilation_shafts'])
        elif enhanced['track_type'] == 'elevated':
            enhanced['infrastructure_elements'].extend(['viaduct', 'bridge'])
        else:
            enhanced['infrastructure_elements'].append('surface')
        
        return enhanced
    
    def _extract_infrastructure_elements(self, bbox: Tuple) -> Dict:
        """Extract detailed infrastructure elements"""
        infrastructure = {
            'tunnels': [],
            'bridges': [],
            'level_crossings': [],
            'quays': [],
            'freight_terminals': [],
            'electrification_sections': []
        }
        
        # Query for tunnels
        try:
            tunnel_query = f"""
            [out:json][timeout:180];
            (
              way["tunnel"="yes"]["railway"]({{bbox[1]}},{{bbox[0]}},{{bbox[3]}},{{bbox[2]}});
              way["railway"="tunnel"]({{bbox[1]}},{{bbox[0]}},{{bbox[3]}},{{bbox[2]}});
            );
            out geom;
            """
            # Note: In production, execute this query via OSM API
            self.logger.info("🚇 Extracting tunnel data...")
        except Exception as e:
            self.logger.warning(f"Could not extract tunnel data: {e}")
        
        # Query for bridges
        try:
            bridge_query = f"""
            [out:json][timeout:180];
            (
              way["bridge"="yes"]["railway"]({{bbox[1]}},{{bbox[0]}},{{bbox[3]}},{{bbox[2]}});
              way["railway"="bridge"]({{bbox[1]}},{{bbox[0]}},{{bbox[3]}},{{bbox[2]}});
            );
            out geom;
            """
            # Note: In production, execute this query via OSM API
            self.logger.info("🌉 Extracting bridge data...")
        except Exception as e:
            self.logger.warning(f"Could not extract bridge data: {e}")
        
        return infrastructure
    
    def _check_for_quay(self, station: Dict) -> bool:
        """Check if station has quay/port connection"""
        # Simple check based on name or tags
        name = station.get('name', '').lower()
        coastal_indicators = ['port', 'maritime', 'ferry', 'harbor', 'quay', 'pier']
        
        return any(indicator in name for indicator in coastal_indicators)
    
    def _get_station_connections(self, station: Dict) -> List[str]:
        """Get transportation connections for station"""
        connections = ['bus']  # Default
        
        platforms = int(station.get('platforms', 1))
        name = station.get('name', '').lower()
        
        # Based on station size
        if platforms >= 10:
            connections.extend(['metro', 'tram', 'taxi'])
        elif platforms >= 6:
            connections.extend(['tram', 'taxi'])
        elif platforms >= 3:
            connections.append('taxi')
        
        # Special connections
        if 'airport' in name or 'flughafen' in name:
            connections.append('airport')
        if self._check_for_quay(station):
            connections.append('ferry')
        if 'hauptbahnhof' in name or 'central' in name:
            if 'metro' not in connections:
                connections.append('metro')
        
        return list(set(connections))  # Remove duplicates
    
    def _prepare_osm_data(self, stations: List[Dict], tracks: List[Dict]) -> Dict:
        """Prepare OSM data for network parsing"""
        osm_data = {'elements': []}
        
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
                    'platforms': str(station.get('platforms', 1)),
                    'station_type': station.get('station_type', 'regional'),
                    'has_quay': 'yes' if station.get('has_quay') else 'no'
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
                    'usage': track.get('usage', 'main'),
                    'track_type': track.get('track_type', 'surface'),
                    'track_category': track.get('track_category', 'regional')
                }
            }
            osm_data['elements'].append(osm_element)
        
        return osm_data
    
    def _generate_enhanced_infrastructure(self, extraction_results: Dict, analysis_results: Dict) -> Dict:
        """Generate enhanced infrastructure data for JSON export"""
        
        stations = extraction_results.get('stations', [])
        tracks = extraction_results.get('tracks', [])
        infrastructure = extraction_results.get('infrastructure_elements', {})
        
        # Categorize tracks by type
        underground_sections = []
        overground_sections = []
        high_speed_sections = []
        
        for track in tracks:
            track_data = {
                'id': f"track_{track['id']}",
                'name': track.get('name', f"Section {track['id']}"),
                'coordinates': self._extract_track_coordinates(track),
                'track_type': track.get('track_type', 'surface'),
                'electrified': track.get('electrified', False),
                'electrification_type': track.get('electrification_type', 'none'),
                'max_speed': int(track.get('maxspeed', 100)),
                'track_category': track.get('track_category', 'regional'),
                'infrastructure_elements': track.get('infrastructure_elements', []),
                'length_km': self._calculate_track_length(track)
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
        
        # Calculate electrification statistics
        total_length = sum(self._calculate_track_length(t) for t in tracks)
        electrified_length = sum(self._calculate_track_length(t) for t in tracks if t.get('electrified'))
        
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
                    'electrification_type': self._get_primary_electrification_type(tracks),
                    'sections': self._get_electrification_sections(tracks)
                },
                'total_length_km': total_length,
                'infrastructure_summary': {
                    'underground_km': sum(t['length_km'] for t in underground_sections),
                    'elevated_km': sum(t['length_km'] for t in overground_sections if 'elevated' in t.get('track_type', '')),
                    'surface_km': sum(t['length_km'] for t in overground_sections if t.get('track_type') == 'surface'),
                    'tunnel_count': len([t for t in tracks if 'tunnel' in t.get('infrastructure_elements', [])]),
                    'bridge_count': len([t for t in tracks if 'bridge' in t.get('infrastructure_elements', [])]),
                    'station_count': len(enhanced_stations),
                    'quay_count': len(quays),
                    'high_speed_km': sum(t['length_km'] for t in high_speed_sections)
                }
            },
            'technical_specifications': {
                'gauge': self._get_primary_gauge(tracks),
                'signaling': self._infer_signaling_system(),
                'loading_gauge': self._infer_loading_gauge(),
                'axle_load': '22.5t',  # Standard
                'platform_height': self._get_platform_heights(),
                'voltage': self._get_voltage_systems(tracks)
            },
            'operational_data': {
                'train_types': self.train_types,
                'max_speed': max(int(t.get('maxspeed', 100)) for t in tracks) if tracks else 160,
                'network_type': self._classify_network_type(analysis_results)
            },
            'metadata': {
                'extraction_date': datetime.now().isoformat(),
                'data_quality_score': self.extracted_data.get('data_quality_score', 0.0),
                'coverage_completeness': self.extracted_data.get('coverage_completeness', 0.0),
                'generator_version': '2.0',
                'data_sources': ['OpenStreetMap', 'Terrain Analysis', 'ML Models']
            }
        }
        
        return infrastructure_data
    
    def _extract_track_coordinates(self, track: Dict) -> List[List[float]]:
        """Extract coordinates from track geometry"""
        coords = []
        geometry = track.get('geometry', [])
        
        for point in geometry:
            if isinstance(point, dict) and 'lat' in point and 'lon' in point:
                coords.append([point['lat'], point['lon']])
        
        # If no detailed geometry, create simple line
        if not coords and len(geometry) >= 2:
            coords = [[geometry[0], geometry[1]], [geometry[-2], geometry[-1]]]
        
        return coords
    
    def _calculate_track_length(self, track: Dict) -> float:
        """Calculate track length from geometry"""
        coords = self._extract_track_coordinates(track)
        if len(coords) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(coords) - 1):
            length = haversine_distance(
                coords[i][0], coords[i][1],
                coords[i+1][0], coords[i+1][1]
            )
            total_length += length
        
        return total_length
    
    def _estimate_daily_passengers(self, station: Dict) -> int:
        """Estimate daily passenger count based on station characteristics"""
        platforms = int(station.get('platforms', 1))
        station_type = station.get('station_type', 'regional')
        
        # Base estimation
        base_passengers = {
            'terminal': 50000,
            'intercity': 20000,
            'regional': 8000,
            'local': 2000
        }
        
        passengers = base_passengers.get(station_type, 5000)
        
        # Adjust for platform count
        passengers *= (platforms / 4)  # Normalize to 4 platforms
        
        # Adjust for connections
        connections = len(station.get('connections', []))
        passengers *= (1 + connections * 0.1)
        
        return int(passengers)
    
    def _get_primary_electrification_type(self, tracks: List[Dict]) -> str:
        """Get the most common electrification type"""
        types = {}
        for track in tracks:
            if track.get('electrified'):
                e_type = track.get('electrification_type', '25kV AC 50Hz')
                types[e_type] = types.get(e_type, 0) + 1
        
        if types:
            return max(types.items(), key=lambda x: x[1])[0]
        return '25kV AC 50Hz'  # Default
    
    def _get_electrification_sections(self, tracks: List[Dict]) -> List[Dict]:
        """Get detailed electrification by section"""
        sections = []
        for track in tracks[:20]:  # Limit for performance
            section = {
                'section_id': track['id'],
                'electrified': track.get('electrified', False),
                'type': track.get('electrification_type', 'none'),
                'length_km': self._calculate_track_length(track)
            }
            sections.append(section)
        return sections
    
    def _get_primary_gauge(self, tracks: List[Dict]) -> str:
        """Get the most common track gauge"""
        gauges = {}
        for track in tracks:
            gauge = track.get('gauge', '1435')
            gauges[gauge] = gauges.get(gauge, 0) + 1
        
        if gauges:
            primary_gauge = max(gauges.items(), key=lambda x: x[1])[0]
            return f"{primary_gauge}mm"
        return "1435mm"  # Standard gauge
    
    def _infer_signaling_system(self) -> str:
        """Infer signaling system based on country"""
        signaling_systems = {
            'DE': 'ETCS Level 2 / PZB',
            'FR': 'ETCS Level 2 / TVM',
            'CH': 'ETCS Level 2 / SIGNUM',
            'IT': 'ETCS Level 2 / SCMT',
            'ES': 'ETCS Level 2 / ASFA',
            'NL': 'ETCS Level 2 / ATB',
            'BE': 'ETCS Level 2 / TBL'
        }
        return signaling_systems.get(self.country, 'ETCS Level 1')
    
    def _infer_loading_gauge(self) -> str:
        """Infer loading gauge based on country"""
        if self.country in ['DE', 'FR', 'CH', 'AT', 'NL', 'BE']:
            return 'UIC GC'
        elif self.country == 'GB':
            return 'W12'
        else:
            return 'UIC GB'
    
    def _get_platform_heights(self) -> str:
        """Get standard platform heights for country"""
        platform_heights = {
            'DE': '760mm / 550mm',
            'FR': '550mm / 920mm',
            'CH': '550mm / 350mm',
            'IT': '550mm',
            'ES': '680mm / 550mm',
            'NL': '760mm',
            'BE': '760mm / 550mm'
        }
        return platform_heights.get(self.country, '550mm')
    
    def _get_voltage_systems(self, tracks: List[Dict]) -> str:
        """Get voltage systems used"""
        voltages = set()
        for track in tracks:
            if track.get('electrified'):
                voltage = track.get('electrification_type', '25kV AC 50Hz')
                voltages.add(voltage)
        
        if voltages:
            return ' / '.join(sorted(voltages))
        return '25kV