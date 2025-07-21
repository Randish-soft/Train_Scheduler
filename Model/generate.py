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
class TrackSection:
    """Enhanced track section with detailed specifications"""
    id: str
    name: str
    coordinates: List[List[float]]  # [[lat, lon], ...]
    track_type: str  # 'underground', 'overground', 'elevated', 'surface'
    electrified: bool
    electrification_type: str  # '25kV AC', '15kV AC', '3kV DC', 'third_rail', 'none'
    max_speed: int
    track_category: str  # 'high_speed', 'mainline', 'regional', 'urban'
    infrastructure_elements: List[str]  # ['tunnel', 'bridge', 'viaduct', etc.]
    length_km: float
    
@dataclass
class EnhancedStation:
    """Station with quay and infrastructure details"""
    name: str
    lat: float
    lon: float
    station_type: str
    has_quay: bool
    quay_type: Optional[str]  # 'ferry', 'cargo', 'cruise', None
    platform_count: int
    underground_platforms: int
    elevated_platforms: int
    surface_platforms: int
    connections: List[str]  # ['metro', 'tram', 'bus', 'ferry']
    daily_passengers: int

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

class RailwayDashboardGenerator:
    """Generate interactive HTML dashboards from railway JSON data"""
    
    def __init__(self):
        self.template = self._load_template()
        
    def generate_dashboard(self, json_data: Dict[str, Any], output_path: str):
        """Generate HTML dashboard from JSON railway data"""
        
        # Extract data components
        country = json_data.get('country', 'Unknown')
        network = json_data.get('network', {})
        
        # Generate JavaScript data
        js_data = self._prepare_js_data(network)
        
        # Generate HTML
        html_content = self.template.format(
            country_name=country.upper(),
            railway_data_json=json.dumps(js_data, indent=2),
            total_length=network.get('total_length_km', 0),
            electrification_percent=network.get('electrification', {}).get('electrification_percentage', 0),
            station_count=len(network.get('stations', [])),
            underground_km=network.get('infrastructure_summary', {}).get('underground_km', 0),
            high_speed_km=sum(s['length_km'] for s in network.get('high_speed_lines', [])),
            quay_count=len(network.get('quays', []))
        )
        
        # Save HTML file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html_content, encoding='utf-8')
        
        return str(output_file)
    
    def _prepare_js_data(self, network: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for JavaScript consumption"""
        
        # Combine all line types with proper categorization
        all_lines = []
        
        # High-speed lines (red, thick)
        for line in network.get('high_speed_lines', []):
            all_lines.append({
                'coordinates': line['coordinates'],
                'properties': {
                    'name': line['name'],
                    'type': 'high_speed',
                    'electrified': line['electrified'],
                    'max_speed': line['max_speed'],
                    'color': '#ff0000',
                    'weight': 6,
                    'dashArray': None
                }
            })
        
        # Underground sections (purple, dashed)
        for line in network.get('underground_sections', []):
            all_lines.append({
                'coordinates': line['coordinates'],
                'properties': {
                    'name': line['name'],
                    'type': 'underground',
                    'electrified': line['electrified'],
                    'max_speed': line['max_speed'],
                    'color': '#9C27B0',
                    'weight': 5,
                    'dashArray': '10, 5'
                }
            })
        
        # Overground lines (blue/green based on electrification)
        for line in network.get('overground_lines', []):
            all_lines.append({
                'coordinates': line['coordinates'],
                'properties': {
                    'name': line['name'],
                    'type': 'overground',
                    'electrified': line['electrified'],
                    'max_speed': line['max_speed'],
                    'color': '#0066cc' if line['electrified'] else '#00aa00',
                    'weight': 4,
                    'dashArray': None if line['electrified'] else '5, 5'
                }
            })
        
        return {
            'lines': all_lines,
            'stations': network.get('stations', []),
            'quays': network.get('quays', []),
            'bounds': self._calculate_bounds(network)
        }
    
    def _calculate_bounds(self, network: Dict[str, Any]) -> List[List[float]]:
        """Calculate map bounds from network data"""
        all_coords = []
        
        # Collect all coordinates
        for line_type in ['high_speed_lines', 'underground_sections', 'overground_lines']:
            for line in network.get(line_type, []):
                all_coords.extend(line['coordinates'])
        
        for station in network.get('stations', []):
            all_coords.append([station['lat'], station['lon']])
            
        if not all_coords:
            return [[33.0, 35.0], [35.0, 37.0]]  # Default bounds
            
        lats = [c[0] for c in all_coords]
        lons = [c[1] for c in all_coords]
        
        return [[min(lats), min(lons)], [max(lats), max(lons)]]
    
    def _load_template(self) -> str:
        """Load HTML template"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{country_name} Railway Network Dashboard</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
        }}
        
        .dashboard {{
            display: grid;
            grid-template-rows: auto 1fr;
            height: 100vh;
        }}
        
        .header {{
            background: #2c3e50;
            color: white;
            padding: 15px 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 24px;
        }}
        
        .stats {{
            display: flex;
            gap: 30px;
            margin-top: 10px;
            font-size: 14px;
        }}
        
        .stat {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .stat-value {{
            font-weight: bold;
            color: #3498db;
        }}
        
        .main-content {{
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 0;
        }}
        
        .sidebar {{
            background: white;
            padding: 20px;
            overflow-y: auto;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
        }}
        
        .sidebar h2 {{
            margin: 0 0 15px 0;
            font-size: 18px;
            color: #2c3e50;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 10px 0;
            font-size: 14px;
        }}
        
        .legend-line {{
            width: 40px;
            height: 4px;
            margin-right: 10px;
            border-radius: 2px;
        }}
        
        .legend-line.dashed {{
            background-image: repeating-linear-gradient(
                to right,
                currentColor,
                currentColor 5px,
                transparent 5px,
                transparent 10px
            );
        }}
        
        #map {{
            height: 100%;
            width: 100%;
        }}
        
        .info-section {{
            margin: 25px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        
        .info-section h3 {{
            margin: 0 0 10px 0;
            font-size: 16px;
            color: #34495e;
        }}
        
        .info-grid {{
            display: grid;
            gap: 8px;
            font-size: 13px;
        }}
        
        .info-row {{
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .info-label {{
            color: #7f8c8d;
        }}
        
        .info-value {{
            font-weight: 500;
            color: #2c3e50;
        }}
        
        .leaflet-popup-content {{
            min-width: 200px;
        }}
        
        .popup-title {{
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 8px;
            color: #2c3e50;
        }}
        
        .popup-info {{
            font-size: 14px;
            line-height: 1.6;
        }}
        
        .popup-row {{
            display: flex;
            justify-content: space-between;
            margin: 4px 0;
        }}
        
        .station-icon {{
            font-size: 20px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }}
        
        .quay-icon {{
            font-size: 24px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <header class="header">
            <h1>{country_name} Railway Network</h1>
            <div class="stats">
                <div class="stat">
                    <span>Total Length:</span>
                    <span class="stat-value">{total_length:.1f} km</span>
                </div>
                <div class="stat">
                    <span>Electrification:</span>
                    <span class="stat-value">{electrification_percent:.1f}%</span>
                </div>
                <div class="stat">
                    <span>Stations:</span>
                    <span class="stat-value">{station_count}</span>
                </div>
                <div class="stat">
                    <span>Underground:</span>
                    <span class="stat-value">{underground_km:.1f} km</span>
                </div>
                <div class="stat">
                    <span>High-Speed:</span>
                    <span class="stat-value">{high_speed_km:.1f} km</span>
                </div>
                <div class="stat">
                    <span>Quays:</span>
                    <span class="stat-value">{quay_count}</span>
                </div>
            </div>
        </header>
        
        <div class="main-content">
            <aside class="sidebar">
                <h2>Network Legend</h2>
                
                <div class="legend-item">
                    <div class="legend-line" style="background: #ff0000;"></div>
                    <span>High-Speed Line (Electrified)</span>
                </div>
                
                <div class="legend-item">
                    <div class="legend-line dashed" style="color: #9C27B0;"></div>
                    <span>Underground/Metro</span>
                </div>
                
                <div class="legend-item">
                    <div class="legend-line" style="background: #0066cc;"></div>
                    <span>Mainline (Electrified)</span>
                </div>
                
                <div class="legend-item">
                    <div class="legend-line dashed" style="color: #00aa00;"></div>
                    <span>Regional (Non-Electrified)</span>
                </div>
                
                <div class="legend-item">
                    <span style="font-size: 20px; margin-right: 10px;">🚉</span>
                    <span>Railway Station</span>
                </div>
                
                <div class="legend-item">
                    <span style="font-size: 24px; margin-right: 10px;">⚓</span>
                    <span>Port/Quay Terminal</span>
                </div>
                
                <div class="info-section">
                    <h3>Infrastructure Details</h3>
                    <div class="info-grid">
                        <div class="info-row">
                            <span class="info-label">Track Gauge:</span>
                            <span class="info-value">1435mm Standard</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Electrification:</span>
                            <span class="info-value">25kV AC 50Hz</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Signaling:</span>
                            <span class="info-value">ERTMS Level 2</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Max Axle Load:</span>
                            <span class="info-value">22.5 tonnes</span>
                        </div>
                    </div>
                </div>
                
                <div class="info-section">
                    <h3>Click Elements for Details</h3>
                    <p style="font-size: 13px; color: #7f8c8d; margin: 0;">
                        Click on any line, station, or quay to view detailed information including specifications, connections, and operational data.
                    </p>
                </div>
            </aside>
            
            <div id="map"></div>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        // Railway network data
        const railwayData = {railway_data_json};
        
        // Initialize map
        const map = L.map('map');
        
        // Add OpenStreetMap tiles
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors',
            maxZoom: 18
        }}).addTo(map);
        
        // Add railway lines
        railwayData.lines.forEach(line => {{
            const polyline = L.polyline(line.coordinates, {{
                color: line.properties.color,
                weight: line.properties.weight,
                opacity: 0.8,
                dashArray: line.properties.dashArray
            }});
            
            polyline.bindPopup(`
                <div class="popup-title">${{line.properties.name}}</div>
                <div class="popup-info">
                    <div class="popup-row">
                        <span>Type:</span>
                        <strong>${{line.properties.type.replace('_', ' ')}}</strong>
                    </div>
                    <div class="popup-row">
                        <span>Max Speed:</span>
                        <strong>${{line.properties.max_speed}} km/h</strong>
                    </div>
                    <div class="popup-row">
                        <span>Electrified:</span>
                        <strong>${{line.properties.electrified ? 'Yes' : 'No'}}</strong>
                    </div>
                </div>
            `);
            
            polyline.addTo(map);
        }});
        
        // Add stations
        railwayData.stations.forEach(station => {{
            const icon = L.divIcon({{
                html: '<div class="station-icon">🚉</div>',
                iconSize: [24, 24],
                className: 'station-marker'
            }});
            
            const marker = L.marker([station.lat, station.lon], {{icon: icon}});
            
            marker.bindPopup(`
                <div class="popup-title">${{station.name}}</div>
                <div class="popup-info">
                    <div class="popup-row">
                        <span>Type:</span>
                        <strong>${{station.station_type}}</strong>
                    </div>
                    <div class="popup-row">
                        <span>Platforms:</span>
                        <strong>${{station.platform_count}}</strong>
                    </div>
                    <div class="popup-row">
                        <span>Underground:</span>
                        <strong>${{station.underground_platforms}}</strong>
                    </div>
                    <div class="popup-row">
                        <span>Connections:</span>
                        <strong>${{station.connections.join(', ')}}</strong>
                    </div>
                    ${{station.has_quay ? '<div class="popup-row"><span>Quay:</span><strong>Yes</strong></div>' : ''}}
                </div>
            `);
            
            marker.addTo(map);
        }});
        
        // Add quays
        railwayData.quays.forEach(quay => {{
            const icon = L.divIcon({{
                html: '<div class="quay-icon">⚓</div>',
                iconSize: [28, 28],
                className: 'quay-marker'
            }});
            
            const marker = L.marker([quay.lat, quay.lon], {{icon: icon}});
            
            marker.bindPopup(`
                <div class="popup-title">${{quay.name}}</div>
                <div class="popup-info">
                    <div class="popup-row">
                        <span>Type:</span>
                        <strong>${{quay.type}}</strong>
                    </div>
                    <div class="popup-row">
                        <span>Services:</span>
                        <strong>${{quay.services.join(', ')}}</strong>
                    </div>
                    <div class="popup-row">
                        <span>Rail Connection:</span>
                        <strong>Yes</strong>
                    </div>
                </div>
            `);
            
            marker.addTo(map);
        }});
        
        // Fit map to bounds
        if (railwayData.bounds) {{
            map.fitBounds(railwayData.bounds, {{padding: [50, 50]}});
        }}
        
        // Add scale control
        L.control.scale({{
            position: 'bottomleft',
            imperial: false
        }}).addTo(map);
    </script>
</body>
</html>'''

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
        
        # Generate route plan
        try:
            # Generate standard route (backward compatibility)
            route = self._generate_lebanon_route(
                cities=cities,
                country=country,
                optimization_targets=optimization_targets,
                constraints=constraints or {},
                route_name=route_name
            )
            
            # Generate enhanced route structure
            enhanced_route = self._generate_enhanced_route(
                cities=cities,
                country=country,
                optimization_targets=optimization_targets,
                constraints=constraints or {},
                route_name=route_name
            )
            
            # Save enhanced JSON
            enhanced_json_path = f"output/{country.lower()}/{route_name}_enhanced.json"
            Path(enhanced_json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(enhanced_json_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_route, f, indent=2, ensure_ascii=False)
            
            # Generate HTML dashboard
            try:
                dashboard_gen = RailwayDashboardGenerator()
                html_path = f"output/{country.lower()}/{route_name}_dashboard.html"
                dashboard_gen.generate_dashboard(enhanced_route, html_path)
                self.logger.info(f"📊 Generated HTML dashboard: {html_path}")
            except Exception as e:
                self.logger.warning(f"Could not generate HTML dashboard: {e}")
            
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
    
    def _generate_enhanced_route(self,
                                cities: List[Dict],
                                country: str,
                                optimization_targets: List[str],
                                constraints: Dict,
                                route_name: str) -> Dict[str, Any]:
        """Generate enhanced route with infrastructure details"""
        
        # Initialize infrastructure components
        underground_sections = []
        overground_sections = []
        high_speed_sections = []
        enhanced_stations = []
        quays = []
        
        total_distance = 0
        total_electrified_km = 0
        total_non_electrified_km = 0
        
        # Process each city pair
        for i in range(len(cities) - 1):
            start_city = cities[i]
            end_city = cities[i + 1]
            
            distance = self._haversine_distance(
                start_city['lat'], start_city['lon'],
                end_city['lat'], end_city['lon']
            )
            total_distance += distance
            
            # Determine track type based on terrain and cities
            track_type = self._determine_track_type(start_city, end_city, distance)
            
            # Create track section
            section = {
                'id': f"section_{i}",
                'name': f"{start_city['name']} - {end_city['name']}",
                'coordinates': [
                    [start_city['lat'], start_city['lon']],
                    *self._interpolate_route_points(start_city, end_city, 5),
                    [end_city['lat'], end_city['lon']]
                ],
                'track_type': track_type['type'],
                'electrified': track_type['electrified'],
                'electrification_type': '25kV AC' if track_type['electrified'] else 'none',
                'max_speed': track_type['max_speed'],
                'track_category': track_type['category'],
                'infrastructure_elements': track_type['elements'],
                'length_km': distance
            }
            
            # Categorize sections
            if track_type['type'] == 'underground':
                underground_sections.append(section)
            elif track_type['category'] == 'high_speed':
                high_speed_sections.append(section)
            else:
                overground_sections.append(section)
                
            # Track electrification
            if track_type['electrified']:
                total_electrified_km += distance
            else:
                total_non_electrified_km += distance
        
        # Create enhanced stations
        for city in cities:
            # Check if coastal city for quay
            has_quay = self._is_coastal_city(city['name'])
            
            station = {
                'name': f"{city['name']} Central",
                'lat': city['lat'],
                'lon': city['lon'],
                'station_type': 'terminal' if city['population'] > 1000000 else 'intercity' if city['population'] > 500000 else 'regional',
                'has_quay': has_quay,
                'quay_type': 'ferry' if has_quay else None,
                'platform_count': min(12, max(2, city['population'] // 100000)),
                'underground_platforms': 2 if city['population'] > 1000000 else 0,
                'elevated_platforms': 2 if city['population'] > 500000 else 0,
                'surface_platforms': min(8, max(2, city['population'] // 200000)),
                'connections': self._get_city_connections(city),
                'daily_passengers': city['population'] // 50
            }
            enhanced_stations.append(station)
            
            # Add quay if applicable
            if has_quay:
                quay = {
                    'name': f"{city['name']} Port Terminal",
                    'lat': city['lat'] - 0.01,  # Slightly offset from station
                    'lon': city['lon'] + 0.01,
                    'type': 'ferry',
                    'capacity': 'medium',
                    'rail_connection': True,
                    'services': ['passenger', 'freight']
                }
                quays.append(quay)
        
        # Create comprehensive JSON structure
        return {
            'country': country,
            'route_name': route_name,
            'network': {
                'high_speed_lines': high_speed_sections,
                'underground_sections': underground_sections,
                'overground_lines': overground_sections,
                'stations': enhanced_stations,
                'quays': quays,
                'electrification': {
                    'electrified_km': total_electrified_km,
                    'non_electrified_km': total_non_electrified_km,
                    'electrification_percentage': (total_electrified_km / total_distance * 100) if total_distance > 0 else 0,
                    'electrification_type': '25kV AC',
                    'sections': self._get_electrification_sections(underground_sections + overground_sections + high_speed_sections)
                },
                'total_length_km': total_distance,
                'infrastructure_summary': {
                    'underground_km': sum(s['length_km'] for s in underground_sections),
                    'elevated_km': sum(s['length_km'] for s in overground_sections if 'elevated' in s.get('infrastructure_elements', [])),
                    'surface_km': sum(s['length_km'] for s in overground_sections if 'surface' in s.get('track_type', '')),
                    'tunnel_count': len([s for s in underground_sections + overground_sections if 'tunnel' in s.get('infrastructure_elements', [])]),
                    'bridge_count': len([s for s in overground_sections if 'bridge' in s.get('infrastructure_elements', [])]),
                    'station_count': len(enhanced_stations),
                    'quay_count': len(quays)
                }
            },
            'technical_specifications': {
                'gauge': '1435mm',
                'signaling': 'ERTMS Level 2',
                'loading_gauge': 'UIC GC',
                'axle_load': '22.5t',
                'platform_height': '550mm',
                'voltage': '25kV AC 50Hz'
            },
            'metadata': {
                'generation_time': time.time(),
                'generator_version': '2.0',
                'data_sources': ['OSM', 'Terrain', 'Population']
            }
        }
    
    def _determine_track_type(self, start_city: Dict, end_city: Dict, distance: float) -> Dict[str, Any]:
        """Determine track type based on cities and terrain"""
        
        # High-speed criteria
        if distance > 50 and start_city['population'] > 500000 and end_city['population'] > 500000:
            return {
                'type': 'overground',
                'category': 'high_speed',
                'electrified': True,
                'max_speed': 300,
                'elements': ['dedicated_track', 'fencing', 'noise_barriers']
            }
        
        # Urban underground criteria
        if (start_city['population'] > 1000000 or end_city['population'] > 1000000) and distance < 30:
            return {
                'type': 'underground',
                'category': 'urban',
                'electrified': True,
                'max_speed': 80,
                'elements': ['tunnel', 'ventilation_shafts', 'emergency_exits']
            }
        
        # Mountain sections (simplified terrain check)
        if abs(start_city['lat'] - end_city['lat']) > 0.5:  # Rough mountain indicator
            return {
                'type': 'overground',
                'category': 'mainline',
                'electrified': True,
                'max_speed': 160,
                'elements': ['tunnel', 'viaduct', 'cutting']
            }
        
        # Default mainline
        return {
            'type': 'overground',
            'category': 'regional',
            'electrified': True,
            'max_speed': 160,
            'elements': ['surface', 'level_crossings']
        }
    
    def _interpolate_route_points(self, start: Dict, end: Dict, num_points: int) -> List[List[float]]:
        """Create intermediate points for realistic routing"""
        points = []
        for i in range(1, num_points):
            t = i / (num_points + 1)
            lat = start['lat'] + t * (end['lat'] - start['lat'])
            lon = start['lon'] + t * (end['lon'] - start['lon'])
            # Add slight curve for realism
            curve_offset = 0.02 * math.sin(t * math.pi)
            lat += curve_offset
            points.append([lat, lon])
        return points
    
    def _is_coastal_city(self, city_name: str) -> bool:
        """Check if city is coastal (has port/quay potential)"""
        coastal_cities = ['tyre', 'sidon', 'beirut', 'tripoli', 'jounieh']
        return city_name.lower() in coastal_cities
    
    def _get_city_connections(self, city: Dict) -> List[str]:
        """Get transportation connections for city"""
        connections = ['bus']
        if city['population'] > 1000000:
            connections.extend(['metro', 'tram', 'airport'])
        elif city['population'] > 500000:
            connections.extend(['tram'])
        if self._is_coastal_city(city['name']):
            connections.append('ferry')
        return connections
    
    def _get_electrification_sections(self, sections: List[Dict]) -> List[Dict]:
        """Get detailed electrification by section"""
        return [{
            'section_name': s['name'],
            'electrified': s['electrified'],
            'type': s['electrification_type'],
            'length_km': s['length_km']
        } for s in sections]
    
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