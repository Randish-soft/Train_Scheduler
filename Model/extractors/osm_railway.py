# File: Model/extractors/osm_railway.py
import requests
import json
import time
import logging
import re
from typing import Dict, List, Tuple, Optional

class OSMRailwayExtractor:
    def __init__(self):
        self.overpass_url = "http://overpass-api.de/api/interpreter"
        self.logger = logging.getLogger(__name__)
        
        # Define city regions with their bounding boxes
        self.city_regions = {
            'brussels': {
                'name': 'Brussels Capital Region',
                'bbox': (4.2, 50.75, 4.5, 50.95),  # (west, south, east, north)
                'radius_km': 25  # Include surrounding areas
            },
            'antwerp': {
                'name': 'Antwerp',
                'bbox': (4.2, 51.1, 4.6, 51.4),
                'radius_km': 20
            },
            'ghent': {
                'name': 'Ghent',
                'bbox': (3.5, 50.95, 3.9, 51.2),
                'radius_km': 20
            },
            'liege': {
                'name': 'Liège',
                'bbox': (5.4, 50.5, 5.7, 50.7),
                'radius_km': 20
            },
            'charleroi': {
                'name': 'Charleroi',
                'bbox': (4.3, 50.3, 4.6, 50.5),
                'radius_km': 15
            },
            'bruges': {
                'name': 'Bruges',
                'bbox': (3.1, 51.15, 3.3, 51.3),
                'radius_km': 15
            },
            'namur': {
                'name': 'Namur',
                'bbox': (4.7, 50.4, 5.0, 50.5),
                'radius_km': 15
            }
        }
    
    def get_city_bbox(self, city: str, include_suburbs: bool = True) -> Tuple[float, float, float, float]:
        """Get bounding box for a city"""
        if city.lower() not in self.city_regions:
            self.logger.warning(f"Unknown city '{city}', using Brussels as default")
            city = 'brussels'
        
        city_info = self.city_regions[city.lower()]
        
        # If including suburbs, expand the bounding box
        if include_suburbs:
            west, south, east, north = city_info['bbox']
            # Expand by approximately radius_km in each direction
            # 1 degree latitude ≈ 111 km, 1 degree longitude ≈ 111 km * cos(latitude)
            lat_expansion = city_info['radius_km'] / 111
            lon_expansion = city_info['radius_km'] / (111 * 0.65)  # cos(50°) ≈ 0.65
            
            bbox = (
                west - lon_expansion,
                south - lat_expansion,
                east + lon_expansion,
                north + lat_expansion
            )
        else:
            bbox = city_info['bbox']
        
        return bbox
    
    def extract_train_stations(self, country: str, city: Optional[str] = None) -> List[Dict]:
        """Extract train stations with optional city focus"""
        if city:
            # City-focused extraction
            bbox = self.get_city_bbox(city, include_suburbs=True)
            self.logger.info(f"Extracting stations for {city.capitalize()} region: {bbox}")
        else:
            # Country-wide extraction
            from ..utils.geo import get_country_bounds
            
            bounds = get_country_bounds(country)
            if not bounds:
                raise ValueError(f"Unknown country code: {country}")
            
            bbox = (bounds.west, bounds.south, bounds.east, bounds.north)
        
        return self._extract_stations_bbox(bbox)
    
    def extract_railway_tracks(self, country: str, city: Optional[str] = None) -> List[Dict]:
        """Extract railway tracks with optional city focus"""
        if city:
            # City-focused extraction
            bbox = self.get_city_bbox(city, include_suburbs=True)
            self.logger.info(f"Extracting tracks for {city.capitalize()} region: {bbox}")
        else:
            # Country-wide extraction
            from ..utils.geo import get_country_bounds
            
            bounds = get_country_bounds(country)
            if not bounds:
                raise ValueError(f"Unknown country code: {country}")
            
            bbox = (bounds.west, bounds.south, bounds.east, bounds.north)
        
        return self._extract_tracks_bbox(bbox)
    
    def _extract_stations_bbox(self, bbox: Tuple[float, float, float, float]) -> List[Dict]:
        """Extract stations within a bounding box"""
        west, south, east, north = bbox
        
        query = f"""
        [out:json][timeout:60];
        (
          node["railway"="station"]({south},{west},{north},{east});
          node["railway"="halt"]({south},{west},{north},{east});
          node["railway"="stop"]({south},{west},{north},{east});
        );
        out body;
        """
        
        data = self._execute_overpass_query(query)
        
        stations = []
        for element in data.get('elements', []):
            if element['type'] == 'node':
                tags = element.get('tags', {})
                
                # Determine station level/category
                if 'station' in tags.get('railway', ''):
                    # Check for main stations vs local
                    if any(key in tags for key in ['uic_ref', 'ref:SNCB']):
                        level = 'intercity'
                    elif tags.get('train', '') == 'yes':
                        level = 'regional'
                    else:
                        level = 'local'
                else:
                    level = 'local'
                
                station = {
                    'id': element['id'],
                    'lat': element['lat'],
                    'lon': element['lon'],
                    'name': tags.get('name', f"Station {element['id']}"),
                    'operator': tags.get('operator', 'SNCB/NMBS'),
                    'platforms': tags.get('platforms', '1'),
                    'level': level,
                    'ref': tags.get('ref', ''),
                    'uic_ref': tags.get('uic_ref', ''),
                    'network': tags.get('network', '')
                }
                stations.append(station)
        
        self.logger.info(f"Extracted {len(stations)} stations from bbox {bbox}")
        return stations
    
    def _extract_tracks_bbox(self, bbox: Tuple[float, float, float, float]) -> List[Dict]:
        """Extract tracks within a bounding box"""
        west, south, east, north = bbox
        
        # Smaller, more focused query
        query = f"""
        [out:json][timeout:120];
        (
            way["railway"="rail"]({south},{west},{north},{east});
            way["railway"="light_rail"]({south},{west},{north},{east});
        );
        out body;
        >;
        out skel qt;
        """
        
        response = self._execute_overpass_query(query)
        
        if not response or 'elements' not in response:
            return []
        
        # Process results
        tracks = []
        ways = {}
        nodes = {}
        
        # First pass: collect all nodes and ways
        for element in response.get('elements', []):
            if element['type'] == 'node':
                nodes[element['id']] = {
                    'lat': element['lat'],
                    'lon': element['lon']
                }
            elif element['type'] == 'way':
                ways[element['id']] = element
        
        # Second pass: build track data with geometry
        for way_id, way in ways.items():
            tags = way.get('tags', {})
            
            # Skip non-rail types
            railway_type = tags.get('railway', '')
            if railway_type not in ['rail', 'light_rail']:
                continue
            
            # Get node references
            way_nodes = way.get('nodes', [])
            if len(way_nodes) < 2:
                continue
            
            # Build geometry from node coordinates
            geometry = []
            valid_nodes = []
            
            for node_id in way_nodes:
                if node_id in nodes:
                    node = nodes[node_id]
                    geometry.append([node['lon'], node['lat']])
                    valid_nodes.append(node_id)
            
            # Only add tracks with valid geometry
            if len(geometry) >= 2:
                # Parse track attributes
                maxspeed = tags.get('maxspeed', '100')
                if isinstance(maxspeed, str):
                    match = re.search(r'\d+', maxspeed)
                    maxspeed = match.group() if match else '100'
                
                # Belgian railways are mostly electrified
                electrified_value = tags.get('electrified', 'yes')
                electrified = electrified_value in ['yes', 'contact_line', 'rail']
                
                track = {
                    'id': str(way_id),
                    'nodes': valid_nodes,
                    'geometry': geometry,
                    'railway_type': railway_type,
                    'maxspeed': maxspeed,
                    'electrified': electrified,
                    'usage': tags.get('usage', 'main'),
                    'gauge': tags.get('gauge', '1435'),
                    'name': tags.get('name', ''),
                    'operator': tags.get('operator', ''),
                    'service': tags.get('service', ''),
                    'tracks': tags.get('tracks', '1'),
                    'tunnel': tags.get('tunnel', 'no') == 'yes',
                    'bridge': tags.get('bridge', 'no') == 'yes',
                    'ref': tags.get('ref', '')
                }
                tracks.append(track)
        
        self.logger.info(f"Extracted {len(tracks)} tracks with geometry from bbox {bbox}")
        return tracks
    
    def _execute_overpass_query(self, query: str, max_retries: int = 3) -> Dict:
        """Execute an Overpass API query with retries"""
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.overpass_url, 
                    data={'data': query}, 
                    timeout=300
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    self.logger.error(f"Failed to execute Overpass query: {e}")
                    raise
        
        return {}
    
    def extract_country_railways(self, country: str) -> Dict:
        """Extract all railway infrastructure for a country"""
        query = f"""
        [out:json][timeout:300];
        area["ISO3166-1"="{country.upper()}"][admin_level=2];
        (
          way["railway"~"^(rail|light_rail|subway|tram)$"](area);
          node["railway"="station"](area);
          node["railway"="halt"](area);
          way["railway"="platform"](area);
          node["railway"="junction"](area);
        );
        out geom;
        """
        
        response = requests.post(self.overpass_url, data={'data': query})
        return response.json()