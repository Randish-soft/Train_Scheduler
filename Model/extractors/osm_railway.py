# File: Model/extractors/osm_railway.py
import requests
import json
import time
import logging
import re
from typing import Dict, List, Tuple

class OSMRailwayExtractor:
    def __init__(self):
        self.overpass_url = "http://overpass-api.de/api/interpreter"
        self.logger = logging.getLogger(__name__)
        
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
    
    def extract_train_stations(self, country: str) -> List[Dict]:
        """Extract train stations with details"""
        from ..utils.geo import get_country_bounds
        
        bounds = get_country_bounds(country)
        if not bounds:
            raise ValueError(f"Unknown country code: {country}")
        
        query = f"""
        [out:json][timeout:180];
        (
          node["railway"="station"]({bounds.south},{bounds.west},{bounds.north},{bounds.east});
          node["railway"="halt"]({bounds.south},{bounds.west},{bounds.north},{bounds.east});
          node["railway"="stop"]({bounds.south},{bounds.west},{bounds.north},{bounds.east});
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
                    level = 'intercity' if tags.get('train', '') == 'yes' else 'regional'
                else:
                    level = 'local'
                
                station = {
                    'id': element['id'],
                    'lat': element['lat'],
                    'lon': element['lon'],
                    'name': tags.get('name', f"Station {element['id']}"),
                    'operator': tags.get('operator', ''),
                    'platforms': tags.get('platforms', '1'),
                    'level': level
                }
                stations.append(station)
        
        return stations
    
    def extract_railway_tracks(self, country: str) -> List[Dict]:
        """Extract railway tracks with geometry from OpenStreetMap"""
        from ..utils.geo import get_country_bounds
        
        bounds = get_country_bounds(country)
        if not bounds:
            raise ValueError(f"Unknown country code: {country}")
        
        # Query that gets ways with their node references and coordinates
        query = f"""
        [out:json][timeout:300];
        (
            way["railway"="rail"]({bounds.south},{bounds.west},{bounds.north},{bounds.east});
            way["railway"="narrow_gauge"]({bounds.south},{bounds.west},{bounds.north},{bounds.east});
            way["railway"="light_rail"]({bounds.south},{bounds.west},{bounds.north},{bounds.east});
            way["railway"="subway"]({bounds.south},{bounds.west},{bounds.north},{bounds.east});
            way["railway"="tram"]({bounds.south},{bounds.west},{bounds.north},{bounds.east});
        );
        out body;
        >;
        out skel qt;
        """
        
        # Execute query
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
            if railway_type not in ['rail', 'narrow_gauge', 'light_rail', 'subway', 'tram']:
                continue
            
            # Get node references
            way_nodes = way.get('nodes', [])
            if len(way_nodes) < 2:
                continue  # Skip ways with insufficient nodes
            
            # Build geometry from node coordinates
            geometry = []
            valid_nodes = []
            
            for node_id in way_nodes:
                if node_id in nodes:
                    node = nodes[node_id]
                    geometry.append([node['lon'], node['lat']])
                    valid_nodes.append(node_id)
            
            # Only add tracks with valid geometry (at least 2 points)
            if len(geometry) >= 2:
                # Parse track attributes
                maxspeed = tags.get('maxspeed', '100')
                if isinstance(maxspeed, str):
                    # Extract numeric speed
                    match = re.search(r'\d+', maxspeed)
                    maxspeed = match.group() if match else '100'
                
                # Determine electrification
                electrified_value = tags.get('electrified', 'no')
                electrified = electrified_value in ['yes', 'contact_line', 'rail', '3rd_rail', '4th_rail']
                
                track = {
                    'id': str(way_id),
                    'nodes': valid_nodes,  # List of node IDs
                    'geometry': geometry,  # List of [lon, lat] coordinates
                    'railway_type': railway_type,
                    'maxspeed': maxspeed,
                    'electrified': electrified,
                    'usage': tags.get('usage', 'main'),
                    'gauge': tags.get('gauge', '1435'),
                    'name': tags.get('name', ''),
                    'operator': tags.get('operator', ''),
                    'service': tags.get('service', ''),  # freight, passenger, etc.
                    'tracks': tags.get('tracks', '1'),  # number of tracks
                    'tunnel': tags.get('tunnel', 'no') == 'yes',
                    'bridge': tags.get('bridge', 'no') == 'yes',
                    'ref': tags.get('ref', '')  # line reference
                }
                tracks.append(track)
        
        self.logger.info(f"Extracted {len(tracks)} tracks with geometry")
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