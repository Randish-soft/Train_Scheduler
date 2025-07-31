"""
Enhanced city data processing module with multiple online data sources.
Dynamically searches OpenStreetMap, GeoNames, Wikidata, and REST Countries APIs.
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import requests
from collections import defaultdict
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class InputProcessor:
    """Main class for processing city input data."""
    
    def __init__(self, country: str):
        self.country = country
        self.cities = []
        self.processor = CityDataProcessor()
        
    def process(self, demand_threshold: int = 50000, min_cities: int = 5, 
                max_cities: int = 20) -> List[Dict[str, Any]]:
        """Process cities for the country."""
        logger.info(f"🌍 Processing city data for {self.country}...")
        
        # Get cities from multiple sources
        self.cities = self.processor.process_cities(self.country, min_cities, max_cities * 2)
        
        # Filter by population threshold
        filtered_cities = [c for c in self.cities if c.get('population', 0) >= demand_threshold]
        
        # If too few cities, progressively lower threshold
        if len(filtered_cities) < min_cities:
            logger.warning(f"Only {len(filtered_cities)} cities above {demand_threshold} population")
            
            # Try with lower thresholds
            thresholds = [30000, 20000, 10000, 5000, 1000]
            for threshold in thresholds:
                filtered_cities = [c for c in self.cities if c.get('population', 0) >= threshold]
                if len(filtered_cities) >= min_cities:
                    logger.info(f"Using population threshold of {threshold} to get {len(filtered_cities)} cities")
                    break
            
            # If still not enough, estimate population for cities without data
            if len(filtered_cities) < min_cities:
                for city in self.cities:
                    if not city.get('population'):
                        self.processor._estimate_population(city)
                
                # Re-filter and sort
                filtered_cities = sorted(self.cities, key=lambda x: x.get('population', 0), reverse=True)[:max_cities]
        
        logger.info(f"✅ Retrieved {len(filtered_cities)} cities for {self.country}")
        return filtered_cities
    
    def get_cities(self) -> List[Dict[str, Any]]:
        """Get the processed cities."""
        return self.cities


class CityDataProcessor:
    """Handles fetching city data from multiple sources."""
    
    def __init__(self):
        # API endpoints
        self.osm_url = "https://nominatim.openstreetmap.org/search"
        self.geonames_url = "http://api.geonames.org/searchJSON"
        self.wikidata_url = "https://query.wikidata.org/sparql"
        self.restcountries_url = "https://restcountries.com/v3.1/name"
        self.overpass_url = "https://overpass-api.de/api/interpreter"
        
        # Free GeoNames username (you can register for free at geonames.org)
        self.geonames_username = "demo"
        
        # Headers for requests
        self.headers = {
            'User-Agent': 'RailwayRasterPipeline/1.0 (https://github.com/railway-raster)'
        }
        
    def process_cities(self, country: str, min_cities: int = 5, max_cities: int = 20) -> List[Dict[str, Any]]:
        """Main entry point to process cities for a country."""
        # Try multiple sources in parallel
        cities = self._fetch_from_multiple_sources(country, max_cities * 2)
        
        if not cities:
            logger.warning(f"⚠️ No cities found for {country}")
            return []
        
        # Deduplicate and merge city data
        cities = self._deduplicate_cities(cities)
        
        # Sort by population and limit
        cities = sorted(cities, key=lambda x: x.get('population', 0), reverse=True)[:max_cities]
        
        return cities
    
    def _fetch_from_multiple_sources(self, country: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch city data from multiple sources in parallel."""
        all_cities = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all API calls
            futures = {
                executor.submit(self._fetch_from_wikidata, country, limit): "Wikidata",
                executor.submit(self._fetch_from_osm, country, limit): "OpenStreetMap",
                executor.submit(self._fetch_from_geonames, country, limit): "GeoNames",
                executor.submit(self._fetch_from_overpass, country, limit): "Overpass",
                executor.submit(self._fetch_country_info, country): "REST Countries"
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                source = futures[future]
                try:
                    result = future.result(timeout=10)
                    if isinstance(result, list):
                        logger.info(f"✅ {source}: Found {len(result)} cities")
                        all_cities.extend(result)
                    elif isinstance(result, dict) and source == "REST Countries":
                        # Use country info to enhance other data
                        self.country_info = result
                except Exception as e:
                    logger.warning(f"⚠️ {source} failed: {str(e)}")
        
        return all_cities
    
    def _fetch_from_wikidata(self, country: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch city data from Wikidata SPARQL endpoint."""
        try:
            # SPARQL query for cities with population
            query = f"""
            SELECT ?city ?cityLabel ?population ?coord WHERE {{
                ?city wdt:P31/wdt:P279* wd:Q515 ;  # instance of city
                      wdt:P17 ?country ;             # country
                      wdt:P1082 ?population ;         # population
                      wdt:P625 ?coord .               # coordinates
                ?country rdfs:label "{country}"@en .
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            ORDER BY DESC(?population)
            LIMIT {limit}
            """
            
            response = requests.get(
                self.wikidata_url,
                params={'query': query, 'format': 'json'},
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                cities = []
                
                for item in data['results']['bindings']:
                    # Parse coordinates
                    coord_str = item['coord']['value']
                    match = re.search(r'Point\(([-\d.]+) ([-\d.]+)\)', coord_str)
                    if match:
                        lon, lat = float(match.group(1)), float(match.group(2))
                        cities.append({
                            'city': item['cityLabel']['value'],
                            'city_name': item['cityLabel']['value'],  # Add for backward compatibility
                            'population': int(item['population']['value']),
                            'latitude': lat,
                            'longitude': lon,
                            'source': 'Wikidata',
                            'country': country
                        })
                
                return cities
        except Exception as e:
            logger.debug(f"Wikidata error: {e}")
        
        return []
    
    def _fetch_from_osm(self, country: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch city data from OpenStreetMap Nominatim."""
        try:
            # Add delay to respect rate limits
            time.sleep(1)
            
            params = {
                'country': country,
                'featuretype': 'city',
                'format': 'json',
                'limit': limit,
                'extratags': 1,
                'addressdetails': 1
            }
            
            response = requests.get(
                self.osm_url,
                params=params,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                cities = []
                
                for item in data:
                    city_data = {
                        'city': item.get('display_name', '').split(',')[0],
                        'city_name': item.get('display_name', '').split(',')[0],  # Add for backward compatibility
                        'latitude': float(item.get('lat', 0)),
                        'longitude': float(item.get('lon', 0)),
                        'source': 'OpenStreetMap',
                        'country': country
                    }
                    
                    # Try to get population from extratags
                    if 'extratags' in item:
                        pop_str = item['extratags'].get('population', '')
                        if pop_str and pop_str.isdigit():
                            city_data['population'] = int(pop_str)
                    
                    cities.append(city_data)
                
                return cities
        except Exception as e:
            logger.debug(f"OSM error: {e}")
        
        return []
    
    def _fetch_from_geonames(self, country: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch city data from GeoNames."""
        try:
            # First get country code
            country_response = requests.get(
                "http://api.geonames.org/searchJSON",
                params={
                    'q': country,
                    'maxRows': 1,
                    'username': self.geonames_username,
                    'featureClass': 'A',
                    'featureCode': 'PCLI'
                },
                timeout=10
            )
            
            if country_response.status_code == 200:
                country_data = country_response.json()
                if country_data.get('geonames'):
                    country_code = country_data['geonames'][0].get('countryCode', '')
                    
                    # Now get cities
                    params = {
                        'country': country_code,
                        'featureClass': 'P',
                        'featureCode': 'PPL',
                        'cities': 'cities15000',  # cities with pop > 15000
                        'maxRows': limit,
                        'orderby': 'population',
                        'username': self.geonames_username
                    }
                    
                    response = requests.get(
                        "http://api.geonames.org/searchJSON",
                        params=params,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        cities = []
                        
                        for item in data.get('geonames', []):
                            cities.append({
                                'city': item.get('name', ''),
                                'city_name': item.get('name', ''),  # Add for backward compatibility
                                'population': item.get('population', 0),
                                'latitude': float(item.get('lat', 0)),
                                'longitude': float(item.get('lng', 0)),
                                'source': 'GeoNames',
                                'country': country
                            })
                        
                        return cities
        except Exception as e:
            logger.debug(f"GeoNames error: {e}")
        
        return []
    
    def _fetch_from_overpass(self, country: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch city data from Overpass API (OpenStreetMap)."""
        try:
            # Overpass QL query
            query = f"""
            [out:json][timeout:25];
            area["name:en"="{country}"]->.searchArea;
            (
              node["place"="city"](area.searchArea);
              node["place"="town"]["population"](area.searchArea);
            );
            out body {limit};
            """
            
            response = requests.post(
                self.overpass_url,
                data={'data': query},
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                cities = []
                
                for element in data.get('elements', []):
                    tags = element.get('tags', {})
                    city_data = {
                        'city': tags.get('name', tags.get('name:en', '')),
                        'city_name': tags.get('name', tags.get('name:en', '')),  # Add for backward compatibility
                        'latitude': element.get('lat', 0),
                        'longitude': element.get('lon', 0),
                        'source': 'Overpass',
                        'country': country
                    }
                    
                    # Try to get population
                    pop_str = tags.get('population', '')
                    if pop_str and pop_str.replace(',', '').isdigit():
                        city_data['population'] = int(pop_str.replace(',', ''))
                    
                    if city_data['city']:  # Only add if city name exists
                        cities.append(city_data)
                
                return cities
        except Exception as e:
            logger.debug(f"Overpass error: {e}")
        
        return []
    
    def _fetch_country_info(self, country: str) -> Dict[str, Any]:
        """Fetch country information from REST Countries API."""
        try:
            response = requests.get(
                f"{self.restcountries_url}/{country}",
                params={'fullText': False},
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    country_data = data[0]
                    return {
                        'name': country_data.get('name', {}).get('common', country),
                        'population': country_data.get('population', 0),
                        'area': country_data.get('area', 0),
                        'capital': country_data.get('capital', [None])[0],
                        'region': country_data.get('region', ''),
                        'subregion': country_data.get('subregion', '')
                    }
        except Exception as e:
            logger.debug(f"REST Countries error: {e}")
        
        return {}
    
    def _deduplicate_cities(self, cities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate and merge city data from multiple sources."""
        city_map = defaultdict(list)
        
        # Group cities by approximate location
        for city in cities:
            if not city.get('city'):
                continue
                
            # Create a key based on rounded coordinates
            lat_key = round(city.get('latitude', 0), 1)
            lon_key = round(city.get('longitude', 0), 1)
            key = (lat_key, lon_key)
            
            city_map[key].append(city)
        
        # Merge cities at same location
        merged_cities = []
        for city_group in city_map.values():
            if not city_group:
                continue
                
            # Merge data, preferring sources with population data
            merged = city_group[0].copy()
            
            # Ensure both city and city_name fields exist
            if 'city' in merged and 'city_name' not in merged:
                merged['city_name'] = merged['city']
            elif 'city_name' in merged and 'city' not in merged:
                merged['city'] = merged['city_name']
            
            for city in city_group[1:]:
                # Update with better data
                if city.get('population', 0) > merged.get('population', 0):
                    merged['population'] = city['population']
                
                # Collect all sources
                if 'source' in city and 'source' in merged:
                    sources = set(merged['source'].split(', '))
                    sources.add(city['source'])
                    merged['source'] = ', '.join(sorted(sources))
            
            # Only include if we have minimum required data
            if merged.get('latitude') and merged.get('longitude'):
                merged_cities.append(merged)
        
        return merged_cities
    
    def _estimate_population(self, city: Dict[str, Any]) -> None:
        """Estimate population if missing using various methods."""
        if city.get('population'):
            return
            
        # Method 1: Use country's capital population as reference
        if hasattr(self, 'country_info') and self.country_info:
            if city['city'] == self.country_info.get('capital'):
                # Capitals typically have 5-10% of country population
                city['population'] = int(self.country_info.get('population', 0) * 0.07)
                city['population_estimated'] = True
                return
        
        # Method 2: Default estimates based on city classification
        # This is a last resort - real data is always preferred
        city['population'] = 100000  # Default assumption
        city['population_estimated'] = True


def process_country_cities(country: str, demand_threshold: int = 50000, 
                          min_cities: int = 5, max_cities: int = 20) -> List[Dict[str, Any]]:
    """Main function called by the pipeline - backward compatibility."""
    processor = InputProcessor(country)
    return processor.process(demand_threshold, min_cities, max_cities)