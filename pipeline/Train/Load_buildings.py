#!/usr/bin/env python3
"""
Load_buildings.py - Module for extracting and processing building footprints
Part of the Train pipeline for countries with excellent railway systems
"""

import os
import json
import logging
from typing import Tuple, Dict, List, Optional, Union
from datetime import datetime

try:
    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import mapping, shape
    import pandas as pd
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("Install with: pip3 install osmnx geopandas shapely pandas")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure osmnx - handle version differences
try:
    # For older versions of osmnx
    ox.config(use_cache=True, log_console=False)
except AttributeError:
    # For newer versions of osmnx (2.0+)
    ox.settings.use_cache = True
    ox.settings.log_console = False

class BuildingLoader:
    """Class for loading and processing building footprints"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize the BuildingLoader
        
        Parameters:
        -----------
        cache_dir : str
            Directory for caching downloaded data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def extract_buildings(self, 
                         location: Union[str, Tuple[float, float]], 
                         radius: int = 1000,
                         simplify: bool = True) -> Optional[gpd.GeoDataFrame]:
        """
        Extract building footprints from OpenStreetMap
        
        Parameters:
        -----------
        location : str or tuple
            Address string or (latitude, longitude) coordinates
        radius : int
            Search radius in meters
        simplify : bool
            Whether to simplify geometries
        
        Returns:
        --------
        GeoDataFrame with building footprints
        """
        try:
            logger.info(f"Extracting buildings for {location} with radius {radius}m")
            
            # Fetch building data
            if isinstance(location, str):
                buildings = ox.features_from_address(
                    location,
                    tags={'building': True},
                    dist=radius
                )
            else:
                buildings = ox.features_from_point(
                    location,
                    tags={'building': True},
                    dist=radius
                )
            
            # Filter to only polygons
            buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            
            # Simplify geometries if requested
            if simplify and not buildings.empty:
                buildings['geometry'] = buildings.geometry.simplify(tolerance=1, preserve_topology=True)
            
            logger.info(f"Extracted {len(buildings)} buildings")
            return buildings
            
        except Exception as e:
            logger.error(f"Error extracting buildings: {e}")
            return None
    
    def process_buildings(self, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Process and standardize building data
        
        Parameters:
        -----------
        buildings : GeoDataFrame
            Raw building data
        
        Returns:
        --------
        Processed GeoDataFrame
        """
        if buildings is None or buildings.empty:
            return buildings
        
        # Reset index
        buildings = buildings.reset_index(drop=True)
        
        # Add standard properties
        processed = gpd.GeoDataFrame()
        processed['geometry'] = buildings['geometry']
        
        # Add building attributes if available
        if 'building' in buildings.columns:
            processed['building_type'] = buildings['building']
        else:
            processed['building_type'] = 'yes'
        
        if 'height' in buildings.columns:
            processed['height'] = buildings['height']
        
        if 'levels' in buildings.columns:
            processed['levels'] = buildings['levels']
        
        # Add visualization properties
        processed['fill'] = '#808080'  # Grey color
        processed['fill_opacity'] = 0.7
        processed['stroke'] = '#404040'  # Darker grey
        processed['stroke_width'] = 1
        
        return processed
    
    def save_as_geojson(self, buildings: gpd.GeoDataFrame, output_path: str):
        """Save buildings as GeoJSON file"""
        if buildings is not None and not buildings.empty:
            buildings.to_file(output_path, driver='GeoJSON')
            logger.info(f"Saved {len(buildings)} buildings to {output_path}")
        else:
            logger.warning("No buildings to save")
    
    def save_as_json(self, buildings: gpd.GeoDataFrame, output_path: str):
        """
        Save buildings as simplified JSON
        
        Parameters:
        -----------
        buildings : GeoDataFrame
            Building data
        output_path : str
            Path to save JSON file
        """
        if buildings is None or buildings.empty:
            logger.warning("No buildings to save")
            return
        
        buildings_list = []
        
        for idx, row in buildings.iterrows():
            geom = row.geometry
            
            building_dict = {
                'id': str(idx),
                'type': geom.geom_type,
                'properties': {
                    'fill': '#808080',
                    'stroke': '#404040',
                    'fill_opacity': 0.7,
                    'stroke_width': 1
                }
            }
            
            # Add building type if available
            if 'building_type' in row:
                building_dict['properties']['building_type'] = row['building_type']
            
            # Extract coordinates
            if geom.geom_type == 'Polygon':
                building_dict['coordinates'] = [list(geom.exterior.coords)]
            elif geom.geom_type == 'MultiPolygon':
                coords = []
                for poly in geom.geoms:
                    coords.append([list(poly.exterior.coords)])
                building_dict['coordinates'] = coords
            
            buildings_list.append(building_dict)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump({
                'type': 'FeatureCollection',
                'features': buildings_list,
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'count': len(buildings_list)
                }
            }, f, indent=2)
        
        logger.info(f"Saved {len(buildings_list)} buildings to {output_path}")

def extract_buildings_for_location(location: Union[str, Tuple[float, float]],
                                  radius: int = 1000,
                                  output_geojson: str = None,
                                  output_json: str = None) -> int:
    """
    Convenience function to extract and save buildings for a location
    
    Parameters:
    -----------
    location : str or tuple
        Address or (lat, lon) coordinates
    radius : int
        Search radius in meters
    output_geojson : str
        Path to save GeoJSON file
    output_json : str
        Path to save JSON file
    
    Returns:
    --------
    int : Number of buildings extracted
    """
    loader = BuildingLoader()
    
    # Extract buildings
    buildings = loader.extract_buildings(location, radius)
    
    if buildings is None or buildings.empty:
        return 0
    
    # Process buildings
    buildings = loader.process_buildings(buildings)
    
    # Save files
    if output_geojson:
        loader.save_as_geojson(buildings, output_geojson)
    
    if output_json:
        loader.save_as_json(buildings, output_json)
    
    return len(buildings)

def batch_extract_buildings(locations: List[Dict], output_dir: str = "data/buildings") -> Dict:
    """
    Extract buildings for multiple locations
    
    Parameters:
    -----------
    locations : list
        List of dictionaries with 'name', 'coords', and 'radius'
    output_dir : str
        Directory to save output files
    
    Returns:
    --------
    dict : Summary of extraction results
    """
    os.makedirs(output_dir, exist_ok=True)
    loader = BuildingLoader()
    results = {}
    
    for loc in locations:
        name = loc['name']
        coords = loc['coords']
        radius = loc.get('radius', 1000)
        
        logger.info(f"Processing {name}...")
        
        # Extract buildings
        buildings = loader.extract_buildings(coords, radius)
        
        if buildings is not None and not buildings.empty:
            # Process
            buildings = loader.process_buildings(buildings)
            
            # Save files
            safe_name = name.replace(' ', '_').lower()
            geojson_path = os.path.join(output_dir, f"{safe_name}_buildings.geojson")
            json_path = os.path.join(output_dir, f"{safe_name}_buildings.json")
            
            loader.save_as_geojson(buildings, geojson_path)
            loader.save_as_json(buildings, json_path)
            
            results[name] = {
                'status': 'success',
                'count': len(buildings),
                'files': {
                    'geojson': geojson_path,
                    'json': json_path
                }
            }
        else:
            results[name] = {
                'status': 'failed',
                'count': 0,
                'error': 'No buildings found'
            }
    
    # Save summary
    summary_path = os.path.join(output_dir, 'extraction_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Batch extraction complete. Summary saved to {summary_path}")
    return results

# Example usage
if __name__ == "__main__":
    # Test with a single location
    print("Testing building extraction for Tokyo Station area...")
    
    # Tokyo Station coordinates
    tokyo_station = (35.6812, 139.7671)
    
    count = extract_buildings_for_location(
        location=tokyo_station,
        radius=500,
        output_geojson="tokyo_station_buildings.geojson",
        output_json="tokyo_station_buildings.json"
    )
    
    print(f"Extracted {count} buildings")
    
    # Test batch extraction
    print("\nTesting batch extraction for multiple train stations...")
    
    train_stations = [
        {'name': 'Tokyo Station', 'coords': (35.6812, 139.7671), 'radius': 500},
        {'name': 'Shinjuku Station', 'coords': (35.6896, 139.7006), 'radius': 500},
        {'name': 'Shibuya Station', 'coords': (35.6580, 139.7016), 'radius': 500},
    ]
    
    results = batch_extract_buildings(train_stations, output_dir="test_output")
    
    for station, data in results.items():
        print(f"{station}: {data['count']} buildings ({data['status']})")