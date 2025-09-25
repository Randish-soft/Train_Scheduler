import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class RailwayDataValidator:
    def __init__(self, config_path: Optional[str] = None):
        self.validation_rules = self._load_validation_rules(config_path)
        self.validation_report = {}
        
    def _load_validation_rules(self, config_path: Optional[str]) -> Dict:
        return {
            'railways': {
                'required_columns': ['geometry'],
                'geometry_type': 'LineString',
                'crs': 'EPSG:4326',
                'max_length_km': 1000,
                'min_length_km': 0.1
            },
            'stations': {
                'required_columns': ['lat', 'lon', 'name'],
                'lat_range': (-90, 90),
                'lon_range': (-180, 180),
                'min_platforms': 1,
                'max_platforms': 50
            },
            'timetables': {
                'required_columns': ['departure_time', 'arrival_time'],
                'max_travel_time_hours': 24,
                'min_frequency_per_day': 1,
                'max_frequency_per_day': 100
            },
            'costs': {
                'min_cost_per_km': 100000,
                'max_cost_per_km': 100000000,
                'inflation_years': (1850, 2025)
            }
        }
    
    def validate_railways(self, gdf: gpd.GeoDataFrame) -> Tuple[bool, List[str]]:
        errors = []
        rules = self.validation_rules['railways']
        
        if gdf.empty:
            errors.append("Railways GeoDataFrame is empty")
            return False, errors
        
        for col in rules['required_columns']:
            if col not in gdf.columns:
                errors.append(f"Missing required column: {col}")
        
        if not gdf.geometry.empty:
            invalid_geoms = ~gdf.geometry.is_valid
            if invalid_geoms.any():
                errors.append(f"Found {invalid_geoms.sum()} invalid geometries")
            
            lengths_km = gdf.geometry.length / 1000
            too_long = lengths_km > rules['max_length_km']
            too_short = lengths_km < rules['min_length_km']
            
            if too_long.any():
                errors.append(f"{too_long.sum()} segments exceed max length")
            if too_short.any():
                errors.append(f"{too_short.sum()} segments below min length")
        
        return len(errors) == 0, errors
    
    def validate_stations(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors = []
        rules = self.validation_rules['stations']
        
        if df.empty:
            errors.append("Stations DataFrame is empty")
            return False, errors
        
        for col in rules['required_columns']:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        if 'lat' in df.columns:
            invalid_lat = (df['lat'] < rules['lat_range'][0]) | (df['lat'] > rules['lat_range'][1])
            if invalid_lat.any():
                errors.append(f"{invalid_lat.sum()} stations with invalid latitude")
        
        if 'lon' in df.columns:
            invalid_lon = (df['lon'] < rules['lon_range'][0]) | (df['lon'] > rules['lon_range'][1])
            if invalid_lon.any():
                errors.append(f"{invalid_lon.sum()} stations with invalid longitude")
        
        if 'platforms' in df.columns:
            invalid_platforms = (df['platforms'] < rules['min_platforms']) | (df['platforms'] > rules['max_platforms'])
            if invalid_platforms.any():
                errors.append(f"{invalid_platforms.sum()} stations with invalid platform count")
        
        return len(errors) == 0, errors
    
    def validate_timetables(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors = []
        rules = self.validation_rules['timetables']
        
        if df.empty:
            errors.append("Timetables DataFrame is empty")
            return False, errors
        
        for col in rules['required_columns']:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        if 'departure_time' in df.columns and 'arrival_time' in df.columns:
            df['departure_time'] = pd.to_datetime(df['departure_time'], errors='coerce')
            df['arrival_time'] = pd.to_datetime(df['arrival_time'], errors='coerce')
            
            invalid_times = df['arrival_time'] < df['departure_time']
            if invalid_times.any():
                errors.append(f"{invalid_times.sum()} entries with arrival before departure")
            
            travel_time = (df['arrival_time'] - df['departure_time']).dt.total_seconds() / 3600
            too_long = travel_time > rules['max_travel_time_hours']
            if too_long.any():
                errors.append(f"{too_long.sum()} journeys exceed max travel time")
        
        return len(errors) == 0, errors
    
    def validate_terrain(self, terrain_array: np.ndarray) -> Tuple[bool, List[str]]:
        errors = []
        
        if terrain_array.size == 0:
            errors.append("Terrain array is empty")
            return False, errors
        
        if np.isnan(terrain_array).any():
            errors.append(f"Found {np.isnan(terrain_array).sum()} NaN values in terrain")
        
        if np.isinf(terrain_array).any():
            errors.append(f"Found {np.isinf(terrain_array).sum()} infinite values in terrain")
        
        min_elev = terrain_array.min()
        max_elev = terrain_array.max()
        
        if min_elev < -500:
            errors.append(f"Unrealistic minimum elevation: {min_elev}m")
        
        if max_elev > 9000:
            errors.append(f"Unrealistic maximum elevation: {max_elev}m")
        
        return len(errors) == 0, errors
    
    def validate_costs(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors = []
        rules = self.validation_rules['costs']
        
        if df.empty:
            errors.append("Costs DataFrame is empty")
            return False, errors
        
        if 'cost_per_km' in df.columns:
            invalid_costs = (df['cost_per_km'] < rules['min_cost_per_km']) | \
                          (df['cost_per_km'] > rules['max_cost_per_km'])
            if invalid_costs.any():
                errors.append(f"{invalid_costs.sum()} entries with unrealistic costs")
        
        if 'year' in df.columns:
            invalid_years = (df['year'] < rules['inflation_years'][0]) | \
                          (df['year'] > rules['inflation_years'][1])
            if invalid_years.any():
                errors.append(f"{invalid_years.sum()} entries with invalid years")
        
        return len(errors) == 0, errors
    
    def validate_all(self, data: Dict[str, Any]) -> Dict[str, Dict]:
        report = {}
        
        for data_type, validator in [
            ('railways', self.validate_railways),
            ('stations', self.validate_stations),
            ('timetables', self.validate_timetables),
            ('terrain', self.validate_terrain),
            ('costs', self.validate_costs)
        ]:
            if data_type in data and data[data_type] is not None:
                if hasattr(data[data_type], 'empty'):
                    if not data[data_type].empty:
                        is_valid, errors = validator(data[data_type])
                    else:
                        is_valid, errors = False, [f"{data_type} is empty"]
                else:
                    is_valid, errors = validator(data[data_type])
                
                report[data_type] = {
                    'valid': is_valid,
                    'errors': errors,
                    'warnings': []
                }
                
                if not is_valid:
                    logger.warning(f"Validation failed for {data_type}: {errors}")
            else:
                report[data_type] = {
                    'valid': False,
                    'errors': [f"{data_type} data not provided"],
                    'warnings': []
                }