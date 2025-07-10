"""
BCPC Pipeline: Data Loading and Validation Module

This module handles CSV data loading, validation, and preprocessing for the BCPC pipeline.
It includes robust error handling and data quality checks.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
from dataclasses import dataclass
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

logger = logging.getLogger(__name__)

@dataclass
class CityData:
    """Data structure for city information."""
    name: str
    population: int
    latitude: float
    longitude: float
    tourism_index: float
    budget: float
    region: str
    country: str
    elevation: Optional[float] = None
    administrative_level: Optional[str] = None

class DataLoader:
    """Handles CSV data loading, validation, and preprocessing."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize the data loader with caching capabilities."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.geocoder = Nominatim(user_agent="bcpc_pipeline")
        
        # Required columns in CSV
        self.required_columns = [
            'city_name', 'population', 'latitude', 'longitude', 
            'tourism_index', 'budget_total_eur'
        ]
        
        # Optional columns
        self.optional_columns = [
            'city_id', 'daily_commuters', 'terrain_ruggedness',
            'elevation', 'administrative_level'
        ]
    
    def load_csv(self, csv_path: str) -> List[CityData]:
        """
        Load and validate CSV data.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List of validated CityData objects
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV data is invalid
        """
        try:
            # Check if file exists
            csv_file = Path(csv_path)
            if not csv_file.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            logger.info(f"Loading CSV data from: {csv_path}")
            
            # Read CSV with error handling
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                logger.warning("UTF-8 decoding failed, trying with latin-1")
                df = pd.read_csv(csv_path, encoding='latin-1')
            except pd.errors.EmptyDataError:
                raise ValueError("CSV file is empty")
            except pd.errors.ParserError as e:
                raise ValueError(f"CSV parsing error: {e}")
            
            logger.info(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Validate and clean data
            df_clean = self._validate_and_clean_data(df)
            
            # Convert to CityData objects
            cities = self._convert_to_city_data(df_clean)
            
            # Geocode missing coordinates
            cities = self._geocode_missing_coordinates(cities)
            
            logger.info(f"Successfully processed {len(cities)} cities")
            return cities
            
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            raise
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the loaded DataFrame.
        
        Args:
            df: Raw DataFrame from CSV
            
        Returns:
            Cleaned and validated DataFrame
        """
        logger.info("Validating and cleaning data...")
        
        # Check for required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create a copy for cleaning
        df_clean = df.copy()
        
        # Clean column names (remove whitespace, convert to lowercase)
        df_clean.columns = df_clean.columns.str.strip().str.lower()
        self.required_columns = [col.lower() for col in self.required_columns]
        
        # Remove empty rows
        df_clean = df_clean.dropna(how='all')
        
        # Clean city names using the correct column name
        df_clean['city_name'] = df_clean['city_name'].astype(str).str.strip()
        df_clean = df_clean[df_clean['city_name'] != '']
        
        # Validate and clean numeric columns
        numeric_columns = ['population', 'latitude', 'longitude', 'tourism_index', 'budget_total_eur']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = self._clean_numeric_column(df_clean[col], col)
        
        # Validate geographic coordinates
        df_clean = self._validate_coordinates(df_clean)
        
        # Validate population data
        df_clean = self._validate_population(df_clean)
        
        # Fill missing values with forward fill strategy
        df_clean = self._forward_fill_missing_values(df_clean)
        
        # Remove duplicates (use city_name instead of name)
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['city_name'], keep='first')
        if len(df_clean) < initial_count:
            logger.warning(f"Removed {initial_count - len(df_clean)} duplicate cities")
        
        logger.info(f"Data validation complete. {len(df_clean)} cities remain after cleaning")
        return df_clean
    
    def _clean_numeric_column(self, series: pd.Series, column_name: str) -> pd.Series:
        """Clean and validate a numeric column."""
        try:
            # Remove any non-numeric characters (except decimal points and minus signs)
            if series.dtype == 'object':
                series = series.astype(str).str.replace(r'[^\d.-]', '', regex=True)
            
            # Convert to numeric
            series = pd.to_numeric(series, errors='coerce')
            
            # Check for invalid values
            invalid_count = series.isna().sum()
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} invalid values in column '{column_name}'")
            
            return series
            
        except Exception as e:
            logger.error(f"Error cleaning numeric column '{column_name}': {e}")
            return series
    
    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate geographic coordinates."""
        logger.info("Validating geographic coordinates...")
        
        # Check latitude bounds (-90 to 90)
        invalid_lat = (df['latitude'] < -90) | (df['latitude'] > 90)
        if invalid_lat.any():
            logger.warning(f"Found {invalid_lat.sum()} cities with invalid latitude")
            df.loc[invalid_lat, 'latitude'] = np.nan
        
        # Check longitude bounds (-180 to 180)
        invalid_lon = (df['longitude'] < -180) | (df['longitude'] > 180)
        if invalid_lon.any():
            logger.warning(f"Found {invalid_lon.sum()} cities with invalid longitude")
            df.loc[invalid_lon, 'longitude'] = np.nan
        
        return df
    
    def _validate_population(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate population data."""
        logger.info("Validating population data...")
        
        # Check for negative populations
        negative_pop = df['population'] < 0
        if negative_pop.any():
            logger.warning(f"Found {negative_pop.sum()} cities with negative population")
            df.loc[negative_pop, 'population'] = np.nan
        
        # Check for unreasonably large populations (> 100 million)
        large_pop = df['population'] > 100_000_000
        if large_pop.any():
            logger.warning(f"Found {large_pop.sum()} cities with population > 100M")
            # Don't automatically remove these, just log them
        
        return df
    
    def _forward_fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward fill missing values based on similar cities.
        This implements the "forward filling" mentioned in the paper.
        """
        logger.info("Forward filling missing values...")
        
        # Group by region/country for similar cities (skip if no region column)
        if 'region' in df.columns:
            for region in df['region'].unique():
                if pd.isna(region):
                    continue
                    
                region_mask = df['region'] == region
                region_df = df[region_mask]
                
                # Fill missing budget values with regional median
                if 'budget_total_eur' in df.columns:
                    budget_median = region_df['budget_total_eur'].median()
                    if not pd.isna(budget_median):
                        df.loc[region_mask, 'budget_total_eur'] = df.loc[region_mask, 'budget_total_eur'].fillna(budget_median)
                
                # Fill missing tourism index with regional median
                if 'tourism_index' in df.columns:
                    tourism_median = region_df['tourism_index'].median()
                    if not pd.isna(tourism_median):
                        df.loc[region_mask, 'tourism_index'] = df.loc[region_mask, 'tourism_index'].fillna(tourism_median)
        
        # Fill remaining missing values with global medians
        numeric_columns = ['population', 'tourism_index', 'budget_total_eur']
        for col in numeric_columns:
            if col in df.columns:
                global_median = df[col].median()
                if not pd.isna(global_median):
                    df[col] = df[col].fillna(global_median)
        
        return df
    
    def _convert_to_city_data(self, df: pd.DataFrame) -> List[CityData]:
        """Convert DataFrame to list of CityData objects."""
        logger.info("Converting data to CityData objects...")
        
        cities = []
        for _, row in df.iterrows():
            try:
                city = CityData(
                    name=str(row['city_name']),
                    population=int(row['population']) if not pd.isna(row['population']) else 0,
                    latitude=float(row['latitude']) if not pd.isna(row['latitude']) else 0.0,
                    longitude=float(row['longitude']) if not pd.isna(row['longitude']) else 0.0,
                    tourism_index=float(row['tourism_index']) if not pd.isna(row['tourism_index']) else 0.0,
                    budget=float(row['budget_total_eur']) if not pd.isna(row['budget_total_eur']) else 0.0,
                    region=str(row.get('region', row.get('city_name', 'Unknown'))),
                    country=str(row.get('country', 'Unknown')),
                    elevation=float(row['elevation']) if 'elevation' in row and not pd.isna(row['elevation']) else None,
                    administrative_level=str(row['administrative_level']) if 'administrative_level' in row and not pd.isna(row['administrative_level']) else None
                )
                cities.append(city)
                
            except Exception as e:
                logger.error(f"Error converting row to CityData: {e}")
                logger.error(f"Row data: {row.to_dict()}")
                continue
        
        return cities
    
    def _geocode_missing_coordinates(self, cities: List[CityData]) -> List[CityData]:
        """
        Geocode cities with missing coordinates using Nominatim.
        Implements caching to avoid repeated API calls.
        """
        logger.info("Geocoding cities with missing coordinates...")
        
        geocoded_cities = []
        geocoding_cache = self._load_geocoding_cache()
        
        for city in cities:
            # Check if coordinates are missing or invalid
            if city.latitude == 0.0 or city.longitude == 0.0 or pd.isna(city.latitude) or pd.isna(city.longitude):
                
                # Check cache first
                cache_key = f"{city.name}_{city.country}".lower()
                if cache_key in geocoding_cache:
                    logger.debug(f"Using cached coordinates for {city.name}")
                    city.latitude = geocoding_cache[cache_key]['latitude']
                    city.longitude = geocoding_cache[cache_key]['longitude']
                else:
                    # Geocode the city
                    coords = self._geocode_city(city.name, city.country)
                    if coords:
                        city.latitude, city.longitude = coords
                        geocoding_cache[cache_key] = {
                            'latitude': city.latitude,
                            'longitude': city.longitude
                        }
                        logger.info(f"Geocoded {city.name}: {city.latitude:.4f}, {city.longitude:.4f}")
                    else:
                        logger.warning(f"Could not geocode {city.name}, {city.country}")
            
            geocoded_cities.append(city)
        
        # Save updated cache
        self._save_geocoding_cache(geocoding_cache)
        
        return geocoded_cities
    
    def _geocode_city(self, city_name: str, country: str) -> Optional[Tuple[float, float]]:
        """Geocode a single city with retry logic."""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Format the query
                query = f"{city_name}, {country}"
                
                # Make the geocoding request
                location = self.geocoder.geocode(query, timeout=10)
                
                if location:
                    return (location.latitude, location.longitude)
                else:
                    # Try with just the city name
                    location = self.geocoder.geocode(city_name, timeout=10)
                    if location:
                        return (location.latitude, location.longitude)
                
                return None
                
            except GeocoderTimedOut:
                logger.warning(f"Geocoding timeout for {city_name} (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                continue
                
            except GeocoderServiceError as e:
                logger.error(f"Geocoding service error for {city_name}: {e}")
                return None
                
            except Exception as e:
                logger.error(f"Unexpected geocoding error for {city_name}: {e}")
                return None
        
        return None
    
    def _load_geocoding_cache(self) -> Dict[str, Dict[str, float]]:
        """Load geocoding cache from file."""
        cache_file = self.cache_dir / "geocoding_cache.json"
        
        if cache_file.exists():
            try:
                import json
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load geocoding cache: {e}")
        
        return {}
    
    def _save_geocoding_cache(self, cache: Dict[str, Dict[str, float]]):
        """Save geocoding cache to file."""
        cache_file = self.cache_dir / "geocoding_cache.json"
        
        try:
            import json
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save geocoding cache: {e}")
    
    def validate_city_data(self, cities: List[CityData]) -> List[CityData]:
        """
        Final validation of city data before processing.
        
        Args:
            cities: List of CityData objects
            
        Returns:
            List of validated CityData objects
        """
        logger.info("Performing final validation of city data...")
        
        valid_cities = []
        
        for city in cities:
            try:
                # Check required fields
                if not city.name or city.name.strip() == '':
                    logger.warning(f"Skipping city with empty name")
                    continue
                
                if city.population <= 0:
                    logger.warning(f"Skipping {city.name} with invalid population: {city.population}")
                    continue
                
                if city.latitude == 0.0 and city.longitude == 0.0:
                    logger.warning(f"Skipping {city.name} with missing coordinates")
                    continue
                
                # Check coordinate bounds
                if not (-90 <= city.latitude <= 90):
                    logger.warning(f"Skipping {city.name} with invalid latitude: {city.latitude}")
                    continue
                
                if not (-180 <= city.longitude <= 180):
                    logger.warning(f"Skipping {city.name} with invalid longitude: {city.longitude}")
                    continue
                
                valid_cities.append(city)
                
            except Exception as e:
                logger.error(f"Error validating city {city.name}: {e}")
                continue
        
        logger.info(f"Final validation complete. {len(valid_cities)} cities validated successfully")
        return valid_cities
    
    def get_data_summary(self, cities: List[CityData]) -> Dict[str, Any]:
        """Generate a summary of the loaded data."""
        if not cities:
            return {'error': 'No cities loaded'}
        
        populations = [city.population for city in cities]
        budgets = [city.budget for city in cities if city.budget > 0]
        tourism_indices = [city.tourism_index for city in cities if city.tourism_index > 0]
        
        return {
            'total_cities': len(cities),
            'countries': len(set(city.country for city in cities)),
            'regions': len(set(city.region for city in cities)),
            'population_stats': {
                'min': min(populations) if populations else 0,
                'max': max(populations) if populations else 0,
                'mean': sum(populations) / len(populations) if populations else 0,
                'median': sorted(populations)[len(populations) // 2] if populations else 0
            },
            'budget_stats': {
                'min': min(budgets) if budgets else 0,
                'max': max(budgets) if budgets else 0,
                'mean': sum(budgets) / len(budgets) if budgets else 0
            },
            'tourism_stats': {
                'min': min(tourism_indices) if tourism_indices else 0,
                'max': max(tourism_indices) if tourism_indices else 0,
                'mean': sum(tourism_indices) / len(tourism_indices) if tourism_indices else 0
            }
        }