import os
import json
import yaml
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class RailwayDataLoader:
    def __init__(self, data_dir: str = "data", config_path: str = "configs/data/train_config.yaml"):
        self.data_dir = Path(data_dir)
        self.config = self._load_config(config_path)
        self.train_countries = ['belgium', 'switzerland', 'netherlands', 'germany', 'france']
        self.test_countries = ['lebanon', 'egypt', 'morocco', 'jordan']
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_osm_railways(self, country: str, data_type: str = 'train') -> gpd.GeoDataFrame:
        file_path = self.data_dir / data_type / country / 'railways.geojson'
        if file_path.exists():
            return gpd.read_file(file_path)
        else:
            logger.warning(f"Railway data not found for {country}")
            return gpd.GeoDataFrame()
    
    def load_stations(self, country: str, data_type: str = 'train') -> pd.DataFrame:
        file_path = self.data_dir / data_type / 'stations' / f'{country}_stations.csv'
        if file_path.exists():
            return pd.read_csv(file_path)
        else:
            logger.warning(f"Station data not found for {country}")
            return pd.DataFrame()
    
    def load_timetables(self, country: str, data_type: str = 'train') -> pd.DataFrame:
        file_path = self.data_dir / data_type / 'timetables' / f'{country}_timetables.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            if 'departure_time' in df.columns:
                df['departure_time'] = pd.to_datetime(df['departure_time'])
            if 'arrival_time' in df.columns:
                df['arrival_time'] = pd.to_datetime(df['arrival_time'])
            return df
        else:
            logger.warning(f"Timetable data not found for {country}")
            return pd.DataFrame()
    
    def load_terrain(self, country: str, data_type: str = 'train') -> np.ndarray:
        file_path = self.data_dir / data_type / 'terrain' / f'{country}_elevation.npy'
        if file_path.exists():
            return np.load(file_path)
        else:
            logger.warning(f"Terrain data not found for {country}")
            return np.array([])
    
    def load_costs(self, country: str, data_type: str = 'train') -> pd.DataFrame:
        file_path = self.data_dir / data_type / 'costs' / f'{country}_costs.csv'
        if file_path.exists():
            return pd.read_csv(file_path)
        else:
            logger.warning(f"Cost data not found for {country}")
            return pd.DataFrame()
    
    def load_passenger_flow(self, country: str, data_type: str = 'train') -> pd.DataFrame:
        file_path = self.data_dir / data_type / 'passenger_flow' / f'{country}_flow.csv'
        if file_path.exists():
            return pd.read_csv(file_path)
        else:
            logger.warning(f"Passenger flow data not found for {country}")
            return pd.DataFrame()
    
    def load_country_data(self, country: str, data_type: str = 'train') -> Dict:
        return {
            'railways': self.load_osm_railways(country, data_type),
            'stations': self.load_stations(country, data_type),
            'timetables': self.load_timetables(country, data_type),
            'terrain': self.load_terrain(country, data_type),
            'costs': self.load_costs(country, data_type),
            'passenger_flow': self.load_passenger_flow(country, data_type)
        }
    
    def load_all_train_data(self) -> Dict[str, Dict]:
        train_data = {}
        for country in self.train_countries:
            logger.info(f"Loading training data for {country}")
            train_data[country] = self.load_country_data(country, 'train')
        return train_data
    
    def load_all_test_data(self) -> Dict[str, Dict]:
        test_data = {}
        for country in self.test_countries:
            logger.info(f"Loading test data for {country}")
            test_data[country] = self.load_country_data(country, 'test')
        return test_data