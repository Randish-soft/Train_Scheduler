import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, List, Optional, Any, Union
import json
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RailwayPreprocessor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame) -> 'RailwayPreprocessor':
        logger.info("Fitting preprocessor...")
        
        # Store feature names
        self.feature_names = data.columns.tolist()
        
        # Fit scalers for numerical features
        numerical_features = data.select_dtypes(include=[np.number]).columns
        
        for feature in numerical_features:
            scaler_type = self.config.get('scaler_type', 'standard')
            
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            scaler.fit(data[[feature]])
            self.scalers[feature] = scaler
        
        # Fit encoders for categorical features
        categorical_features = data.select_dtypes(include=['object']).columns
        
        for feature in categorical_features:
            unique_values = data[feature].unique()
            self.encoders[feature] = {val: idx for idx, val in enumerate(unique_values)}
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        transformed_data = data.copy()
        
        # Scale numerical features
        for feature, scaler in self.scalers.items():
            if feature in transformed_data.columns:
                transformed_data[feature] = scaler.transform(transformed_data[[feature]])
        
        # Encode categorical features
        for feature, encoder in self.encoders.items():
            if feature in transformed_data.columns:
                transformed_data[feature] = transformed_data[feature].map(encoder)
                transformed_data[feature].fillna(-1, inplace=True)  # Handle unknown categories
        
        # Handle missing values
        if self.config.get('handle_missing', True):
            strategy = self.config.get('missing_strategy', 'median')
            
            if strategy == 'median':
                transformed_data.fillna(transformed_data.median(), inplace=True)
            elif strategy == 'mean':
                transformed_data.fillna(transformed_data.mean(), inplace=True)
            elif strategy == 'zero':
                transformed_data.fillna(0, inplace=True)
        
        return transformed_data.values
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        return self.fit(data).transform(data)
    
    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'config': self.config,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Preprocessor saved to {path}")
    
    def load(self, path: Union[str, Path]):
        path = Path(path)
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self.scalers = state['scalers']
        self.encoders = state['encoders']
        self.feature_names = state['feature_names']
        self.is_fitted = state['is_fitted']
        
        logger.info(f"Preprocessor loaded from {path}")
        return self

class InputValidator:
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        
        # Check required fields
        required_fields = self.schema.get('required', [])
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check data types
        type_constraints = self.schema.get('types', {})
        for field, expected_type in type_constraints.items():
            if field in data:
                if not isinstance(data[field], expected_type):
                    errors.append(f"Field {field} should be {expected_type.__name__}")
        
        # Check value ranges
        range_constraints = self.schema.get('ranges', {})
        for field, (min_val, max_val) in range_constraints.items():
            if field in data:
                value = data[field]
                if value < min_val or value > max_val:
                    errors.append(f"Field {field} should be between {min_val} and {max_val}")
        
        # Check array shapes
        shape_constraints = self.schema.get('shapes', {})
        for field, expected_shape in shape_constraints.items():
            if field in data:
                if hasattr(data[field], 'shape'):
                    actual_shape = data[field].shape
                    if len(actual_shape) != len(expected_shape):
                        errors.append(f"Field {field} has wrong number of dimensions")
                    else:
                        for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
                            if expected != -1 and actual != expected:
                                errors.append(f"Field {field} dimension {i} should be {expected}, got {actual}")
        
        return len(errors) == 0, errors

class FeatureEngineering:
    def __init__(self):
        self.engineered_features = []
        
    def create_railway_features(self, data: pd.DataFrame) -> pd.DataFrame:
        enhanced_data = data.copy()
        
        # Create interaction features
        if 'distance' in data.columns and 'elevation_change' in data.columns:
            enhanced_data['gradient'] = data['elevation_change'] / (data['distance'] + 1e-6)
            self.engineered_features.append('gradient')
        
        if 'population' in data.columns and 'area' in data.columns:
            enhanced_data['population_density'] = data['population'] / (data['area'] + 1e-6)
            self.engineered_features.append('population_density')
        
        # Create polynomial features for important variables
        if 'distance' in data.columns:
            enhanced_data['distance_squared'] = data['distance'] ** 2
            enhanced_data['distance_log'] = np.log1p(data['distance'])
            self.engineered_features.extend(['distance_squared', 'distance_log'])
        
        # Create binned features
        if 'elevation' in data.columns:
            enhanced_data['elevation_category'] = pd.cut(
                data['elevation'],
                bins=[-np.inf, 100, 500, 1000, 2000, np.inf],
                labels=['low', 'medium', 'high', 'very_high', 'extreme']
            )
            self.engineered_features.append('elevation_category')
        
        # Create rolling statistics if time series
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp')
            
            for col in ['passengers', 'revenue', 'delays']:
                if col in data.columns:
                    enhanced_data[f'{col}_rolling_mean_7'] = data[col].rolling(7).mean()
                    enhanced_data[f'{col}_rolling_std_7'] = data[col].rolling(7).std()
                    self.engineered_features.extend([f'{col}_rolling_mean_7', f'{col}_rolling_std_7'])
        
        return enhanced_data
    
    def create_graph_features(self, adjacency_matrix: np.ndarray) -> pd.DataFrame:
        features = pd.DataFrame()
        
        # Degree centrality
        features['degree'] = adjacency_matrix.sum(axis=1)
        
        # Clustering coefficient
        n = len(adjacency_matrix)
        clustering = []
        
        for i in range(n):
            neighbors = np.where(adjacency_matrix[i] > 0)[0]
            k = len(neighbors)
            
            if k < 2:
                clustering.append(0)
            else:
                edges = 0
                for j in neighbors:
                    for l in neighbors:
                        if adjacency_matrix[j, l] > 0:
                            edges += 1
                clustering.append(edges / (k * (k - 1)))
        
        features['clustering_coefficient'] = clustering
        
        # PageRank approximation (simplified)
        pagerank = np.ones(n) / n
        damping = 0.85
        
        for _ in range(10):  # iterations
            new_pagerank = (1 - damping) / n + damping * adjacency_matrix.T @ pagerank
            pagerank = new_pagerank
        
        features['pagerank'] = pagerank
        
        return features