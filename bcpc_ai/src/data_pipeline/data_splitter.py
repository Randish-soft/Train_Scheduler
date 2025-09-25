import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class RailwayDataSplitter:
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
        
    def split_by_country(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict, Dict]:
        countries = list(data.keys())
        np.random.seed(self.random_state)
        np.random.shuffle(countries)
        
        n_test = int(len(countries) * self.test_size)
        n_val = int(len(countries) * self.val_size)
        
        test_countries = countries[:n_test]
        val_countries = countries[n_test:n_test + n_val]
        train_countries = countries[n_test + n_val:]
        
        train_data = {c: data[c] for c in train_countries}
        val_data = {c: data[c] for c in val_countries}
        test_data = {c: data[c] for c in test_countries}
        
        logger.info(f"Split data: {len(train_countries)} train, {len(val_countries)} val, {len(test_countries)} test countries")
        
        return train_data, val_data, test_data
    
    def split_by_routes(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, test_size=self.test_size, random_state=self.random_state
        )
        
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
        )
        
        logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def split_temporal(self, df: pd.DataFrame, time_column: str = 'timestamp') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = df.sort_values(time_column)
        n_samples = len(df)
        
        train_end = int(n_samples * (1 - self.test_size - self.val_size))
        val_end = int(n_samples * (1 - self.test_size))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        logger.info(f"Temporal split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def split_spatial(self, gdf, bbox_splits: int = 3) -> List[Tuple]:
        bounds = gdf.total_bounds
        x_splits = np.linspace(bounds[0], bounds[2], bbox_splits + 1)
        y_splits = np.linspace(bounds[1], bounds[3], bbox_splits + 1)
        
        splits = []
        for i in range(bbox_splits):
            for j in range(bbox_splits):
                mask = (
                    (gdf.geometry.x >= x_splits[i]) & 
                    (gdf.geometry.x < x_splits[i+1]) &
                    (gdf.geometry.y >= y_splits[j]) & 
                    (gdf.geometry.y < y_splits[j+1])
                )
                splits.append(gdf[mask])
        
        return splits
    
    def create_folds(self, X: pd.DataFrame, y: pd.Series = None) -> List[Tuple]:
        folds = []
        
        if y is not None and len(np.unique(y)) <= 10:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            for train_idx, val_idx in splitter.split(X, y):
                folds.append((X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]))
        else:
            for train_idx, val_idx in self.kfold.split(X):
                if y is not None:
                    folds.append((X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]))
                else:
                    folds.append((X.iloc[train_idx], X.iloc[val_idx]))
        
        return folds