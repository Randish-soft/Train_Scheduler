import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from typing import Dict, List, Tuple, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class RailwayCrossValidator:
    def __init__(self, n_splits: int = 5, validation_strategy: str = 'kfold', 
                 random_state: int = 42):
        self.n_splits = n_splits
        self.validation_strategy = validation_strategy
        self.random_state = random_state
        self.results = []
        
    def validate_model(self, model_class: type, model_params: Dict,
                      X: np.ndarray, y: np.ndarray,
                      trainer_class: type, trainer_params: Dict) -> Dict[str, float]:
        
        if self.validation_strategy == 'kfold':
            splitter = KFold(n_splits=self.n_splits, shuffle=True, 
                           random_state=self.random_state)
        elif self.validation_strategy == 'stratified':
            splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                      random_state=self.random_state)
        elif self.validation_strategy == 'timeseries':
            splitter = TimeSeriesSplit(n_splits=self.n_splits)
        else:
            raise ValueError(f"Unknown validation strategy: {self.validation_strategy}")
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            logger.info(f"Training fold {fold + 1}/{self.n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create model instance
            model = model_class(**model_params)
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Train model
            trainer = trainer_class(model, trainer_params)
            trainer.train(train_loader, val_loader, epochs=trainer_params.get('epochs', 50))
            
            # Evaluate
            metrics = trainer.validate(val_loader)
            fold_results.append(metrics)
            
            logger.info(f"Fold {fold + 1} results: {metrics}")
        
        # Aggregate results
        aggregated = self._aggregate_fold_results(fold_results)
        self.results.append(aggregated)
        
        return aggregated
    
    def validate_ensemble(self, models: List[nn.Module], X: np.ndarray, 
                         y: np.ndarray, weights: Optional[List[float]] = None) -> Dict[str, float]:
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        splitter = KFold(n_splits=self.n_splits, shuffle=True, 
                        random_state=self.random_state)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(splitter.split(X)):
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            # Get predictions from each model
            ensemble_pred = np.zeros_like(y_val)
            
            for model, weight in zip(models, weights):
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_val)
                    pred = model(X_tensor).numpy()
                    ensemble_pred += weight * pred
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                'mse': mean_squared_error(y_val, ensemble_pred),
                'mae': mean_absolute_error(y_val, ensemble_pred),
                'r2': r2_score(y_val, ensemble_pred)
            }
            
            fold_results.append(metrics)
        
        return self._aggregate_fold_results(fold_results)
    
    def country_based_validation(self, model_class: type, model_params: Dict,
                                country_data: Dict[str, Dict],
                                trainer_class: type, trainer_params: Dict) -> Dict[str, Dict]:
        
        countries = list(country_data.keys())
        results = {}
        
        for test_country in countries:
            logger.info(f"Testing on {test_country}")
            
            # Use all other countries for training
            train_countries = [c for c in countries if c != test_country]
            
            # Prepare training data
            train_features = []
            train_labels = []
            
            for country in train_countries:
                if 'features' in country_data[country]:
                    train_features.append(country_data[country]['features'])
                    train_labels.append(country_data[country]['labels'])
            
            if train_features:
                X_train = np.concatenate(train_features)
                y_train = np.concatenate(train_labels)
                
                # Test data
                X_test = country_data[test_country]['features']
                y_test = country_data[test_country]['labels']
                
                # Train model
                model = model_class(**model_params)
                
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train),
                    torch.FloatTensor(y_train)
                )
                test_dataset = TensorDataset(
                    torch.FloatTensor(X_test),
                    torch.FloatTensor(y_test)
                )
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                trainer = trainer_class(model, trainer_params)
                trainer.train(train_loader, test_loader, 
                            epochs=trainer_params.get('epochs', 50))
                
                # Evaluate
                metrics = trainer.validate(test_loader)
                results[test_country] = metrics
                
                logger.info(f"{test_country} results: {metrics}")
        
        return results
    
    def temporal_validation(self, model: nn.Module, data: pd.DataFrame,
                          time_column: str, feature_columns: List[str],
                          target_column: str, test_size: float = 0.2) -> Dict[str, float]:
        
        # Sort by time
        data = data.sort_values(time_column)
        
        # Split temporally
        split_point = int(len(data) * (1 - test_size))
        train_data = data[:split_point]
        test_data = data[split_point:]
        
        X_train = train_data[feature_columns].values
        y_train = train_data[target_column].values
        X_test = test_data[feature_columns].values
        y_test = test_data[target_column].values
        
        # Train model
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Evaluate
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                outputs = model(inputs)
                predictions.append(outputs.numpy())
                actuals.append(targets.numpy())
        
        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        return {
            'mse': mean_squared_error(actuals, predictions),
            'mae': mean_absolute_error(actuals, predictions),
            'r2': r2_score(actuals, predictions)
        }
    
    def _aggregate_fold_results(self, fold_results: List[Dict]) -> Dict[str, float]:
        aggregated = {}
        
        if not fold_results:
            return aggregated
        
        # Get all metric keys
        all_keys = set()
        for result in fold_results:
            all_keys.update(result.keys())
        
        for key in all_keys:
            values = [r[key] for r in fold_results if key in r]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated
    
    def get_summary_report(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        
        return pd.DataFrame(self.results)