# File: railway_ai/utils/ml.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error, r2_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from typing import Dict, List, Tuple, Optional, Union, Any
import joblib
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class ModelType(Enum):
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    CLUSTERER = "clusterer"

@dataclass
class ModelPerformance:
    model_type: str
    accuracy: Optional[float] = None
    r2_score: Optional[float] = None
    rmse: Optional[float] = None
    cross_val_score: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    silhouette_score: Optional[float] = None
    n_samples: int = 0
    training_time: float = 0.0

@dataclass
class FeatureImportance:
    feature_name: str
    importance: float
    rank: int

class MLPipeline:
    """Comprehensive ML pipeline for railway intelligence"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.performance_history = {}
        
    def prepare_features(self, data: Union[pd.DataFrame, List[Dict]], 
                        target_column: Optional[str] = None,
                        feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare features for ML training with automatic preprocessing"""
        
        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Select features
        if feature_columns:
            X_df = df[feature_columns]
        elif target_column:
            X_df = df.drop(columns=[target_column])
        else:
            X_df = df
        
        # Encode categorical variables
        X_processed = self._encode_categorical_features(X_df)
        
        # Extract target if specified
        y = df[target_column].values if target_column and target_column in df.columns else None
        
        return X_processed, y
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies"""
        df_clean = df.copy()
        
        for column in df_clean.columns:
            if df_clean[column].dtype in ['object', 'category']:
                # Categorical: fill with mode
                mode_value = df_clean[column].mode()
                fill_value = mode_value[0] if len(mode_value) > 0 else 'unknown'
                df_clean[column] = df_clean[column].fillna(fill_value)
            else:
                # Numerical: fill with median
                median_value = df_clean[column].median()
                df_clean[column] = df_clean[column].fillna(median_value)
        
        return df_clean
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> np.ndarray:
        """Encode categorical features"""
        df_encoded = df.copy()
        
        for column in df_encoded.columns:
            if df_encoded[column].dtype in ['object', 'category']:
                # Use label encoding for categorical variables
                if column not in self.encoders:
                    self.encoders[column] = LabelEncoder()
                    df_encoded[column] = self.encoders[column].fit_transform(df_encoded[column].astype(str))
                else:
                    # Handle new categories during prediction
                    try:
                        df_encoded[column] = self.encoders[column].transform(df_encoded[column].astype(str))
                    except ValueError:
                        # New category encountered
                        known_categories = list(self.encoders[column].classes_)
                        new_categories = set(df_encoded[column].astype(str)) - set(known_categories)
                        
                        # Extend encoder with new categories
                        extended_categories = known_categories + list(new_categories)
                        self.encoders[column].classes_ = np.array(extended_categories)
                        df_encoded[column] = self.encoders[column].transform(df_encoded[column].astype(str))
        
        return df_encoded.values
    
    def train_classifier(self, X: np.ndarray, y: np.ndarray, 
                        model_name: str = "default_classifier",
                        algorithm: str = "random_forest",
                        hyperparameter_tuning: bool = False) -> ModelPerformance:
        """Train classification model"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers[model_name] = scaler
        
        # Select algorithm
        if algorithm == "random_forest":
            if hyperparameter_tuning:
                model = self._tune_random_forest_classifier(X_train_scaled, y_train)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        # Train model
        import time
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Store model
        self.models[model_name] = model
        
        # Evaluate performance
        y_pred = model.predict(X_test_scaled)
        accuracy = (y_pred == y_test).mean()
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = {f"feature_{i}": imp for i, imp in enumerate(model.feature_importances_)}
        
        performance = ModelPerformance(
            model_type="classifier",
            accuracy=accuracy,
            cross_val_score=cv_scores.mean(),
            feature_importance=feature_importance,
            n_samples=len(X),
            training_time=training_time
        )
        
        self.performance_history[model_name] = performance
        return performance
    
    def train_regressor(self, X: np.ndarray, y: np.ndarray,
                       model_name: str = "default_regressor",
                       algorithm: str = "random_forest",
                       hyperparameter_tuning: bool = False) -> ModelPerformance:
        """Train regression model"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers[model_name] = scaler
        
        # Select algorithm
        if algorithm == "random_forest":
            if hyperparameter_tuning:
                model = self._tune_random_forest_regressor(X_train_scaled, y_train)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        elif algorithm == "gradient_boosting":
            if hyperparameter_tuning:
                model = self._tune_gradient_boosting_regressor(X_train_scaled, y_train)
            else:
                model = GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        
        # Train model
        import time
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Store model
        self.models[model_name] = model
        
        # Evaluate performance
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = {f"feature_{i}": imp for i, imp in enumerate(model.feature_importances_)}
        
        performance = ModelPerformance(
            model_type="regressor",
            r2_score=r2,
            rmse=rmse,
            cross_val_score=cv_scores.mean(),
            feature_importance=feature_importance,
            n_samples=len(X),
            training_time=training_time
        )
        
        self.performance_history[model_name] = performance
        return performance
    
    def train_clusterer(self, X: np.ndarray, 
                       model_name: str = "default_clusterer",
                       algorithm: str = "kmeans",
                       n_clusters: Optional[int] = None) -> ModelPerformance:
        """Train clustering model"""
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Store scaler
        self.scalers[model_name] = scaler
        
        # Auto-determine optimal clusters if not specified
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(X_scaled, max_clusters=min(10, len(X) // 2))
        
        # Select algorithm
        if algorithm == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        elif algorithm == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)
        elif algorithm == "agglomerative":
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            model = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        
        # Train model
        import time
        start_time = time.time()
        cluster_labels = model.fit_predict(X_scaled)
        training_time = time.time() - start_time
        
        # Store model
        self.models[model_name] = model
        
        # Evaluate clustering quality
        if len(np.unique(cluster_labels)) > 1:
            silhouette = silhouette_score(X_scaled, cluster_labels)
        else:
            silhouette = 0.0
        
        performance = ModelPerformance(
            model_type="clusterer",
            silhouette_score=silhouette,
            n_samples=len(X),
            training_time=training_time
        )
        
        self.performance_history[model_name] = performance
        return performance
    
    def predict(self, X: np.ndarray, model_name: str) -> np.ndarray:
        """Make predictions using trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        scaler = self.scalers.get(model_name)
        
        # Scale features if scaler exists
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        if hasattr(model, 'predict'):
            return model.predict(X_scaled)
        elif hasattr(model, 'fit_predict'):
            # For clustering models that don't have separate predict
            return model.fit_predict(X_scaled)
        else:
            raise ValueError(f"Model '{model_name}' doesn't support prediction")
    
    def predict_proba(self, X: np.ndarray, model_name: str) -> np.ndarray:
        """Get prediction probabilities for classification models"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        scaler = self.scalers.get(model_name)
        
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model '{model_name}' doesn't support probability prediction")
        
        # Scale features if scaler exists
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        return model.predict_proba(X_scaled)
    
    def get_feature_importance(self, model_name: str, feature_names: Optional[List[str]] = None) -> List[FeatureImportance]:
        """Get feature importance rankings"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model '{model_name}' doesn't provide feature importance")
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Create importance objects with rankings
        importance_objects = []
        sorted_indices = np.argsort(importances)[::-1]  # Sort in descending order
        
        for rank, idx in enumerate(sorted_indices):
            importance_objects.append(FeatureImportance(
                feature_name=feature_names[idx],
                importance=importances[idx],
                rank=rank + 1
            ))
        
        return importance_objects
    
    def feature_selection(self, X: np.ndarray, y: np.ndarray, 
                         k: int = 10, method: str = "mutual_info") -> Tuple[np.ndarray, List[int]]:
        """Select top k features using specified method"""
        
        if method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        elif method == "f_regression":
            selector = SelectKBest(score_func=f_regression, k=k)
        else:
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        
        return X_selected, list(selected_indices)
    
    def dimensionality_reduction(self, X: np.ndarray, 
                               method: str = "pca", 
                               n_components: Optional[int] = None) -> Tuple[np.ndarray, object]:
        """Reduce dimensionality of features"""
        
        if n_components is None:
            n_components = min(10, X.shape[1])
        
        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=self.random_state)
        else:
            reducer = PCA(n_components=n_components, random_state=self.random_state)
        
        X_reduced = reducer.fit_transform(X)
        
        return X_reduced, reducer
    
    def _tune_random_forest_classifier(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """Hyperparameter tuning for Random Forest Classifier"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=self.random_state)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_
    
    def _tune_random_forest_regressor(self, X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
        """Hyperparameter tuning for Random Forest Regressor"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=self.random_state)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_
    
    def _tune_gradient_boosting_regressor(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingRegressor:
        """Hyperparameter tuning for Gradient Boosting Regressor"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        
        gb = GradientBoostingRegressor(random_state=self.random_state)
        grid_search = GridSearchCV(gb, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_
    
    def _find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using elbow method"""
        inertias = []
        K_range = range(2, min(max_clusters + 1, len(X)))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point (simplified)
        if len(inertias) >= 2:
            # Calculate rate of change
            deltas = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
            # Find point where improvement slows down significantly
            optimal_idx = 0
            for i in range(1, len(deltas)):
                if deltas[i] < deltas[i-1] * 0.7:  # 30% less improvement
                    optimal_idx = i
                    break
            
            return K_range[optimal_idx]
        
        return 3  # Default fallback
    
    def save_models(self, directory: str) -> None:
        """Save all trained models and preprocessing objects"""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        models_dir = save_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, models_dir / f"{name}.pkl")
        
        # Save scalers
        scalers_dir = save_dir / "scalers"
        scalers_dir.mkdir(exist_ok=True)
        
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, scalers_dir / f"{name}.pkl")
        
        # Save encoders
        encoders_dir = save_dir / "encoders"
        encoders_dir.mkdir(exist_ok=True)
        
        for name, encoder in self.encoders.items():
            joblib.dump(encoder, encoders_dir / f"{name}.pkl")
        
        # Save performance history
        with open(save_dir / "performance_history.json", "w") as f:
            # Convert to serializable format
            serializable_history = {}
            for name, perf in self.performance_history.items():
                serializable_history[name] = {
                    "model_type": perf.model_type,
                    "accuracy": perf.accuracy,
                    "r2_score": perf.r2_score,
                    "rmse": perf.rmse,
                    "cross_val_score": perf.cross_val_score,
                    "feature_importance": perf.feature_importance,
                    "silhouette_score": perf.silhouette_score,
                    "n_samples": perf.n_samples,
                    "training_time": perf.training_time
                }
            json.dump(serializable_history, f, indent=2)
    
    def load_models(self, directory: str) -> None:
        """Load saved models and preprocessing objects"""
        load_dir = Path(directory)
        
        if not load_dir.exists():
            raise FileNotFoundError(f"Directory {directory} not found")
        
        # Load models
        models_dir = load_dir / "models"
        if models_dir.exists():
            for model_file in models_dir.glob("*.pkl"):
                name = model_file.stem
                self.models[name] = joblib.load(model_file)
        
        # Load scalers
        scalers_dir = load_dir / "scalers"
        if scalers_dir.exists():
            for scaler_file in scalers_dir.glob("*.pkl"):
                name = scaler_file.stem
                self.scalers[name] = joblib.load(scaler_file)
        
        # Load encoders
        encoders_dir = load_dir / "encoders"
        if encoders_dir.exists():
            for encoder_file in encoders_dir.glob("*.pkl"):
                name = encoder_file.stem
                self.encoders[name] = joblib.load(encoder_file)
        
        # Load performance history
        perf_file = load_dir / "performance_history.json"
        if perf_file.exists():
            with open(perf_file, "r") as f:
                history_data = json.load(f)
                
            for name, perf_dict in history_data.items():
                self.performance_history[name] = ModelPerformance(**perf_dict)
    
    def get_model_summary(self) -> Dict[str, Dict]:
        """Get summary of all trained models"""
        summary = {}
        
        for name, performance in self.performance_history.items():
            model_info = {
                "type": performance.model_type,
                "n_samples": performance.n_samples,
                "training_time": f"{performance.training_time:.2f}s"
            }
            
            if performance.accuracy is not None:
                model_info["accuracy"] = f"{performance.accuracy:.3f}"
            
            if performance.r2_score is not None:
                model_info["r2_score"] = f"{performance.r2_score:.3f}"
            
            if performance.rmse is not None:
                model_info["rmse"] = f"{performance.rmse:.3f}"
            
            if performance.cross_val_score is not None:
                model_info["cv_score"] = f"{performance.cross_val_score:.3f}"
            
            if performance.silhouette_score is not None:
                model_info["silhouette"] = f"{performance.silhouette_score:.3f}"
            
            summary[name] = model_info
        
        return summary

# Specialized ML functions for railway applications

class RailwayMLUtils:
    """Specialized ML utilities for railway planning"""
    
    @staticmethod
    def create_demand_features(stations: List[Dict], population_data: List[Dict]) -> np.ndarray:
        """Create feature matrix for demand prediction"""
        features = []
        
        for station in stations:
            station_features = [
                station.get('population_1km', 0),
                station.get('population_5km', 0),
                station.get('commercial_density', 0),
                station.get('industrial_proximity', 0),
                station.get('transport_connections', 0),
                station.get('tourist_attractions', 0),
                station.get('employment_centers', 0),
                station.get('university_proximity', 0),
                station.get('airport_distance', 100),  # km
                station.get('city_center_distance', 10)  # km
            ]
            features.append(station_features)
        
        return np.array(features)
    
    @staticmethod
    def create_cost_features(route_segments: List[Dict]) -> np.ndarray:
        """Create feature matrix for cost prediction"""
        features = []
        
        for segment in route_segments:
            segment_features = [
                segment.get('length_km', 0),
                segment.get('max_gradient_percent', 0),
                segment.get('avg_gradient_percent', 0),
                segment.get('elevation_change_m', 0),
                segment.get('min_curve_radius_m', 1000),
                segment.get('urban_percentage', 0),
                segment.get('water_crossings', 0),
                segment.get('protected_areas', 0),
                segment.get('soil_stability', 0.8),
                segment.get('seismic_risk', 0.1)
            ]
            features.append(segment_features)
        
        return np.array(features)
    
    @staticmethod
    def create_station_clustering_features(stations: List[Dict]) -> np.ndarray:
        """Create features for station clustering analysis"""
        features = []
        
        for station in stations:
            station_features = [
                station.get('lat', 0),
                station.get('lon', 0),
                station.get('daily_passengers', 0),
                station.get('platform_count', 2),
                station.get('services_count', 0),
                station.get('accessibility_score', 0.5),
                station.get('transfer_potential', 0.3),
                station.get('parking_capacity', 0),
                station.get('bike_facilities', 0),
                station.get('commercial_facilities', 0)
            ]
            features.append(station_features)
        
        return np.array(features)
    
    @staticmethod
    def predict_travel_demand(origin_features: np.ndarray, 
                            destination_features: np.ndarray,
                            distance_km: float,
                            ml_pipeline: MLPipeline,
                            model_name: str = "demand_predictor") -> float:
        """Predict travel demand between two locations"""
        
        # Combine origin, destination, and route features
        combined_features = np.concatenate([
            origin_features.flatten(),
            destination_features.flatten(),
            [distance_km, distance_km**2, 1/max(1, distance_km)]  # Distance transformations
        ]).reshape(1, -1)
        
        try:
            demand = ml_pipeline.predict(combined_features, model_name)[0]
            return max(0, demand)  # Ensure non-negative demand
        except:
            # Fallback to gravity model
            pop_origin = origin_features[0] if len(origin_features) > 0 else 10000
            pop_destination = destination_features[0] if len(destination_features) > 0 else 10000
            return (pop_origin * pop_destination) / (distance_km ** 2)
    
    @staticmethod
    def optimize_station_spacing(route_points: List[Tuple[float, float]],
                               demand_predictions: List[float],
                               ml_pipeline: MLPipeline,
                               target_stations: int = 10) -> List[int]:
        """Use ML to optimize station placement along route"""
        
        # Create features for each potential station location
        features = []
        for i, point in enumerate(route_points):
            point_features = [
                i / len(route_points),  # Position along route
                demand_predictions[i] if i < len(demand_predictions) else 0,
                # Distance to previous/next high-demand points
                RailwayMLUtils._calculate_demand_gradient(demand_predictions, i),
                # Local demand concentration
                RailwayMLUtils._calculate_local_demand_sum(demand_predictions, i, window=3)
            ]
            features.append(point_features)
        
        # Use clustering to find optimal station locations
        try:
            X = np.array(features)
            ml_pipeline.train_clusterer(X, "station_placement", n_clusters=target_stations)
            cluster_labels = ml_pipeline.predict(X, "station_placement")
            
            # Find representative points for each cluster
            station_indices = []
            for cluster_id in range(target_stations):
                cluster_points = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                if cluster_points:
                    # Choose point with highest demand in cluster
                    best_point = max(cluster_points, 
                                   key=lambda i: demand_predictions[i] if i < len(demand_predictions) else 0)
                    station_indices.append(best_point)
            
            return sorted(station_indices)
            
        except:
            # Fallback to regular spacing
            step = len(route_points) // target_stations
            return list(range(0, len(route_points), step))[:target_stations]
    
    @staticmethod
    def _calculate_demand_gradient(demands: List[float], index: int, window: int = 2) -> float:
        """Calculate demand gradient around a point"""
        if len(demands) <= 1:
            return 0.0
        
        # Calculate average demand before and after the point
        start_idx = max(0, index - window)
        end_idx = min(len(demands), index + window + 1)
        
        before_avg = np.mean(demands[start_idx:index]) if index > start_idx else 0
        after_avg = np.mean(demands[index+1:end_idx]) if index + 1 < end_idx else 0
        
        return abs(after_avg - before_avg)
    
    @staticmethod
    def _calculate_local_demand_sum(demands: List[float], index: int, window: int = 3) -> float:
        """Calculate sum of demand in local window"""
        start_idx = max(0, index - window)
        end_idx = min(len(demands), index + window + 1)
        
        return sum(demands[start_idx:end_idx])

# Ensemble methods for railway planning

class RailwayEnsemble:
    """Ensemble methods specifically designed for railway planning"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        
    def train_route_cost_ensemble(self, training_data: List[Dict], ml_pipeline: MLPipeline) -> Dict[str, float]:
        """Train ensemble of models for route cost prediction"""
        
        # Prepare features and targets
        X, y = ml_pipeline.prepare_features(training_data, target_column='construction_cost')
        
        # Train multiple models
        models_performance = {}
        
        # Random Forest
        rf_performance = ml_pipeline.train_regressor(X, y, "cost_rf", "random_forest")
        models_performance["cost_rf"] = rf_performance.r2_score or 0
        
        # Gradient Boosting
        gb_performance = ml_pipeline.train_regressor(X, y, "cost_gb", "gradient_boosting")
        models_performance["cost_gb"] = gb_performance.r2_score or 0
        
        # Calculate ensemble weights based on performance
        total_performance = sum(models_performance.values())
        if total_performance > 0:
            self.weights = {model: perf / total_performance 
                          for model, perf in models_performance.items()}
        else:
            # Equal weights if no valid performance scores
            self.weights = {model: 1.0 / len(models_performance) 
                          for model in models_performance.keys()}
        
        return models_performance
    
    def predict_route_cost(self, X: np.ndarray, ml_pipeline: MLPipeline) -> Tuple[float, Dict[str, float]]:
        """Make ensemble prediction for route cost"""
        
        predictions = {}
        ensemble_prediction = 0.0
        
        for model_name, weight in self.weights.items():
            try:
                pred = ml_pipeline.predict(X, model_name)[0]
                predictions[model_name] = pred
                ensemble_prediction += weight * pred
            except:
                continue
        
        return ensemble_prediction, predictions
    
    def train_demand_ensemble(self, training_data: List[Dict], ml_pipeline: MLPipeline) -> Dict[str, float]:
        """Train ensemble for passenger demand prediction"""
        
        X, y = ml_pipeline.prepare_features(training_data, target_column='daily_passengers')
        
        models_performance = {}
        
        # Multiple algorithms for demand prediction
        rf_performance = ml_pipeline.train_regressor(X, y, "demand_rf", "random_forest")
        models_performance["demand_rf"] = rf_performance.r2_score or 0
        
        gb_performance = ml_pipeline.train_regressor(X, y, "demand_gb", "gradient_boosting")
        models_performance["demand_gb"] = gb_performance.r2_score or 0
        
        # Calculate weights
        total_performance = sum(models_performance.values())
        if total_performance > 0:
            demand_weights = {model: perf / total_performance 
                            for model, perf in models_performance.items()}
        else:
            demand_weights = {model: 1.0 / len(models_performance) 
                            for model in models_performance.keys()}
        
        self.weights.update(demand_weights)
        return models_performance

# Advanced feature engineering for railway data

class RailwayFeatureEngineering:
    """Advanced feature engineering for railway planning"""
    
    @staticmethod
    def create_temporal_features(timestamp_data: List[str]) -> np.ndarray:
        """Create temporal features from timestamp data"""
        import datetime
        
        features = []
        for timestamp_str in timestamp_data:
            try:
                dt = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                
                temporal_features = [
                    dt.hour,  # Hour of day
                    dt.weekday(),  # Day of week
                    dt.month,  # Month
                    1 if dt.weekday() < 5 else 0,  # Is weekday
                    1 if 7 <= dt.hour <= 9 or 17 <= dt.hour <= 19 else 0,  # Is peak hour
                    dt.day,  # Day of month
                    1 if dt.month in [6, 7, 8] else 0,  # Summer season
                    1 if dt.month in [12, 1, 2] else 0,  # Winter season
                ]
                features.append(temporal_features)
            except:
                # Default values if parsing fails
                features.append([12, 2, 6, 1, 0, 15, 0, 0])
        
        return np.array(features)
    
    @staticmethod
    def create_network_features(station: Dict, all_stations: List[Dict]) -> np.ndarray:
        """Create network topology features for a station"""
        from railway_ai.utils.geo import haversine_distance
        
        # Calculate distances to all other stations
        distances = []
        for other_station in all_stations:
            if other_station != station:
                dist = haversine_distance(
                    station['lat'], station['lon'],
                    other_station['lat'], other_station['lon']
                )
                distances.append(dist)
        
        if not distances:
            return np.array([0, 0, 0, 0, 0, 0])
        
        distances = sorted(distances)
        
        network_features = [
            distances[0] if len(distances) > 0 else 0,  # Distance to nearest station
            np.mean(distances[:3]) if len(distances) >= 3 else np.mean(distances),  # Avg distance to 3 nearest
            len([d for d in distances if d <= 10]),  # Stations within 10km
            len([d for d in distances if d <= 25]),  # Stations within 25km
            len([d for d in distances if d <= 50]),  # Stations within 50km
            np.std(distances) if len(distances) > 1 else 0  # Distance variability
        ]
        
        return np.array(network_features)
    
    @staticmethod
    def create_route_complexity_features(elevation_profile: List[Dict]) -> np.ndarray:
        """Create features representing route complexity"""
        if len(elevation_profile) < 2:
            return np.array([0, 0, 0, 0, 0, 0, 0])
        
        elevations = [p['elevation'] for p in elevation_profile]
        distances = [p['distance_km'] for p in elevation_profile]
        
        # Calculate gradients
        gradients = []
        for i in range(1, len(elevation_profile)):
            rise = elevation_profile[i]['elevation'] - elevation_profile[i-1]['elevation']
            run = (elevation_profile[i]['distance_km'] - elevation_profile[i-1]['distance_km']) * 1000
            if run > 0:
                gradient = abs(rise / run) * 100
                gradients.append(gradient)
        
        # Calculate curvature (simplified)
        curvatures = []
        if len(elevation_profile) >= 3:
            for i in range(1, len(elevation_profile) - 1):
                # Approximate curvature using three points
                dx1 = distances[i] - distances[i-1]
                dx2 = distances[i+1] - distances[i]
                dy1 = elevations[i] - elevations[i-1]
                dy2 = elevations[i+1] - elevations[i]
                
                if dx1 > 0 and dx2 > 0:
                    slope1 = dy1 / (dx1 * 1000)
                    slope2 = dy2 / (dx2 * 1000)
                    curvature = abs(slope2 - slope1) / ((dx1 + dx2) / 2 * 1000)
                    curvatures.append(curvature)
        
        complexity_features = [
            max(elevations) - min(elevations),  # Total elevation change
            np.mean(gradients) if gradients else 0,  # Average gradient
            max(gradients) if gradients else 0,  # Maximum gradient
            np.std(gradients) if len(gradients) > 1 else 0,  # Gradient variability
            len([g for g in gradients if g > 2.0]),  # Count of steep sections
            np.mean(curvatures) if curvatures else 0,  # Average curvature
            max(curvatures) if curvatures else 0  # Maximum curvature
        ]
        
        return np.array(complexity_features)
    
    @staticmethod
    def create_economic_features(region_data: Dict) -> np.ndarray:
        """Create economic features for demand and cost modeling"""
        economic_features = [
            region_data.get('gdp_per_capita', 35000),  # GDP per capita
            region_data.get('unemployment_rate', 0.08),  # Unemployment rate
            region_data.get('population_density', 300),  # People per km²
            region_data.get('average_income', 45000),  # Average household income
            region_data.get('tourism_index', 0.5),  # Tourism activity index
            region_data.get('business_density', 0.3),  # Businesses per capita
            region_data.get('education_level', 0.6),  # Higher education percentage
            region_data.get('car_ownership', 0.7),  # Cars per household
            region_data.get('public_transport_usage', 0.4),  # PT usage rate
            region_data.get('cost_of_living_index', 1.0)  # Relative cost of living
        ]
        
        return np.array(economic_features)

# Model validation and testing utilities

class RailwayModelValidator:
    """Validation utilities for railway ML models"""
    
    @staticmethod
    def validate_demand_model(ml_pipeline: MLPipeline, test_data: List[Dict], 
                            model_name: str = "demand_predictor") -> Dict[str, float]:
        """Validate demand prediction model"""
        
        X_test, y_test = ml_pipeline.prepare_features(test_data, target_column='actual_demand')
        
        try:
            predictions = ml_pipeline.predict(X_test, model_name)
            
            # Calculate validation metrics
            mae = np.mean(np.abs(predictions - y_test))
            mse = np.mean((predictions - y_test) ** 2)
            rmse = np.sqrt(mse)
            
            # R² score
            ss_res = np.sum((y_test - predictions) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((y_test - predictions) / np.maximum(y_test, 1))) * 100
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'n_samples': len(y_test)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def cross_validate_models(ml_pipeline: MLPipeline, data: List[Dict], 
                            target_column: str, k_folds: int = 5) -> Dict[str, Dict]:
        """Perform k-fold cross-validation on all models"""
        
        X, y = ml_pipeline.prepare_features(data, target_column=target_column)
        
        # Split data into k folds
        fold_size = len(X) // k_folds
        results = {}
        
        for model_name in ml_pipeline.models.keys():
            fold_scores = []
            
            for fold in range(k_folds):
                # Create train/test split for this fold
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < k_folds - 1 else len(X)
                
                X_test_fold = X[start_idx:end_idx]
                y_test_fold = y[start_idx:end_idx]
                
                X_train_fold = np.concatenate([X[:start_idx], X[end_idx:]])
                y_train_fold = np.concatenate([y[:start_idx], y[end_idx:]])
                
                try:
                    # Train on fold training data
                    fold_model_name = f"{model_name}_fold_{fold}"
                    
                    if 'classifier' in model_name or 'classification' in model_name:
                        ml_pipeline.train_classifier(X_train_fold, y_train_fold, fold_model_name)
                        predictions = ml_pipeline.predict(X_test_fold, fold_model_name)
                        score = np.mean(predictions == y_test_fold)  # Accuracy
                    else:
                        ml_pipeline.train_regressor(X_train_fold, y_train_fold, fold_model_name)
                        predictions = ml_pipeline.predict(X_test_fold, fold_model_name)
                        score = r2_score(y_test_fold, predictions)  # R² score
                    
                    fold_scores.append(score)
                    
                except Exception as e:
                    print(f"Error in fold {fold} for model {model_name}: {e}")
                    continue
            
            if fold_scores:
                results[model_name] = {
                    'mean_score': np.mean(fold_scores),
                    'std_score': np.std(fold_scores),
                    'scores': fold_scores
                }
        
        return results
    
    @staticmethod
    def compare_model_performance(ml_pipeline: MLPipeline) -> pd.DataFrame:
        """Compare performance of all trained models"""
        
        performance_data = []
        
        for model_name, performance in ml_pipeline.performance_history.items():
            row = {
                'Model': model_name,
                'Type': performance.model_type,
                'Samples': performance.n_samples,
                'Training Time (s)': round(performance.training_time, 2)
            }
            
            if performance.accuracy is not None:
                row['Accuracy'] = round(performance.accuracy, 3)
            
            if performance.r2_score is not None:
                row['R² Score'] = round(performance.r2_score, 3)
            
            if performance.rmse is not None:
                row['RMSE'] = round(performance.rmse, 3)
            
            if performance.cross_val_score is not None:
                row['CV Score'] = round(performance.cross_val_score, 3)
            
            if performance.silhouette_score is not None:
                row['Silhouette'] = round(performance.silhouette_score, 3)
            
            performance_data.append(row)
        
        return pd.DataFrame(performance_data)

# Example usage and testing
def example_railway_ml_workflow():
    """Example workflow for railway ML pipeline"""
    
    # Initialize ML pipeline
    ml_pipeline = MLPipeline(random_state=42)
    
    # Sample training data for demand prediction
    demand_training_data = [
        {
            'population_1km': 5000, 'population_5km': 25000, 'commercial_density': 0.3,
            'transport_connections': 3, 'employment_centers': 2, 'daily_passengers': 1200
        },
        {
            'population_1km': 15000, 'population_5km': 80000, 'commercial_density': 0.7,
            'transport_connections': 5, 'employment_centers': 8, 'daily_passengers': 4500
        },
        {
            'population_1km': 2000, 'population_5km': 8000, 'commercial_density': 0.1,
            'transport_connections': 1, 'employment_centers': 1, 'daily_passengers': 300
        }
    ] * 20  # Replicate for more training data
    
    # Train demand prediction model
    X_demand, y_demand = ml_pipeline.prepare_features(
        demand_training_data, target_column='daily_passengers'
    )
    
    demand_performance = ml_pipeline.train_regressor(
        X_demand, y_demand, "demand_predictor", "random_forest"
    )
    
    print(f"Demand Model Performance:")
    print(f"  R² Score: {demand_performance.r2_score:.3f}")
    print(f"  RMSE: {demand_performance.rmse:.1f}")
    print(f"  Training Time: {demand_performance.training_time:.2f}s")
    
    # Feature importance
    importance = ml_pipeline.get_feature_importance("demand_predictor")
    print(f"\nTop 3 Important Features:")
    for i, feat in enumerate(importance[:3]):
        print(f"  {i+1}. {feat.feature_name}: {feat.importance:.3f}")
    
    # Make predictions on new data
    new_station_data = np.array([[8000, 40000, 0.5, 3, 4]])
    predicted_demand = ml_pipeline.predict(new_station_data, "demand_predictor")
    print(f"\nPredicted demand for new station: {predicted_demand[0]:.0f} passengers/day")
    
    # Save models
    ml_pipeline.save_models("data/models")
    print(f"\nModels saved to data/models/")
    
    return ml_pipeline

if __name__ == "__main__":
    # Run example workflow
    pipeline = example_railway_ml_workflow()