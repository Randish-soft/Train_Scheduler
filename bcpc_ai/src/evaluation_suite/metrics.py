import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import torch
from scipy.spatial.distance import hausdorff_distance
import logging

logger = logging.getLogger(__name__)

class RailwayMetrics:
    def __init__(self):
        self.metrics_history = []
        
    def calculate_route_metrics(self, predicted: np.ndarray, 
                               actual: np.ndarray) -> Dict[str, float]:
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = mean_squared_error(actual, predicted)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(actual, predicted)
        metrics['r2'] = r2_score(actual, predicted)
        
        # Custom railway metrics
        if len(predicted.shape) >= 2 and predicted.shape[1] >= 2:
            metrics['hausdorff_distance'] = self._calculate_hausdorff(predicted, actual)
            metrics['frechet_distance'] = self._calculate_frechet(predicted, actual)
            metrics['route_similarity'] = self._calculate_route_similarity(predicted, actual)
        
        # Length difference
        pred_length = self._calculate_total_length(predicted)
        actual_length = self._calculate_total_length(actual)
        metrics['length_error_pct'] = abs(pred_length - actual_length) / actual_length * 100
        
        return metrics
    
    def calculate_cost_metrics(self, predicted_costs: np.ndarray,
                             actual_costs: np.ndarray) -> Dict[str, float]:
        metrics = {}
        
        metrics['cost_mae'] = mean_absolute_error(actual_costs, predicted_costs)
        metrics['cost_mape'] = np.mean(np.abs((actual_costs - predicted_costs) / actual_costs)) * 100
        
        # Budget overrun metrics
        overruns = predicted_costs > actual_costs
        metrics['overrun_percentage'] = np.mean(overruns) * 100
        metrics['avg_overrun_amount'] = np.mean(predicted_costs[overruns] - actual_costs[overruns]) if overruns.any() else 0
        
        # Underestimation metrics
        underruns = predicted_costs < actual_costs
        metrics['underrun_percentage'] = np.mean(underruns) * 100
        metrics['avg_underrun_amount'] = np.mean(actual_costs[underruns] - predicted_costs[underruns]) if underruns.any() else 0
        
        return metrics
    
    def calculate_timetable_metrics(self, predicted_times: pd.DataFrame,
                                   actual_times: pd.DataFrame) -> Dict[str, float]:
        metrics = {}
        
        if 'departure_time' in predicted_times.columns and 'departure_time' in actual_times.columns:
            time_diff = (predicted_times['departure_time'] - actual_times['departure_time']).dt.total_seconds() / 60
            metrics['avg_departure_deviation_min'] = np.mean(np.abs(time_diff))
            metrics['max_departure_deviation_min'] = np.max(np.abs(time_diff))
        
        if 'frequency' in predicted_times.columns and 'frequency' in actual_times.columns:
            freq_error = np.abs(predicted_times['frequency'] - actual_times['frequency'])
            metrics['frequency_mae'] = np.mean(freq_error)
            metrics['frequency_accuracy'] = np.mean(freq_error <= 2) * 100
        
        return metrics
    
    def calculate_station_metrics(self, predicted_stations: pd.DataFrame,
                                 actual_stations: pd.DataFrame) -> Dict[str, float]:
        metrics = {}
        
        if 'lat' in predicted_stations.columns and 'lon' in predicted_stations.columns:
            pred_coords = predicted_stations[['lat', 'lon']].values
            actual_coords = actual_stations[['lat', 'lon']].values
            
            # Average location error in km
            distances = []
            for pred, actual in zip(pred_coords, actual_coords):
                dist = self._haversine_distance(pred[0], pred[1], actual[0], actual[1])
                distances.append(dist)
            
            metrics['avg_station_location_error_km'] = np.mean(distances)
            metrics['max_station_location_error_km'] = np.max(distances)
            metrics['stations_within_1km'] = np.mean(np.array(distances) <= 1) * 100
            metrics['stations_within_5km'] = np.mean(np.array(distances) <= 5) * 100
        
        # Platform count accuracy
        if 'platforms' in predicted_stations.columns and 'platforms' in actual_stations.columns:
            platform_diff = np.abs(predicted_stations['platforms'] - actual_stations['platforms'])
            metrics['platform_count_mae'] = np.mean(platform_diff)
            metrics['platform_count_accuracy'] = np.mean(platform_diff == 0) * 100
        
        return metrics
    
    def calculate_efficiency_metrics(self, route_data: Dict) -> Dict[str, float]:
        metrics = {}
        
        if 'travel_time' in route_data and 'distance' in route_data:
            avg_speed = route_data['distance'] / route_data['travel_time']
            metrics['avg_speed_kmh'] = avg_speed
            metrics['efficiency_score'] = min(avg_speed / 160, 1.0) * 100  # 160 km/h as baseline
        
        if 'passenger_capacity' in route_data and 'actual_passengers' in route_data:
            utilization = route_data['actual_passengers'] / route_data['passenger_capacity']
            metrics['capacity_utilization'] = utilization * 100
            metrics['overcrowding_risk'] = max(0, (utilization - 0.85) / 0.15) * 100
        
        if 'energy_consumption' in route_data:
            metrics['energy_per_passenger_km'] = route_data['energy_consumption'] / (
                route_data['actual_passengers'] * route_data['distance']
            )
        
        return metrics
    
    def _calculate_hausdorff(self, pred: np.ndarray, actual: np.ndarray) -> float:
        try:
            return hausdorff_distance(pred, actual)
        except:
            return float('inf')
    
    def _calculate_frechet(self, pred: np.ndarray, actual: np.ndarray) -> float:
        # Simplified Frechet distance calculation
        n, m = len(pred), len(actual)
        dp = np.full((n, m), float('inf'))
        
        dp[0, 0] = np.linalg.norm(pred[0] - actual[0])
        
        for i in range(1, n):
            dp[i, 0] = max(dp[i-1, 0], np.linalg.norm(pred[i] - actual[0]))
        
        for j in range(1, m):
            dp[0, j] = max(dp[0, j-1], np.linalg.norm(pred[0] - actual[j]))
        
        for i in range(1, n):
            for j in range(1, m):
                dp[i, j] = max(
                    min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1]),
                    np.linalg.norm(pred[i] - actual[j])
                )
        
        return dp[n-1, m-1]
    
    def _calculate_route_similarity(self, pred: np.ndarray, actual: np.ndarray) -> float:
        # Calculate similarity based on direction and curvature
        if len(pred) < 2 or len(actual) < 2:
            return 0.0
        
        pred_vectors = np.diff(pred, axis=0)
        actual_vectors = np.diff(actual, axis=0)
        
        # Normalize vectors
        pred_norm = pred_vectors / (np.linalg.norm(pred_vectors, axis=1, keepdims=True) + 1e-8)
        actual_norm = actual_vectors / (np.linalg.norm(actual_vectors, axis=1, keepdims=True) + 1e-8)
        
        # Calculate average cosine similarity
        min_len = min(len(pred_norm), len(actual_norm))
        similarities = [np.dot(pred_norm[i], actual_norm[i]) for i in range(min_len)]
        
        return np.mean(similarities) * 100
    
    def _calculate_total_length(self, route: np.ndarray) -> float:
        if len(route) < 2:
            return 0.0
        
        total_length = 0
        for i in range(len(route) - 1):
            total_length += np.linalg.norm(route[i+1] - route[i])
        
        return total_length
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        aggregated = {}
        
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        
        for key in all_keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated