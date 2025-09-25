import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RailwayPostprocessor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.inverse_scalers = {}
        self.inverse_encoders = {}
        
    def process_predictions(self, predictions: Union[np.ndarray, Dict],
                          prediction_type: str) -> Dict[str, Any]:
        
        if prediction_type == 'route':
            return self._process_route_predictions(predictions)
        elif prediction_type == 'cost':
            return self._process_cost_predictions(predictions)
        elif prediction_type == 'timetable':
            return self._process_timetable_predictions(predictions)
        elif prediction_type == 'stations':
            return self._process_station_predictions(predictions)
        else:
            return {'raw_predictions': predictions}
    
    def _process_route_predictions(self, predictions: Union[np.ndarray, Dict]) -> Dict[str, Any]:
        if isinstance(predictions, dict):
            route_points = predictions.get('route', np.array([]))
            costs = predictions.get('cost', np.array([]))
            speeds = predictions.get('speed', np.array([]))
        else:
            route_points = predictions
            costs = np.array([])
            speeds = np.array([])
        
        processed = {
            'route': {
                'coordinates': self._convert_to_geojson(route_points),
                'length_km': self._calculate_route_length(route_points),
                'segments': self._segment_route(route_points)
            }
        }
        
        if costs.size > 0:
            processed['cost'] = {
                'total': float(np.sum(costs)),
                'per_km': float(np.mean(costs)),
                'breakdown': self._cost_breakdown(costs)
            }
        
        if speeds.size > 0:
            processed['speed'] = {
                'average': float(np.mean(speeds)),
                'maximum': float(np.max(speeds)),
                'minimum': float(np.min(speeds))
            }
        
        return processed
    
    def _process_cost_predictions(self, predictions: Union[np.ndarray, Dict]) -> Dict[str, Any]:
        if isinstance(predictions, dict):
            total_cost = predictions.get('total_cost', 0)
            cost_breakdown = predictions.get('cost_breakdown', {})
            uncertainty = predictions.get('uncertainty', 0)
        else:
            total_cost = float(predictions[0]) if predictions.size > 0 else 0
            cost_breakdown = {}
            uncertainty = 0
        
        processed = {
            'total_cost_usd': total_cost,
            'total_cost_millions': total_cost / 1e6,
            'confidence_interval': {
                'lower': total_cost * (1 - uncertainty),
                'upper': total_cost * (1 + uncertainty)
            },
            'breakdown': cost_breakdown,
            'currency': 'USD',
            'base_year': 2024
        }
        
        return processed
    
    def _process_timetable_predictions(self, predictions: Union[np.ndarray, Dict]) -> Dict[str, Any]:
        if isinstance(predictions, dict):
            departure_times = predictions.get('departure_times', np.array([]))
            frequencies = predictions.get('frequencies', np.array([]))
            dwell_times = predictions.get('dwell_times', np.array([]))
        else:
            departure_times = predictions
            frequencies = np.array([])
            dwell_times = np.array([])
        
        processed = {
            'schedule': []
        }
        
        for i, time in enumerate(departure_times):
            schedule_entry = {
                'departure': self._minutes_to_time(time),
                'frequency_per_day': int(frequencies[i]) if i < len(frequencies) else 1,
                'dwell_minutes': float(dwell_times[i]) if i < len(dwell_times) else 2
            }
            processed['schedule'].append(schedule_entry)
        
        return processed
    
    def _process_station_predictions(self, predictions: Union[np.ndarray, Dict]) -> Dict[str, Any]:
        if isinstance(predictions, dict):
            locations = predictions.get('locations', np.array([]))
            types = predictions.get('types', np.array([]))
            capacities = predictions.get('capacities', np.array([]))
        else:
            locations = predictions
            types = np.array([])
            capacities = np.array([])
        
        processed = {
            'stations': []
        }
        
        for i, loc in enumerate(locations):
            if len(loc) >= 2:
                station = {
                    'id': f'station_{i+1}',
                    'latitude': float(loc[0]),
                    'longitude': float(loc[1]),
                    'type': self._decode_station_type(types[i]) if i < len(types) else 'standard',
                    'daily_capacity': int(capacities[i] * 1000) if i < len(capacities) else 5000
                }
                processed['stations'].append(station)
        
        return processed
    
    def _convert_to_geojson(self, route_points: np.ndarray) -> Dict:
        if route_points.size == 0:
            return {}
        
        coordinates = []
        for point in route_points:
            if len(point) >= 2:
                coordinates.append([float(point[1]), float(point[0])])  # lon, lat for GeoJSON
        
        return {
            'type': 'LineString',
            'coordinates': coordinates
        }
    
    def _calculate_route_length(self, route_points: np.ndarray) -> float:
        if len(route_points) < 2:
            return 0.0
        
        total_length = 0
        for i in range(len(route_points) - 1):
            dist = self._haversine_distance(
                route_points[i][0], route_points[i][1],
                route_points[i+1][0], route_points[i+1][1]
            )
            total_length += dist
        
        return total_length
    
    def _segment_route(self, route_points: np.ndarray) -> List[Dict]:
        segments = []
        
        if len(route_points) < 2:
            return segments
        
        for i in range(len(route_points) - 1):
            segment = {
                'start': {
                    'lat': float(route_points[i][0]),
                    'lon': float(route_points[i][1])
                },
                'end': {
                    'lat': float(route_points[i+1][0]),
                    'lon': float(route_points[i+1][1])
                },
                'length_km': self._haversine_distance(
                    route_points[i][0], route_points[i][1],
                    route_points[i+1][0], route_points[i+1][1]
                )
            }
            
            if route_points.shape[1] > 2:
                segment['elevation_change'] = float(route_points[i+1][2] - route_points[i][2])
            
            segments.append(segment)
        
        return segments
    
    def _cost_breakdown(self, costs: np.ndarray) -> Dict[str, float]:
        total = float(np.sum(costs))
        
        return {
            'construction': total * 0.6,
            'land_acquisition': total * 0.15,
            'equipment': total * 0.15,
            'planning': total * 0.05,
            'contingency': total * 0.05
        }
    
    def _minutes_to_time(self, minutes: float) -> str:
        hours = int(minutes // 60) % 24
        mins = int(minutes % 60)
        return f"{hours:02d}:{mins:02d}"
    
    def _decode_station_type(self, type_code: int) -> str:
        station_types = {
            0: 'terminal',
            1: 'junction',
            2: 'standard',
            3: 'express',
            4: 'local'
        }
        return station_types.get(int(type_code), 'standard')
    
    def _haversine_distance(self, lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c

class OutputFormatter:
    def __init__(self, format_type: str = 'json'):
        self.format_type = format_type
        
    def format(self, data: Dict[str, Any]) -> Union[str, Dict, pd.DataFrame]:
        if self.format_type == 'json':
            return json.dumps(data, indent=2)
        elif self.format_type == 'dataframe':
            return self._to_dataframe(data)
        elif self.format_type == 'html':
            return self._to_html(data)
        elif self.format_type == 'markdown':
            return self._to_markdown(data)
        else:
            return data
    
    def _to_dataframe(self, data: Dict) -> pd.DataFrame:
        flat_data = self._flatten_dict(data)
        return pd.DataFrame([flat_data])
    
    def _to_html(self, data: Dict) -> str:
        df = self._to_dataframe(data)
        return df.to_html()
    
    def _to_markdown(self, data: Dict) -> str:
        lines = ["# Railway Prediction Results\n"]
        
        for key, value in data.items():
            lines.append(f"\n## {key.replace('_', ' ').title()}\n")
            
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    lines.append(f"- **{sub_key}**: {sub_value}")
            elif isinstance(value, list):
                for item in value:
                    lines.append(f"- {item}")
            else:
                lines.append(f"{value}")
        
        return '\n'.join(lines)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        items = []
        
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        
        return dict(items)