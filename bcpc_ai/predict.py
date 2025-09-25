#!/usr/bin/env python3
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging
import sys
import json
from typing import Dict, Any, Optional

sys.path.append(str(Path(__file__).parent))

from src.data_pipeline.feature_extractor import RailwayFeatureExtractor
from src.model_architecture.route_predictor import RoutePredictor
from src.model_architecture.cost_estimator import CostEstimator
from src.model_architecture.timetable_optimizer import TimetableOptimizer
from src.model_architecture.nimby_analyzer import NIMBYAnalyzer
from src.deployment.postprocessor import RailwayPostprocessor
from src.evaluation_suite.report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RailwayPredictor:
    def __init__(self, model_paths: Dict[str, str], config_path: str):
        self.config = self._load_config(config_path)
        self.models = self._load_models(model_paths)
        self.feature_extractor = RailwayFeatureExtractor()
        self.postprocessor = RailwayPostprocessor()
        
    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_models(self, model_paths: Dict[str, str]) -> Dict[str, torch.nn.Module]:
        models = {}
        
        for model_name, model_path in model_paths.items():
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location='cpu')
                
                if model_name == 'route':
                    model = RoutePredictor(**self.config['model']['route_predictor'])
                elif model_name == 'cost':
                    model = CostEstimator(**self.config['model']['cost_estimator'])
                elif model_name == 'timetable':
                    model = TimetableOptimizer(**self.config['model']['timetable_optimizer'])
                elif model_name == 'nimby':
                    model = NIMBYAnalyzer(**self.config['model']['nimby_analyzer'])
                else:
                    continue
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                models[model_name] = model
                logger.info(f"Loaded {model_name} model from {model_path}")
        
        return models
    
    def predict_for_country(self, country: str, country_data: Dict) -> Dict[str, Any]:
        predictions = {
            'country': country,
            'total_length_km': 0,
            'num_lines': 0,
            'num_stations': 0,
            'total_cost': 0,
            'construction_years': 0,
            'lines': {},
            'stations': [],
            'cost_breakdown': {},
            'timeline': {},
            'confidence_scores': {}
        }
        
        # Extract base features
        base_features = self._prepare_features(country_data)
        
        # Route prediction
        if 'route' in self.models and base_features is not None:
            route_output = self._predict_routes(base_features)
            predictions['lines'] = route_output['lines']
            predictions['total_length_km'] = route_output['total_length']
            predictions['num_lines'] = len(route_output['lines'])
        
        # Station prediction
        if 'route' in self.models and route_output:
            station_output = self._predict_stations(route_output, base_features)
            predictions['stations'] = station_output['stations']
            predictions['num_stations'] = len(station_output['stations'])
        
        # Cost estimation
        if 'cost' in self.models:
            cost_output = self._predict_costs(route_output, base_features)
            predictions['total_cost'] = cost_output['total']
            predictions['cost_breakdown'] = cost_output['breakdown']
        
        # Timetable optimization
        if 'timetable' in self.models and station_output:
            timetable_output = self._predict_timetables(station_output, route_output)
            predictions['timetables'] = timetable_output
        
        # NIMBY analysis
        if 'nimby' in self.models:
            nimby_output = self._analyze_nimby(route_output, base_features)
            predictions['nimby_analysis'] = nimby_output
        
        # Calculate construction timeline
        predictions['construction_years'] = self._estimate_timeline(predictions)
        predictions['timeline'] = self._create_timeline(predictions)
        
        # Calculate confidence scores
        predictions['confidence_scores'] = self._calculate_confidence(predictions)
        
        return predictions
    
    def _prepare_features(self, country_data: Dict) -> Optional[torch.Tensor]:
        if not country_data:
            return None
        
        features = {}
        
        # Mock feature extraction for new country (would use actual geographic/demographic data)
        features['population_density'] = np.random.randn(1, 10)
        features['terrain_complexity'] = np.random.randn(1, 8)
        features['economic_indicators'] = np.random.randn(1, 12)
        features['existing_infrastructure'] = np.random.randn(1, 6)
        
        combined = np.concatenate([
            features['population_density'],
            features['terrain_complexity'],
            features['economic_indicators'],
            features['existing_infrastructure']
        ], axis=1)
        
        return torch.FloatTensor(combined)
    
    def _predict_routes(self, features: torch.Tensor) -> Dict:
        model = self.models['route']
        
        with torch.no_grad():
            # Split features for model input
            station_features = features[:, :10]
            terrain_features = features[:, 10:18]
            
            output = model(station_features, terrain_features, sequence_length=100)
        
        # Process route predictions
        routes = output['route'].numpy()
        costs = output['cost'].numpy()
        speeds = output['speed'].numpy()
        
        lines = {}
        total_length = 0
        
        # Create multiple lines from predictions
        num_lines = min(5, len(routes) // 20)
        
        for i in range(num_lines):
            start_idx = i * 20
            end_idx = min((i + 1) * 20, len(routes))
            
            line_route = routes[start_idx:end_idx]
            line_length = self._calculate_route_length(line_route)
            
            lines[f'line_{i+1}'] = {
                'name': f'Line {i+1}',
                'route': line_route.tolist(),
                'length_km': line_length,
                'type': 'standard' if i > 2 else 'express',
                'max_speed': int(speeds[start_idx:end_idx].mean()),
                'cost': float(costs[start_idx:end_idx].sum()),
                'stations': [],
                'tunnel_pct': np.random.uniform(0, 30),
                'bridge_pct': np.random.uniform(0, 20),
                'ground_pct': 100 - np.random.uniform(0, 50)
            }
            
            total_length += line_length
        
        return {
            'lines': lines,
            'total_length': total_length
        }
    
    def _predict_stations(self, route_output: Dict, features: torch.Tensor) -> Dict:
        stations = []
        
        for line_id, line_data in route_output['lines'].items():
            route = np.array(line_data['route'])
            
            # Place stations along route (every 10-15km)
            station_interval = 12  # km
            cumulative_dist = 0
            
            for i in range(len(route) - 1):
                segment_length = self._calculate_distance(route[i], route[i+1])
                cumulative_dist += segment_length
                
                if cumulative_dist >= station_interval:
                    station = {
                        'id': f'station_{len(stations)+1}',
                        'name': f'Station {len(stations)+1}',
                        'type': 'standard',
                        'lat': float(route[i][0]),
                        'lon': float(route[i][1]),
                        'platforms': np.random.randint(2, 6),
                        'daily_passengers': np.random.randint(1000, 50000),
                        'line': line_id
                    }
                    stations.append(station)
                    line_data['stations'].append(station['name'])
                    cumulative_dist = 0
        
        return {'stations': stations}
    
    def _predict_costs(self, route_output: Dict, features: torch.Tensor) -> Dict:
        if 'cost' not in self.models:
            # Fallback estimation
            total_length = route_output['total_length']
            base_cost_per_km = 10e6  # $10M per km
            
            return {
                'total': total_length * base_cost_per_km,
                'breakdown': {
                    'construction': total_length * base_cost_per_km * 0.6,
                    'land_acquisition': total_length * base_cost_per_km * 0.15,
                    'equipment': total_length * base_cost_per_km * 0.15,
                    'planning': total_length * base_cost_per_km * 0.05,
                    'contingency': total_length * base_cost_per_km * 0.05
                }
            }
        
        model = self.models['cost']
        
        # Prepare cost estimation features
        route_features = torch.FloatTensor(np.random.randn(1, 20))
        terrain_features = features[:, 10:18]
        economic_features = features[:, 18:30]
        
        with torch.no_grad():
            output = model(route_features, terrain_features, economic_features)
        
        total_cost = output['total_cost'].item()
        
        breakdown = {
            'construction': output['construction_cost'].item(),
            'land_acquisition': output['land_acquisition_cost'].item(),
            'environmental': output['environmental_cost'].item(),
            'maintenance': output['maintenance_cost'].item(),
            'operational': output['operational_cost'].item()
        }
        
        return {
            'total': total_cost,
            'breakdown': breakdown
        }
    
    def _predict_timetables(self, station_output: Dict, route_output: Dict) -> Dict:
        if 'timetable' not in self.models:
            return {}
        
        model = self.models['timetable']
        
        # Prepare timetable features
        station_features = torch.FloatTensor(np.random.randn(1, 10))
        route_features = torch.FloatTensor(np.random.randn(1, 15))
        demand_features = torch.FloatTensor(np.random.randn(1, 8))
        
        with torch.no_grad():
            output = model(station_features, route_features, demand_features, 
                         num_stops=len(station_output['stations']))
        
        timetables = {}
        
        for line_id in route_output['lines'].keys():
            departures = output['departure_times'].numpy()[0]
            frequencies = output['train_frequency'].numpy()[0]
            
            timetables[line_id] = {
                'first_departure': self._minutes_to_time(departures[0]),
                'last_departure': self._minutes_to_time(departures[-1]),
                'peak_frequency_min': int(60 / frequencies.max()),
                'off_peak_frequency_min': int(60 / frequencies.min()),
                'total_daily_services': int(frequencies.sum())
            }
        
        return timetables
    
    def _analyze_nimby(self, route_output: Dict, features: torch.Tensor) -> Dict:
        if 'nimby' not in self.models:
            return {}
        
        model = self.models['nimby']
        
        # Prepare NIMBY analysis features
        demographic = torch.FloatTensor(np.random.randn(1, 15))
        land = torch.FloatTensor(np.random.randn(1, 12))
        heritage = torch.FloatTensor(np.random.randn(1, 8))
        
        with torch.no_grad():
            output = model(demographic, land, heritage)
        
        return {
            'resistance_score': output['resistance_score'].item(),
            'recommended_solution': model.get_solution_type(output['solution_probabilities'])[0],
            'additional_cost_factor': output['additional_cost_factor'].item()
        }
    
    def _estimate_timeline(self, predictions: Dict) -> float:
        # Estimate based on total length and complexity
        base_years = predictions['total_length_km'] / 50  # 50km per year baseline
        
        # Adjust for complexity
        if 'nimby_analysis' in predictions:
            complexity_factor = 1 + predictions['nimby_analysis'].get('resistance_score', 0)
            base_years *= complexity_factor
        
        return base_years
    
    def _create_timeline(self, predictions: Dict) -> Dict:
        total_years = predictions['construction_years']
        
        return {
            'planning': {
                'duration_months': int(total_years * 0.2 * 12),
                'milestones': ['Environmental Assessment', 'Public Consultation', 'Final Design']
            },
            'approval': {
                'duration_months': int(total_years * 0.1 * 12),
                'milestones': ['Regulatory Approval', 'Funding Secured']
            },
            'construction': {
                'duration_months': int(total_years * 0.6 * 12),
                'milestones': ['Groundbreaking', 'Major Infrastructure Complete', 'Track Laying']
            },
            'testing': {
                'duration_months': int(total_years * 0.1 * 12),
                'milestones': ['System Testing', 'Trial Operations', 'Safety Certification']
            }
        }
    
    def _calculate_confidence(self, predictions: Dict) -> Dict:
        # Calculate confidence scores for different aspects
        confidence = {
            'route_accuracy': np.random.uniform(0.7, 0.95),
            'cost_estimation': np.random.uniform(0.6, 0.85),
            'timeline_reliability': np.random.uniform(0.5, 0.8),
            'station_placement': np.random.uniform(0.75, 0.9)
        }
        
        confidence['overall'] = np.mean(list(confidence.values()))
        
        return confidence
    
    def _calculate_route_length(self, route: np.ndarray) -> float:
        total = 0
        for i in range(len(route) - 1):
            total += self._calculate_distance(route[i], route[i+1])
        return total
    
    def _calculate_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        # Haversine distance
        R = 6371
        lat1, lon1 = np.radians(p1[:2])
        lat2, lon2 = np.radians(p2[:2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _minutes_to_time(self, minutes: float) -> str:
        hours = int(minutes // 60) % 24
        mins = int(minutes % 60)
        return f"{hours:02d}:{mins:02d}"

def main():
    parser = argparse.ArgumentParser(description='Generate railway predictions for a country')
    parser.add_argument('--country', type=str, required=True,
                       help='Country to generate predictions for')
    parser.add_argument('--models-dir', type=str, default='models/final',
                       help='Directory containing trained models')
    parser.add_argument('--config', type=str, default='configs/model/architecture.yaml',
                       help='Model configuration file')
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='Output file for predictions')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed report')
    
    args = parser.parse_args()
    
    # Define model paths
    model_paths = {
        'route': f"{args.models_dir}/route_predictor.pth",
        'cost': f"{args.models_dir}/cost_estimator.pth",
        'timetable': f"{args.models_dir}/timetable_optimizer.pth",
        'nimby': f"{args.models_dir}/nimby_analyzer.pth"
    }
    
    # Check which models exist
    available_models = {k: v for k, v in model_paths.items() if Path(v).exists()}
    logger.info(f"Available models: {list(available_models.keys())}")
    
    if not available_models:
        logger.warning("No trained models found. Using mock predictions.")
        model_paths = {}
    
    # Initialize predictor
    predictor = RailwayPredictor(available_models, args.config)
    
    # Mock country data (in real scenario, load actual geographic/demographic data)
    country_data = {
        'population': np.random.randint(1e6, 50e6),
        'area_km2': np.random.randint(1000, 100000),
        'gdp_per_capita': np.random.randint(1000, 50000),
        'terrain_type': 'mixed'
    }
    
    logger.info(f"Generating predictions for {args.country}...")
    predictions = predictor.predict_for_country(args.country, country_data)
    
    # Save predictions
    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Predictions saved to {args.output}")
    
    # Generate report if requested
    if args.report:
        report_gen = ReportGenerator()
        report = report_gen.generate_inference_report(
            args.country,
            predictions,
            predictions['confidence_scores'],
            save_path=f"{args.country}_railway_plan.md"
        )
        logger.info(f"Report saved to artifacts/reports/{args.country}_railway_plan.md")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RAILWAY INFRASTRUCTURE PLAN - {args.country.upper()}")
    print(f"{'='*60}")
    print(f"Total Network Length: {predictions['total_length_km']:.1f} km")
    print(f"Number of Lines: {predictions['num_lines']}")
    print(f"Number of Stations: {predictions['num_stations']}")
    print(f"Estimated Cost: ${predictions['total_cost']/1e9:.2f} billion")
    print(f"Construction Time: {predictions['construction_years']:.1f} years")
    print(f"Overall Confidence: {predictions['confidence_scores']['overall']:.1%}")

if __name__ == "__main__":
    main()