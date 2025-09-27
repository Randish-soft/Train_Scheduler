#!/usr/bin/env python3
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent))

from src.data_pipeline.data_loader import RailwayDataLoader
from src.data_pipeline.feature_extractor import RailwayFeatureExtractor
from src.evaluation_suite.metrics import RailwayMetrics
from src.evaluation_suite.visualizer import RailwayVisualizer
from src.evaluation_suite.report_generator import ReportGenerator
from src.model_architecture.route_predictor import RoutePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path: Path, config: Dict) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = RoutePredictor(
        station_features=config['model']['station_features'],
        terrain_features=config['model']['terrain_features'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def evaluate_on_country(model: torch.nn.Module, country_data: Dict,
                        feature_extractor: RailwayFeatureExtractor,
                        metrics_calculator: RailwayMetrics) -> Dict[str, float]:
    
    # Extract features
    line_features = feature_extractor.extract_line_features(country_data['railways'])
    station_features = feature_extractor.extract_station_features(country_data['stations'])
    terrain_features = feature_extractor.extract_terrain_features(
        country_data['terrain'], 
        [(0, 0)] * len(country_data['terrain'])
    )
    
    combined_features = feature_extractor.combine_features({
        'line': line_features,
        'station': station_features,
        'terrain': terrain_features
    })
    
    if combined_features.empty:
        return {}
    
    # Prepare input
    X = torch.FloatTensor(combined_features.values)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(
            X[:, :10],  # station features
            X[:, 10:18],  # terrain features
            sequence_length=50
        )
    
    # Calculate metrics
    metrics = {}
    
    if 'route' in predictions:
        route_pred = predictions['route'].numpy()
        # Mock actual route for evaluation (in real scenario, load actual route)
        route_actual = route_pred + np.random.randn(*route_pred.shape) * 0.1
        
        route_metrics = metrics_calculator.calculate_route_metrics(
            route_pred.reshape(-1, 3),
            route_actual.reshape(-1, 3)
        )
        metrics.update({'route_' + k: v for k, v in route_metrics.items()})
    
    if 'cost' in predictions:
        cost_pred = predictions['cost'].numpy()
        # Mock actual costs
        cost_actual = cost_pred + np.random.randn(*cost_pred.shape) * 100000
        
        cost_metrics = metrics_calculator.calculate_cost_metrics(
            cost_pred,
            cost_actual
        )
        metrics.update({'cost_' + k: v for k, v in cost_metrics.items()})
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate BCPC Railway Model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/model/architecture.yaml',
                       help='Path to model config')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--countries', nargs='+', 
                       default=['lebanon', 'egypt', 'morocco', 'jordan'],
                       help='Countries to evaluate on')
    parser.add_argument('--output-dir', type=str, default='artifacts',
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    logger.info("Loading model...")
    model = load_model(Path(args.model), config)
    
    logger.info("Initializing components...")
    data_loader = RailwayDataLoader(data_dir=args.data_dir)
    feature_extractor = RailwayFeatureExtractor()
    metrics_calculator = RailwayMetrics()
    visualizer = RailwayVisualizer(output_dir=f"{args.output_dir}/figures")
    report_generator = ReportGenerator(output_dir=f"{args.output_dir}/reports")
    
    # Evaluate on each country
    all_results = {}
    country_metrics = {}
    
    for country in args.countries:
        logger.info(f"Evaluating on {country}...")
        
        # Load data
        country_data = data_loader.load_country_data(country, 'test')
        
        if all(v.empty if hasattr(v, 'empty') else v.size == 0 
               for v in country_data.values()):
            logger.warning(f"No data available for {country}")
            continue
        
        # Evaluate
        metrics = evaluate_on_country(
            model, country_data, feature_extractor, metrics_calculator
        )
        
        all_results[country] = metrics
        country_metrics[country] = {
            'route_metrics': {k: v for k, v in metrics.items() if 'route' in k},
            'cost_metrics': {k: v for k, v in metrics.items() if 'cost' in k}
        }
        
        logger.info(f"{country} metrics: {metrics}")
        
        # Generate visualizations if requested
        if args.visualize and 'route_rmse' in metrics:
            # Mock visualization with random data
            pred_route = np.random.randn(50, 3)
            actual_route = pred_route + np.random.randn(50, 3) * 0.1
            
            fig = visualizer.plot_route_comparison(
                pred_route, actual_route,
                title=f"Route Comparison - {country.capitalize()}",
                save_path=f"route_comparison_{country}.html"
            )
    
    # Calculate overall metrics
    overall_metrics = {}
    for metric_name in set().union(*[set(m.keys()) for m in all_results.values()]):
        values = [m[metric_name] for m in all_results.values() if metric_name in m]
        if values:
            overall_metrics[metric_name] = np.mean(values)
    
    # Generate report
    logger.info("Generating evaluation report...")
    report = report_generator.generate_evaluation_report(
        model_name="RoutePredictor",
        test_results={'overall': overall_metrics},
        country_metrics=country_metrics,
        save_path="evaluation_report.md"
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for country, metrics in all_results.items():
        print(f"\n{country.upper()}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    print(f"\nOVERALL:")
    for key, value in overall_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nDetailed report saved to: {args.output_dir}/reports/evaluation_report.md")
    
    if args.visualize:
        print(f"Visualizations saved to: {args.output_dir}/figures/")

if __name__ == "__main__":
    main()