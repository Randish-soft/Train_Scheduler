#!/usr/bin/env python3
"""
BCPC Pipeline: Main Entry Point
Bring Cities Back to the People, not the Cars

This is the main orchestrator for the BCPC rail corridor feasibility pipeline.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import traceback

# Import all pipeline modules
from src.data_loader import DataLoader
from src.demand_analysis import DemandAnalyzer
from src.train_selection import TrainSelector
from src.terrain_analysis import TerrainAnalyzer
from src.route_mapping import RouteMapper
from src.route_optimizer import RouteOptimizer
from src.station_placement import StationPlacer
from src.cost_analysis import CostAnalyzer
from src.visualizer import Visualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bcpc_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BCPCPipeline:
    """Main pipeline orchestrator for BCPC rail corridor analysis."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.data_loader = DataLoader()
        self.demand_analyzer = DemandAnalyzer()
        self.train_selector = TrainSelector()
        self.terrain_analyzer = TerrainAnalyzer()
        self.route_mapper = RouteMapper()
        self.route_optimizer = RouteOptimizer()
        self.station_placer = StationPlacer()
        self.cost_analyzer = CostAnalyzer()
        self.visualizer = Visualizer()
        
        # Pipeline state
        self.pipeline_data = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'input_csv': 'input/cities.csv',
            'output_dir': 'output',
            'cache_dir': 'cache',
            'max_route_distance': 1000,  # km
            'min_population': 50000,
            'elevation_tolerance': 0.05,  # 5% grade max
            'cost_per_km_rail': 25000000,  # $25M per km
            'cost_per_station': 50000000,  # $50M per station
            'visualization_format': 'html'
        }
        
        if config_path and Path(config_path).exists():
            try:
                import json
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
                logger.info("Using default configuration")
        
        return default_config
    
    def run_pipeline(self, csv_path: str) -> Dict[str, Any]:
        """
        Run the complete BCPC pipeline.
        
        Args:
            csv_path: Path to the input CSV file
            
        Returns:
            Dictionary containing all pipeline results
        """
        try:
            logger.info("Starting BCPC Pipeline execution")
            
            # Step 1: Read and validate CSV data
            logger.info("Step 1: Loading and validating CSV data")
            self.pipeline_data['cities'] = self.data_loader.load_csv(csv_path)
            
            # Step 2: Understand demand sources
            logger.info("Step 2: Analyzing demand sources")
            self.pipeline_data['demand'] = self.demand_analyzer.analyze_demand(
                self.pipeline_data['cities']
            )
            
            # Step 3: Understand costs and train types
            logger.info("Step 3: Analyzing train types and costs")
            self.pipeline_data['trains'] = self.train_selector.get_available_trains()
            
            # Step 4: Understand terrain using open source
            logger.info("Step 4: Analyzing terrain data")
            self.pipeline_data['terrain'] = self.terrain_analyzer.analyze_terrain(
                self.pipeline_data['cities']
            )
            
            # Step 5: Map sample routes with tunnels and elevated sections
            logger.info("Step 5: Mapping potential routes")
            self.pipeline_data['routes'] = self.route_mapper.map_routes(
                self.pipeline_data['cities'],
                self.pipeline_data['terrain']
            )
            
            # Step 6: Optimize route with range fixes
            logger.info("Step 6: Optimizing routes")
            self.pipeline_data['optimized_routes'] = self.route_optimizer.optimize_routes(
                self.pipeline_data['routes'],
                self.pipeline_data['demand'],
                self.pipeline_data['trains']
            )
            
            # Step 7: Choose best route based on cost, time, and cities
            logger.info("Step 7: Selecting optimal route")
            self.pipeline_data['best_route'] = self.route_optimizer.select_best_route(
                self.pipeline_data['optimized_routes']
            )
            
            # Step 8: Choose best train type for route
            logger.info("Step 8: Selecting optimal train type")
            self.pipeline_data['selected_train'] = self.train_selector.select_optimal_train(
                self.pipeline_data['best_route'],
                self.pipeline_data['demand']
            )
            
            # Step 9: Choose best location/train station
            logger.info("Step 9: Placing stations optimally")
            self.pipeline_data['stations'] = self.station_placer.place_stations(
                self.pipeline_data['best_route'],
                self.pipeline_data['cities']
            )
            
            # Step 10: Cost analysis
            logger.info("Step 10: Performing cost analysis")
            self.pipeline_data['cost_analysis'] = self.cost_analyzer.analyze_costs(
                self.pipeline_data['best_route'],
                self.pipeline_data['selected_train'],
                self.pipeline_data['stations']
            )
            
            # Step 11: Visualize results
            logger.info("Step 11: Generating visualization")
            self.pipeline_data['visualization'] = self.visualizer.create_visualization(
                self.pipeline_data
            )
            
            # Save results
            self._save_results()
            
            logger.info("BCPC Pipeline completed successfully!")
            return self.pipeline_data
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _save_results(self):
        """Save pipeline results to output directory."""
        try:
            output_dir = Path(self.config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save summary results as JSON
            import json
            summary = {
                'best_route': self.pipeline_data.get('best_route', {}),
                'selected_train': self.pipeline_data.get('selected_train', {}),
                'cost_analysis': self.pipeline_data.get('cost_analysis', {}),
                'stations': self.pipeline_data.get('stations', [])
            }
            
            with open(output_dir / 'pipeline_results.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Results saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='BCPC Pipeline: Sketch City-to-City Rail Corridors'
    )
    parser.add_argument(
        'csv_file',
        help='Path to CSV file containing city data'
    )
    parser.add_argument(
        '--config',
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not Path(args.csv_file).exists():
        logger.error(f"Input CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    try:
        # Initialize and run pipeline
        pipeline = BCPCPipeline(args.config)
        pipeline.config['output_dir'] = args.output_dir
        
        results = pipeline.run_pipeline(args.csv_file)
        
        # Print summary
        print("\n" + "="*60)
        print("BCPC PIPELINE SUMMARY")
        print("="*60)
        
        if 'best_route' in results:
            route = results['best_route']
            print(f"Best Route: {route.get('name', 'Unknown')}")
            print(f"Distance: {route.get('distance_km', 0):.1f} km")
            print(f"Travel Time: {route.get('travel_time_hours', 0):.1f} hours")
        
        if 'selected_train' in results:
            train = results['selected_train']
            print(f"Selected Train: {train.get('name', 'Unknown')}")
            print(f"Max Speed: {train.get('max_speed_kmh', 0)} km/h")
            print(f"Capacity: {train.get('capacity', 0)} passengers")
        
        if 'cost_analysis' in results:
            costs = results['cost_analysis']
            print(f"Total Cost: ${costs.get('total_cost', 0):,.0f}")
            print(f"Cost per km: ${costs.get('cost_per_km', 0):,.0f}")
        
        print(f"\nDetailed results saved to: {args.output_dir}")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()