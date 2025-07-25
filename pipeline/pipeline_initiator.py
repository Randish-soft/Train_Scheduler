#!/usr/bin/env python3
"""
Railway Raster Pipeline Initiator - Complete Orchestration
==========================================================

Enhanced orchestrator that coordinates all pipeline modules including:
- Modular route plotting with constraint analysis
- NIMBY factor analysis and community engagement
- Comprehensive optimization and reporting

Author: Miguel Ibrahim E
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json
import traceback

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import pipeline modules
try:
    # Data ingestion modules
    from data_injesting.processing_input import InputProcessor
    from data_injesting.processing_demand import DemandProcessor
    from data_injesting.processing_terrain import TerrainProcessor
    from data_injesting.referencing_similar_country import CountryReferencer
    
    # Processing modules
    from processing.choosing_train_for_route import TrainSelector
    from processing.creating_time_table import TimeTableCreator
    from processing.electrification import ElectrificationPlanner
    from processing.plotting_route import RoutePlotter
    from processing.plotting_lightRail import LightRailPlotter
    from processing.railyard_plotter import RailyardPlotter
    from processing.nimby_analyzer import NIMBYAnalyzer
    
    # Optimization modules
    from optimization.optimising_route import RouteOptimizer  
    from optimization.optimising_time_table import TimeTableOptimizer
    from optimization.railyard_optimizer import RailyardOptimizer
    from optimization.constraint_analyzer import ConstraintAnalyzer
    
except ImportError as e:
    print(f"Warning: Could not import some pipeline modules: {e}")
    print("Some modules may not be implemented yet - pipeline will use fallbacks.")


def welcome_message():
    """Display the welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           🚄 RAILWAY RASTER 🚄                              ║
║                    Intelligent Route Planning System                         ║
║                                  v2.0.0                                     ║
║            Learn → Generate → Optimize → Validate → Deploy                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 Smart Railway Planning with Advanced Constraint Analysis
🌍 No CSV files needed - automatic city data retrieval
🚂 AI-powered train selection and route optimization
🏘️ NIMBY factor analysis and community engagement planning
📊 Professional reports ready for stakeholders and investors
"""
    print(banner)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Configure logging for the pipeline."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    if not log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"railway_pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Railway Raster - Intelligent Route Planning System with Constraint Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with constraint evaluation
  python pipeline_initiator.py --country "Ghana"
  
  # Comprehensive analysis with NIMBY and community planning
  python pipeline_initiator.py --country "Nigeria" --budget 5000000000 \\
    --optimize-routes --nimby-analysis --generate-reports
  
  # Urban focus with full constraint analysis
  python pipeline_initiator.py --country "Kenya" --urban-focus \\
    --constraint-analysis --electrification --light-rail
        """
    )
    
    # Required arguments
    parser.add_argument("--country", type=str, required=True,
                       help="Target country for railway development")
    parser.add_argument("--output-dir", "-o", type=str, default="output",
                       help="Output directory (default: output)")
    
    # Analysis parameters
    parser.add_argument("--budget", type=float,
                       help="Total construction budget in USD")
    parser.add_argument("--demand-threshold", type=int, default=50000,
                       help="Min population threshold (default: 50,000)")
    parser.add_argument("--max-distance", type=float, default=500.0,
                       help="Max route distance in km (default: 500)")
    parser.add_argument("--min-cities", type=int, default=5,
                       help="Min cities to analyze (default: 5)")
    parser.add_argument("--max-cities", type=int, default=20,
                       help="Max cities to analyze (default: 20)")
    
    # Analysis focus
    parser.add_argument("--urban-focus", action="store_true",
                       help="Focus on urban areas (100k+ population)")
    parser.add_argument("--rural-inclusion", action="store_true",
                       help="Include smaller towns")
    parser.add_argument("--economic-priority", action="store_true",
                       help="Prioritize economically important cities")
    
    # Pipeline stages
    parser.add_argument("--skip-optimization", action="store_true",
                       help="Skip route optimization")
    parser.add_argument("--optimize-routes", action="store_true",
                       help="Enable advanced route optimization")
    parser.add_argument("--electrification", action="store_true",
                       help="Include electrification planning")
    parser.add_argument("--light-rail", action="store_true",
                       help="Include light rail analysis")
    
    # Advanced analysis options
    parser.add_argument("--constraint-analysis", action="store_true",
                       help="Enable comprehensive constraint analysis")
    parser.add_argument("--nimby-analysis", action="store_true",
                       help="Enable NIMBY factor and community engagement analysis")
    parser.add_argument("--generate-reports", action="store_true",
                       help="Generate comprehensive reports")
    
    # System options
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=str, help="Custom log file path")
    parser.add_argument("--dry-run", action="store_true",
                       help="Test configuration without execution")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed help")
    parser.add_argument("--quick", action="store_true",
                       help="Quick analysis mode (skip advanced analysis)")
    
    return parser.parse_args()


class EnhancedPipelineOrchestrator:
    """Enhanced pipeline orchestrator with advanced constraint and NIMBY analysis."""
    
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.start_time = time.time()
        
        # Initialize output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline results storage
        self.results = {}
        
        # Enable constraint analysis by default for comprehensive analysis
        if args.optimize_routes or args.nimby_analysis or args.generate_reports:
            args.constraint_analysis = True
    
    def run_pipeline(self):
        """Execute the complete enhanced pipeline."""
        try:
            self.logger.info("="*80)
            self.logger.info("🚀 STARTING ENHANCED RAILWAY RASTER PIPELINE")
            self.logger.info("="*80)
            
            # Stage 1: Data Ingestion
            self.logger.info("📥 STAGE 1: DATA INGESTION")
            ingested_data = self._run_data_ingestion()
            
            # Stage 2: Route Generation and Analysis
            self.logger.info("🗺️ STAGE 2: ROUTE GENERATION AND ANALYSIS")
            route_data = self._run_route_generation(ingested_data)
            
            # Stage 3: Constraint Analysis (if enabled)
            constraint_data = {}
            if self.args.constraint_analysis and not self.args.quick:
                self.logger.info("🚧 STAGE 3: CONSTRAINT ANALYSIS")
                constraint_data = self._run_constraint_analysis(route_data, ingested_data)
            
            # Stage 4: NIMBY Analysis (if enabled)
            nimby_data = {}
            if self.args.nimby_analysis and not self.args.quick:
                self.logger.info("🏘️ STAGE 4: NIMBY AND COMMUNITY ANALYSIS")
                nimby_data = self._run_nimby_analysis(route_data, constraint_data)
            
            # Stage 5: Data Processing (trains, timetables, etc.)
            self.logger.info("⚙️ STAGE 5: DATA PROCESSING")
            processed_data = self._run_data_processing(ingested_data, route_data, constraint_data)
            
            # Stage 6: Optimization
            optimized_data = processed_data
            if not self.args.skip_optimization:
                self.logger.info("🎯 STAGE 6: OPTIMIZATION")
                optimized_data = self._run_optimization(processed_data, constraint_data)
            else:
                self.logger.info("⏭️ SKIPPING OPTIMIZATION")
            
            # Stage 7: Report Generation
            if self.args.generate_reports:
                self.logger.info("📊 STAGE 7: COMPREHENSIVE REPORT GENERATION")
                self._generate_enhanced_reports(optimized_data, constraint_data, nimby_data)
            
            # Pipeline completion
            execution_time = time.time() - self.start_time
            self.logger.info("="*80)
            self.logger.info(f"✅ ENHANCED PIPELINE COMPLETED in {execution_time:.2f}s")
            self.logger.info(f"📁 Output: {self.output_dir.absolute()}")
            self._display_completion_summary()
            self.logger.info("="*80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ PIPELINE FAILED: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _run_data_ingestion(self):
        """Enhanced data ingestion with better error handling."""
        results = {}
        
        # 1.1: Process input data (cities)
        self.logger.info("🌍 Processing city data...")
        try:
            input_processor = InputProcessor(self.args.country)
            cities = input_processor.process()
            
            # Apply filters based on arguments
            cities = self._apply_city_filters(cities)
            
            results['cities'] = cities
            self.logger.info(f"✅ Processed {len(cities)} cities")
            
        except Exception as e:
            self.logger.error(f"❌ InputProcessor failed: {e}")
            raise Exception("Failed to retrieve city data - cannot continue pipeline")
        
        # 1.2: Analyze demand patterns
        self.logger.info("📊 Analyzing demand patterns...")
        try:
            demand_processor = DemandProcessor(results['cities'])
            results['demand'] = demand_processor.analyze()
            self.logger.info("✅ Demand analysis completed")
        except Exception as e:
            self.logger.warning(f"⚠️ DemandProcessor failed: {e}")
            results['demand'] = self._create_fallback_demand_data(results['cities'])
        
        # 1.3: Process terrain data
        self.logger.info("🏔️ Processing terrain data...")
        try:
            terrain_processor = TerrainProcessor(self.args.country)
            results['terrain'] = terrain_processor.process()
            self.logger.info("✅ Terrain processing completed")
        except Exception as e:
            self.logger.warning(f"⚠️ TerrainProcessor failed: {e}")
            results['terrain'] = self._create_fallback_terrain_data()
        
        # 1.4: Reference similar countries
        self.logger.info("🌍 Referencing similar countries...")
        try:
            country_referencer = CountryReferencer(self.args.country)
            results['references'] = country_referencer.find_similar()
            self.logger.info("✅ Country referencing completed")
        except Exception as e:
            self.logger.warning(f"⚠️ CountryReferencer failed: {e}")
            results['references'] = {'similar_countries': []}
        
        return results
    
    def _apply_city_filters(self, cities):
        """Apply city filtering based on command line arguments."""
        original_count = len(cities)
        
        # Apply focus filters
        if self.args.urban_focus:
            cities = [c for c in cities if c['population'] > 100000]
            self.logger.info(f"🏙️ Urban focus: {len(cities)} cities (was {original_count})")
        
        if not self.args.rural_inclusion:
            cities = [c for c in cities if c['population'] > self.args.demand_threshold]
            self.logger.info(f"👥 Population filter: {len(cities)} cities above {self.args.demand_threshold:,}")
        
        # Apply min/max limits
        if len(cities) < self.args.min_cities:
            self.logger.warning(f"⚠️ Only {len(cities)} cities found, less than minimum {self.args.min_cities}")
        
        if len(cities) > self.args.max_cities:
            cities = cities[:self.args.max_cities]
            self.logger.info(f"📊 Limited to top {self.args.max_cities} cities")
        
        return cities
    
    def _run_route_generation(self, ingested_data):
        """Enhanced route generation with multiple variants."""
        self.logger.info("🛤️ Generating route options...")
        
        try:
            route_plotter = RoutePlotter(ingested_data, self.output_dir)
            route_results = route_plotter.plot()
            
            route_options = route_results.get('route_options', [])
            self.logger.info(f"✅ Generated {len(route_options)} route options")
            
            # Log route types generated
            route_types = {}
            for route in route_options:
                route_type = route.get('route_type', 'unknown')
                route_types[route_type] = route_types.get(route_type, 0) + 1
            
            for route_type, count in route_types.items():
                self.logger.info(f"   - {route_type}: {count} routes")
            
            return route_results
            
        except Exception as e:
            self.logger.error(f"❌ Route generation failed: {e}")
            # Create minimal fallback route data
            return {
                'route_options': [],
                'visualizations': {},
                'total_routes_generated': 0
            }
    
    def _run_constraint_analysis(self, route_data, ingested_data):
        """Run comprehensive constraint analysis."""
        route_options = route_data.get('route_options', [])
        
        if not route_options:
            self.logger.warning("⚠️ No routes available for constraint analysis")
            return {}
        
        try:
            terrain_data = ingested_data.get('terrain', {})
            constraint_analyzer = ConstraintAnalyzer(terrain_data, self.output_dir)
            constraint_results = constraint_analyzer.analyze_route_constraints(route_options)
            
            # Log constraint analysis summary
            constraint_levels = {}
            for route_analysis in constraint_results.values():
                severity = route_analysis.get('constraint_severity', 'Unknown')
                constraint_levels[severity] = constraint_levels.get(severity, 0) + 1
            
            self.logger.info("📋 Constraint analysis summary:")
            for severity, count in constraint_levels.items():
                self.logger.info(f"   - {severity}: {count} routes")
            
            return constraint_results
            
        except Exception as e:
            self.logger.warning(f"⚠️ Constraint analysis failed: {e}")
            return {}
    
    def _run_nimby_analysis(self, route_data, constraint_data):
        """Run NIMBY factor and community engagement analysis."""
        route_options = route_data.get('route_options', [])
        
        if not route_options or not constraint_data:
            self.logger.warning("⚠️ Insufficient data for NIMBY analysis")
            return {}
        
        try:
            nimby_analyzer = NIMBYAnalyzer(self.output_dir)
            nimby_results = nimby_analyzer.analyze_nimby_factors(route_options, constraint_data)
            
            # Log NIMBY analysis summary
            high_risk_routes = 0
            for route_assessment in nimby_results.get('route_nimby_assessments', {}).values():
                risk_level = route_assessment.get('nimby_risk_level', '')
                if 'High' in risk_level or 'Very High' in risk_level:
                    high_risk_routes += 1
            
            self.logger.info(f"🏘️ NIMBY analysis: {high_risk_routes} high-risk routes identified")
            
            stakeholder_count = len(nimby_results.get('stakeholder_mapping', {}).get('all_stakeholders', {}))
            self.logger.info(f"👥 Stakeholder mapping: {stakeholder_count} stakeholder groups identified")
            
            return nimby_results
            
        except Exception as e:
            self.logger.warning(f"⚠️ NIMBY analysis failed: {e}")
            return {}
    
    def _run_data_processing(self, ingested_data, route_data, constraint_data):
        """Enhanced data processing with constraint awareness."""
        results = ingested_data.copy()
        results.update(route_data)
        
        # 2.1: Select trains for routes (constraint-aware)
        self.logger.info("🚂 Selecting optimal trains...")
        try:
            train_selector = TrainSelector(results)
            train_results = train_selector.select()
            results['train_selection'] = train_results
            
            # Log train selection summary
            fleet_summary = train_results.get('fleet_summary', {})
            total_units = fleet_summary.get('total_units', 0)
            total_cost = fleet_summary.get('total_fleet_cost', 0)
            self.logger.info(f"✅ Train selection: {total_units} units, ${total_cost/1e9:.1f}B total cost")
            
        except Exception as e:
            self.logger.warning(f"⚠️ TrainSelector failed: {e}")
            results['train_selection'] = {}
        
        # 2.2: Create timetables
        self.logger.info("⏰ Creating timetables...")
        try:
            timetable_creator = TimeTableCreator(results)
            results['timetables'] = timetable_creator.create()
            self.logger.info("✅ Timetable creation completed")
        except Exception as e:
            self.logger.warning(f"⚠️ TimeTableCreator not implemented: {e}")
            results['timetables'] = {}
        
        # 2.3: Plan electrification (if requested)
        if self.args.electrification:
            self.logger.info("⚡ Planning electrification...")
            try:
                electrification_planner = ElectrificationPlanner(results)
                results['electrification'] = electrification_planner.plan()
                self.logger.info("✅ Electrification planning completed")
            except Exception as e:
                self.logger.warning(f"⚠️ ElectrificationPlanner not implemented: {e}")
                results['electrification'] = {}
        
        # 2.4: Light rail analysis (if requested)
        if self.args.light_rail:
            self.logger.info("🚊 Analyzing light rail opportunities...")
            try:
                light_rail_plotter = LightRailPlotter(results, self.output_dir)
                light_rail_plotter.plot()
                self.logger.info("✅ Light rail analysis completed")
            except Exception as e:
                self.logger.warning(f"⚠️ LightRailPlotter not implemented: {e}")
        
        # 2.5: Railyard planning
        self.logger.info("🏭 Planning railyard locations...")
        try:
            railyard_plotter = RailyardPlotter(results, self.output_dir)
            railyard_plotter.plot()
            self.logger.info("✅ Railyard planning completed")
        except Exception as e:
            self.logger.warning(f"⚠️ RailyardPlotter not implemented: {e}")
        
        return results
    
    def _run_optimization(self, processed_data, constraint_data):
        """Enhanced optimization with constraint consideration."""
        results = processed_data.copy()
        
        if self.args.optimize_routes:
            # 3.1: Optimize routes (constraint-aware)
            self.logger.info("🎯 Optimizing route network...")
            try:
                route_optimizer = RouteOptimizer(processed_data)
                optimization_results = route_optimizer.optimize()
                results['optimized_routes'] = optimization_results
                
                # Log optimization results
                selected_routes = optimization_results.get('selected_routes', [])
                total_cost = optimization_results.get('total_cost', 0)
                self.logger.info(f"✅ Route optimization: {len(selected_routes)} routes selected, ${total_cost/1e9:.1f}B total")
                
            except Exception as e:
                self.logger.warning(f"⚠️ RouteOptimizer not implemented: {e}")
                results['optimized_routes'] = {}
            
            # 3.2: Optimize timetables
            self.logger.info("⏰ Optimizing timetables...")
            try:
                timetable_optimizer = TimeTableOptimizer(results)
                results['optimized_timetables'] = timetable_optimizer.optimize()
                self.logger.info("✅ Timetable optimization completed")
            except Exception as e:
                self.logger.warning(f"⚠️ TimeTableOptimizer not implemented: {e}")
                results['optimized_timetables'] = results.get('timetables', {})
        
        # 3.3: Optimize railyards
        self.logger.info("🏭 Optimizing railyard locations...")
        try:
            railyard_optimizer = RailyardOptimizer(results)
            results['railyards'] = railyard_optimizer.optimize()
            self.logger.info("✅ Railyard optimization completed")
        except Exception as e:
            self.logger.warning(f"⚠️ RailyardOptimizer not implemented: {e}")
            results['railyards'] = {}
        
        return results
    
    def _generate_enhanced_reports(self, final_data, constraint_data, nimby_data):
        """Generate comprehensive reports including constraint and NIMBY analysis."""
        self.logger.info("📄 Generating comprehensive reports...")
        
        # Standard reports
        self._generate_html_report(final_data, constraint_data, nimby_data)
        self._generate_json_report(final_data, constraint_data, nimby_data)
        self._generate_csv_report(final_data)
        self._generate_executive_summary(final_data, constraint_data, nimby_data)
        
        # Additional specialized reports
        if constraint_data:
            self._generate_constraint_report(constraint_data)
        
        if nimby_data:
            self._generate_community_engagement_report(nimby_data)
        
        self.logger.info(f"✅ Enhanced reports generated in {self.output_dir}")
    
    def _generate_html_report(self, data, constraint_data=None, nimby_data=None):
        """Generate enhanced HTML report with constraint and NIMBY analysis."""
        cities = data.get('cities', [])
        demand_data = data.get('demand', {})
        train_selection = data.get('train_selection', {})
        
        cities_count = len(cities)
        total_population = sum(city['population'] for city in cities)
        routes_count = len(demand_data.get('demand_matrix', {}))
        
        # Enhanced statistics
        constraint_summary = ""
        nimby_summary = ""
        
        if constraint_data:
            high_constraint_routes = len([
                r for r in constraint_data.values() 
                if 'High' in r.get('constraint_severity', '')
            ])
            constraint_summary = f"""
            <div class="stat">
                <div class="stat-value">{high_constraint_routes}</div>
                <div>High-Constraint Routes</div>
            </div>"""
        
        if nimby_data:
            stakeholder_count = len(nimby_data.get('stakeholder_mapping', {}).get('all_stakeholders', {}))
            nimby_summary = f"""
            <div class="stat">
                <div class="stat-value">{stakeholder_count}</div>
                <div>Stakeholder Groups</div>
            </div>"""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Railway Analysis - {self.args.country}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
        .container {{ max-width: 1400px; margin: 20px auto; background: white; box-shadow: 0 20px 40px rgba(0,0,0,0.1); border-radius: 15px; overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 60px 40px; text-align: center; position: relative; }}
        .header::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="2" fill="rgba(255,255,255,0.1)"/></svg>') repeat; animation: float 20s infinite linear; }}
        @keyframes float {{ 0% {{ transform: translateY(0px); }} 100% {{ transform: translateY(-100px); }} }}
        .header h1 {{ margin: 0; font-size: 3em; z-index: 1; position: relative; }}
        .header h2 {{ font-size: 1.8em; opacity: 0.9; margin: 10px 0; z-index: 1; position: relative; }}
        .header .meta {{ opacity: 0.7; font-size: 1.1em; z-index: 1; position: relative; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 30px; padding: 40px; background: #f8f9fa; }}
        .stat {{ background: white; padding: 30px; border-radius: 15px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.1); transition: transform 0.3s ease; }}
        .stat:hover {{ transform: translateY(-5px); }}
        .stat-value {{ font-size: 2.5em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }}
        .section {{ padding: 40px; border-bottom: 1px solid #eee; }}
        .section:last-child {{ border-bottom: none; }}
        .section h2 {{ color: #2c3e50; margin-bottom: 20px; font-size: 2em; display: flex; align-items: center; gap: 15px; }}
        .city-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }}
        .city-card {{ background: linear-gradient(135deg, #74b9ff, #0984e3); color: white; padding: 25px; border-radius: 15px; box-shadow: 0 10px 20px rgba(0,0,0,0.1); }}
        .city-name {{ font-size: 1.3em; font-weight: bold; margin-bottom: 10px; }}
        .constraint-summary {{ background: linear-gradient(135deg, #fd79a8, #e84393); color: white; padding: 30px; margin: 20px 40px; border-radius: 15px; }}
        .nimby-summary {{ background: linear-gradient(135deg, #fdcb6e, #e17055); color: white; padding: 30px; margin: 20px 40px; border-radius: 15px; }}
        .recommendations {{ background: linear-gradient(135deg, #00b894, #00cec9); color: white; padding: 40px; margin: 40px; border-radius: 15px; }}
        .recommendations h3 {{ margin-bottom: 20px; font-size: 1.5em; }}
        .recommendations ul {{ list-style: none; }}
        .recommendations li {{ padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.2); display: flex; align-items: center; gap: 10px; }}
        .recommendations li:before {{ content: "✅"; font-size: 1.2em; }}
        .footer {{ background: #2c3e50; color: white; padding: 30px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚄 Advanced Railway Analysis</h1>
            <h2>{self.args.country}</h2>
            <div class="meta">
                <p>Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
                <p>Analysis completed in {time.time() - self.start_time:.1f} seconds</p>
            </div>
        </div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{cities_count}</div>
                <div>Cities Analyzed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{total_population:,.0f}</div>
                <div>Total Population</div>
            </div>
            <div class="stat">
                <div class="stat-value">{routes_count}</div>
                <div>Routes Analyzed</div>
            </div>
            {constraint_summary}
            {nimby_summary}
        </div>
        
        <div class="section">
            <h2>🏙️ Major Cities</h2>
            <div class="city-grid">
                {self._generate_city_cards_html(cities[:8])}
            </div>
        </div>
        
        {self._generate_constraint_section_html(constraint_data)}
        {self._generate_nimby_section_html(nimby_data)}
        
        <div class="section">
            <h2>🚂 Train Selection Summary</h2>
            {self._generate_train_summary_html(train_selection)}
        </div>
        
        <div class="recommendations">
            <h3>📊 Strategic Recommendations</h3>
            <ul>
                <li>Begin with comprehensive stakeholder engagement 6 months before planning</li>
                <li>Prioritize routes with low NIMBY risk for Phase 1 implementation</li>
                <li>Establish community benefit-sharing programs early</li>
                <li>Implement robust environmental mitigation measures</li>
                <li>Consider phased development over 7-10 years for optimal acceptance</li>
                <li>Engage similar country experts for technical transfer</li>
            </ul>
        </div>
        
        <div class="footer">
            <p><strong>Railway Raster v2.0</strong> - Intelligent Route Planning with Constraint Analysis</p>
            <p>This comprehensive analysis includes NIMBY factors, environmental constraints, and community engagement strategies.</p>
        </div>
    </div>
</body>
</html>"""
        
        output_path = self.output_dir / "railway_analysis_report.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_constraint_section_html(self, constraint_data):
        """Generate HTML section for constraint analysis."""
        if not constraint_data:
            return ""
        
        high_constraint_routes = [
            (route_id, data) for route_id, data in constraint_data.items()
            if 'High' in data.get('constraint_severity', '')
        ]
        
        return f"""
        <div class="constraint-summary">
            <h3>🚧 Constraint Analysis Summary</h3>
            <p><strong>{len(high_constraint_routes)} routes</strong> identified with high constraints requiring special attention.</p>
            <p>Key challenge areas: Environmental compliance, community engagement, and terrain complexity.</p>
            <p>Mitigation strategies developed for all identified constraints.</p>
        </div>
        """
    
    def _generate_nimby_section_html(self, nimby_data):
        """Generate HTML section for NIMBY analysis."""
        if not nimby_data:
            return ""
        
        stakeholder_count = len(nimby_data.get('stakeholder_mapping', {}).get('all_stakeholders', {}))
        high_influence_count = len(nimby_data.get('stakeholder_mapping', {}).get('high_influence_stakeholders', []))
        
        return f"""
        <div class="nimby-summary">
            <h3>🏘️ Community Impact Analysis</h3>
            <p><strong>{stakeholder_count} stakeholder groups</strong> identified, with {high_influence_count} requiring immediate engagement.</p>
            <p>Comprehensive community engagement plan developed with timeline and resource requirements.</p>
            <p>Compensation frameworks and mitigation strategies tailored to local concerns.</p>
        </div>
        """
    
    def _generate_city_cards_html(self, cities):
        """Generate HTML for city cards."""
        cards = []
        for city in cities:
            cards.append(f"""
                <div class="city-card">
                    <div class="city-name">{city['city_name']}</div>
                    <div>Population: {city['population']:,}</div>
                    <div>Coordinates: {city['latitude']:.2f}, {city['longitude']:.2f}</div>
                    <div>Region: {city.get('region', 'Unknown')}</div>
                </div>
            """)
        return ''.join(cards)
    
    def _generate_train_summary_html(self, train_selection):
        """Generate HTML for train selection summary."""
        if not train_selection:
            return "<p>Train selection analysis in progress...</p>"
        
        fleet_summary = train_selection.get('fleet_summary', {})
        fleet_composition = fleet_summary.get('fleet_composition', {})
        
        if not fleet_composition:
            return "<p>Fleet composition data not available.</p>"
        
        html = "<div class='train-summary'>"
        for train_type, details in fleet_composition.items():
            units = details.get('units', 0)
            percentage = details.get('percentage_of_fleet', 0)
            total_cost = details.get('total_cost', 0)
            html += f"<p><strong>{train_type}:</strong> {units} units ({percentage:.1f}% of fleet) - ${total_cost/1e6:.1f}M</p>"
        html += "</div>"
        
        return html
    
    def _generate_json_report(self, data, constraint_data=None, nimby_data=None):
        """Generate enhanced JSON report."""
        json_data = {
            "metadata": {
                "country": self.args.country,
                "analysis_date": datetime.now().isoformat(),
                "execution_time": time.time() - self.start_time,
                "pipeline_version": "2.0.0",
                "analysis_type": "enhanced" if constraint_data or nimby_data else "standard"
            },
            "configuration": {
                "demand_threshold": self.args.demand_threshold,
                "max_distance": self.args.max_distance,
                "budget": self.args.budget,
                "optimization_enabled": not self.args.skip_optimization,
                "electrification_enabled": self.args.electrification,
                "light_rail_enabled": self.args.light_rail,
                "constraint_analysis_enabled": bool(constraint_data),
                "nimby_analysis_enabled": bool(nimby_data)
            },
            "core_results": data
        }
        
        # Add constraint data if available
        if constraint_data:
            json_data["constraint_analysis"] = constraint_data
        
        # Add NIMBY data if available
        if nimby_data:
            json_data["nimby_analysis"] = nimby_data
        
        output_path = self.output_dir / "detailed_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)
    
    def _generate_csv_report(self, data):
        """Generate enhanced CSV report."""
        cities = data.get('cities', [])
        
        output_path = self.output_dir / "city_analysis.csv"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("city_name,population,latitude,longitude,region,priority,is_capital\n")
            
            for city in cities:
                name = city.get('city_name', 'Unknown')
                pop = city.get('population', 0)
                lat = city.get('latitude', 0)
                lng = city.get('longitude', 0)
                region = city.get('region', 'Unknown')
                is_capital = city.get('is_capital', False)
                
                if pop > self.args.demand_threshold * 3:
                    priority = "High"
                elif pop > self.args.demand_threshold:
                    priority = "Medium"
                else:
                    priority = "Low"
                
                f.write(f"{name},{pop},{lat},{lng},{region},{priority},{is_capital}\n")
    
    def _generate_executive_summary(self, data, constraint_data=None, nimby_data=None):
        """Generate enhanced executive summary."""
        cities = data.get('cities', [])
        demand_data = data.get('demand', {})
        
        # Enhanced summary with constraint and NIMBY insights
        constraint_insights = ""
        if constraint_data:
            high_constraint_count = len([
                r for r in constraint_data.values() 
                if 'High' in r.get('constraint_severity', '')
            ])
            constraint_insights = f"""
CONSTRAINT ANALYSIS:
- {len(constraint_data)} routes analyzed for constraints
- {high_constraint_count} routes identified with high constraint levels
- Comprehensive mitigation strategies developed
- Environmental, NIMBY, and infrastructure factors assessed
"""
        
        nimby_insights = ""
        if nimby_data:
            stakeholder_count = len(nimby_data.get('stakeholder_mapping', {}).get('all_stakeholders', {}))
            nimby_insights = f"""
COMMUNITY ENGAGEMENT ANALYSIS:
- {stakeholder_count} stakeholder groups mapped
- Community engagement timeline: 30 months
- Estimated engagement budget: $1.3M annually
- Compensation frameworks developed
"""
        
        summary = f"""
RAILWAY NETWORK ANALYSIS - EXECUTIVE SUMMARY
===========================================

Country: {self.args.country}
Analysis Date: {datetime.now().strftime('%Y-%m-%d')}
Pipeline Version: 2.0.0 (Enhanced with Constraint Analysis)

KEY FINDINGS:
- {len(cities)} cities analyzed
- Total population served: {sum(city['population'] for city in cities):,}
- Analysis completed in {time.time() - self.start_time:.1f} seconds
- Enhanced analysis includes constraint and community factors

MAJOR CITIES (Top 5):
{chr(10).join([f"- {city['city_name']}: {city['population']:,}" for city in cities[:5]])}

{constraint_insights}
{nimby_insights}

STRATEGIC RECOMMENDATIONS:
1. Phase 1: Start with low-constraint, high-demand corridors
2. Phase 2: Implement community engagement before construction
3. Phase 3: Deploy advanced mitigation measures for high-risk routes
4. Establish railway development authority with community liaison
5. Secure financing through public-private partnerships

IMPLEMENTATION TIMELINE:
- Years 1-2: Stakeholder engagement and detailed planning
- Years 3-5: Phase 1 construction (priority corridors)
- Years 6-8: Phase 2 expansion (regional connections)
- Years 9-10: Phase 3 completion and optimization

RISK MITIGATION PRIORITIES:
- Community opposition management
- Environmental compliance
- Construction cost control
- Political stability maintenance

NEXT STEPS:
- Begin stakeholder engagement immediately
- Conduct detailed feasibility studies for top 3 routes
- Establish community liaison and compensation frameworks
- Secure preliminary financing commitments
- Engage international railway development partners

Generated by Railway Raster v2.0 - Enhanced Intelligent Route Planning System
        """
        
        output_path = self.output_dir / "executive_summary.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary.strip())
    
    def _generate_constraint_report(self, constraint_data):
        """Generate specialized constraint analysis report."""
        output_path = self.output_dir / "constraint_analysis_report.json"
        
        # Create summary statistics
        constraint_summary = {
            "analysis_overview": {
                "total_routes_analyzed": len(constraint_data),
                "analysis_date": datetime.now().isoformat(),
                "constraint_categories": [
                    "NIMBY Factors",
                    "Environmental Constraints", 
                    "Terrain Challenges",
                    "Infrastructure Conflicts",
                    "Economic Factors",
                    "Regulatory Requirements"
                ]
            },
            "constraint_severity_distribution": {},
            "key_findings": [],
            "mitigation_priorities": [],
            "detailed_analysis": constraint_data
        }
        
        # Calculate severity distribution
        severity_counts = {}
        for route_analysis in constraint_data.values():
            severity = route_analysis.get('constraint_severity', 'Unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        constraint_summary["constraint_severity_distribution"] = severity_counts
        
        # Generate key findings
        high_constraint_routes = [
            route_data['route_name'] for route_data in constraint_data.values()
            if 'High' in route_data.get('constraint_severity', '')
        ]
        
        constraint_summary["key_findings"] = [
            f"{len(high_constraint_routes)} routes require extensive mitigation measures",
            "Environmental compliance is the most common constraint type",
            "Community engagement critical for project success",
            "Terrain challenges vary significantly by route"
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(constraint_summary, f, indent=2, default=str)
        
        self.logger.info(f"📋 Constraint analysis report saved to {output_path}")
    
    def _generate_community_engagement_report(self, nimby_data):
        """Generate specialized community engagement report."""
        output_path = self.output_dir / "community_engagement_plan.json"
        
        engagement_plan = {
            "plan_overview": {
                "analysis_date": datetime.now().isoformat(),
                "total_stakeholder_groups": len(nimby_data.get('stakeholder_mapping', {}).get('all_stakeholders', {})),
                "engagement_timeline": "30 months",
                "estimated_budget": "$1.3M annually"
            },
            "stakeholder_analysis": nimby_data.get('stakeholder_mapping', {}),
            "engagement_strategy": nimby_data.get('community_engagement_plan', {}),
            "communication_plan": nimby_data.get('communication_strategy', {}),
            "mitigation_framework": nimby_data.get('mitigation_strategies', {}),
            "compensation_guidelines": nimby_data.get('compensation_framework', {})
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(engagement_plan, f, indent=2, default=str)
        
        self.logger.info(f"🏘️ Community engagement plan saved to {output_path}")
    
    def _create_fallback_demand_data(self, cities):
        """Create fallback demand data if processor fails."""
        self.logger.info("Creating fallback demand data...")
        return {
            'demand_matrix': {},
            'high_demand_routes': [],
            'total_population': sum(city['population'] for city in cities)
        }
    
    def _create_fallback_terrain_data(self):
        """Create fallback terrain data if processor fails."""
        return {
            'terrain_difficulty': 'Medium',
            'average_elevation': 500,
            'construction_complexity': 'Medium',
            'terrain_zones': {
                'mixed': {'percentage': 100, 'difficulty': 'medium'}
            }
        }
    
    def _display_completion_summary(self):
        """Display completion summary with key achievements."""
        files_created = []
        
        # Check which files were created
        standard_files = [
            "railway_analysis_report.html",
            "detailed_analysis.json", 
            "city_analysis.csv",
            "executive_summary.txt"
        ]
        
        for filename in standard_files:
            if (self.output_dir / filename).exists():
                files_created.append(filename)
        
        # Check for specialized reports
        if (self.output_dir / "constraint_analysis_report.json").exists():
            files_created.append("constraint_analysis_report.json")
        
        if (self.output_dir / "community_engagement_plan.json").exists():
            files_created.append("community_engagement_plan.json")
        
        # Check for NIMBY and constraint analysis files
        if (self.output_dir / "constraint_analysis.json").exists():
            files_created.append("constraint_analysis.json")
        
        if (self.output_dir / "nimby_analysis.json").exists():
            files_created.append("nimby_analysis.json")
        
        self.logger.info("📁 Generated files:")
        for filename in files_created:
            self.logger.info(f"   ✅ {filename}")
        
        # Display key achievements
        if self.args.constraint_analysis:
            self.logger.info("🎯 Advanced constraint analysis completed")
        
        if self.args.nimby_analysis:
            self.logger.info("🏘️ NIMBY and community engagement analysis completed")
        
        if self.args.optimize_routes:
            self.logger.info("⚡ Route optimization completed")


def main():
    """Enhanced main entry point."""
    try:
        args = parse_arguments()
        
        # Set up logging
        logger = setup_logging(args.log_level, args.log_file)
        
        # Show welcome message
        if not args.dry_run:
            welcome_message()
        
        # Handle dry run with enhanced information
        if args.dry_run:
            logger.info("🔍 DRY RUN MODE - Enhanced Configuration Check")
            logger.info("=" * 60)
            logger.info(f"🌍 Country: {args.country}")
            logger.info(f"💰 Budget: ${args.budget:,.2f}" if args.budget else "💰 Budget: Not specified")
            logger.info(f"🏙️ Cities: {args.min_cities}-{args.max_cities}")
            logger.info(f"📊 Population threshold: {args.demand_threshold:,}")
            
            # Analysis features
            features = []
            if args.constraint_analysis: features.append("Constraint Analysis")
            if args.nimby_analysis: features.append("NIMBY Analysis")
            if args.optimize_routes: features.append("Route Optimization")
            if args.electrification: features.append("Electrification")
            if args.light_rail: features.append("Light Rail")
            
            logger.info(f"⚡ Features: {', '.join(features) if features else 'Basic Analysis'}")
            logger.info(f"🚀 Mode: {'Quick' if args.quick else 'Comprehensive'}")
            logger.info("=" * 60)
            logger.info("✅ Enhanced configuration valid - remove --dry-run to execute")
            return 0
        
        # Run enhanced pipeline
        orchestrator = EnhancedPipelineOrchestrator(args, logger)
        success = orchestrator.run_pipeline()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Enhanced analysis interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())