# File: Model/__main__.py
"""
Railway AI - Intelligent Railway Route Planning and Optimization
Main entry point for the railway intelligence system.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List
import logging

# Import all railway AI modules
from .config import RailwayConfig, LogLevel
from .learn import RailwayLearner
from .generate import RouteGenerator
from .test import ScenarioTester
from .grade import PlanGrader

def setup_logging(log_level: LogLevel = LogLevel.INFO, log_file: Optional[str] = None):
    """Setup logging configuration"""
    level_map = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.WARNING: logging.WARNING,
        LogLevel.ERROR: logging.ERROR
    }
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level_map[log_level],
        format=log_format,
        handlers=handlers
    )

def print_banner():
    """Print railway AI banner"""
    banner = """
    ╔════════════════════════════════════════════════════════════╗
    ║                    🚄 RAILWAY AI 🚄                       ║
    ║              Intelligent Route Planning System             ║
    ║                                                            ║
    ║  Learn → Generate → Optimize → Validate → Deploy          ║
    ╚════════════════════════════════════════════════════════════╝
    """
    print(banner)

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="Railway AI - Intelligent Railway Route Planning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Learn from German railway network
  python -m Model --mode learn --country germany --train-types "ICE,IC,S"
  
  # Generate route plan for Belgian cities
  python -m Model --mode generate --input cities.csv --country belgium
  
  # Test mountain crossing scenario
  python -m Model --mode test --scenario alpine_crossing
  
  # Grade generated plan
  python -m Model --mode grade --plan outputs/plan.json --metrics cost,feasibility
        """
    )
    
    # Core operation mode
    parser.add_argument(
        "--mode", 
        choices=["learn", "generate", "test", "grade"],
        required=True,
        help="Operation mode"
    )
    
    # Common arguments
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output-dir", default="data/outputs", help="Output directory")
    parser.add_argument("--log-level", choices=["debug", "info", "warning", "error"], 
                       default="info", help="Logging level")
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Learning mode arguments
    learn_group = parser.add_argument_group("Learning Mode")
    learn_group.add_argument("--country", help="Country to learn from")
    learn_group.add_argument("--train-types", default="S,IC,ICE", 
                           help="Train types to consider (comma-separated)")
    learn_group.add_argument("--focus", help="Learning focus areas (comma-separated)")
    learn_group.add_argument("--data-sources", help="Additional data sources")
    
    # Generation mode arguments
    gen_group = parser.add_argument_group("Generation Mode")
    gen_group.add_argument("--input", help="Input CSV file or city list")
    gen_group.add_argument("--optimize", default="cost,ridership", 
                          help="Optimization targets (comma-separated)")
    gen_group.add_argument("--constraints", help="Constraint file path")
    gen_group.add_argument("--route-name", help="Name for generated route")
    
    
    # Testing mode arguments
    test_group = parser.add_argument_group("Testing Mode")
    test_group.add_argument("--scenario", help="Test scenario name")
    test_group.add_argument("--test-suite", help="Test suite file")
    test_group.add_argument("--benchmark", action="store_true", 
                           help="Run benchmark tests")
    
    # Grading mode arguments
    grade_group = parser.add_argument_group("Grading Mode")
    grade_group.add_argument("--plan", help="Plan file to grade")
    grade_group.add_argument("--metrics", default="all", 
                            help="Grading metrics (comma-separated)")
    grade_group.add_argument("--reference", help="Reference solution for comparison")
    
    return parser

def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command line arguments"""
    errors = []
    
    # Mode-specific validation
    if args.mode == "learn":
        if not args.country:
            errors.append("--country is required for learn mode")
    
    elif args.mode == "generate":
        if not args.input:
            errors.append("--input is required for generate mode")
    
    
    elif args.mode == "test":
        if not args.scenario and not args.test_suite and not args.benchmark:
            errors.append("One of --scenario, --test-suite, or --benchmark is required for test mode")
    
    elif args.mode == "grade":
        if not args.plan:
            errors.append("--plan is required for grade mode")
    
    # Check file existence
    if args.config and not Path(args.config).exists():
        errors.append(f"Configuration file not found: {args.config}")
    
    if args.input and not Path(args.input).exists():
        errors.append(f"Input file not found: {args.input}")
    
    if args.plan and not Path(args.plan).exists():
        errors.append(f"Plan file not found: {args.plan}")
    
    # Print errors
    if errors:
        for error in errors:
            print(f"Error: {error}", file=sys.stderr)
        return False
    
    return True

def execute_learn_mode(args: argparse.Namespace, config: RailwayConfig) -> int:
    """Execute learning mode"""
    logger = logging.getLogger(__name__)
    logger.info(f"🧠 Learning from {args.country} railway network...")
    
    try:
        # Parse train types
        train_types = [t.strip() for t in args.train_types.split(",")]
        
        # Parse focus areas
        focus_areas = None
        if args.focus:
            focus_areas = [f.strip() for f in args.focus.split(",")]
        
        # Initialize learner
        learner = RailwayLearner(
            country=args.country,
            train_types=train_types,
            config=config
        )
        
        # Execute learning
        results = learner.execute(focus=focus_areas, data_sources=args.data_sources)
        
        # Print results summary
        logger.info("✅ Learning completed successfully!")
        
        # Access LearningResults dataclass attributes directly
        logger.info(f"📊 Learned from {results.stations_analyzed} stations")
        logger.info(f"🛤️  Analyzed {results.track_segments} track segments")
        logger.info(f"🏭 Identified {results.railyards_found} railyard patterns")
        logger.info(f"🚄 Analyzed {results.train_services_analyzed} train services")
        
        # Display quality metrics
        logger.info(f"\n📈 Learning Quality:")
        logger.info(f"  🎯 Data quality: {results.data_quality_score:.1%}")
        logger.info(f"  📡 Coverage: {results.coverage_completeness:.1%}")
        logger.info(f"  🔮 Learning confidence: {results.learning_confidence:.1%}")
        
        # Show model performance if available
        if results.model_accuracies:
            logger.info(f"\n🤖 Model Performance:")
            for model_name, accuracy in results.model_accuracies.items():
                logger.info(f"  • {model_name}: {accuracy:.3f}")
        
        # Show pattern statistics
        if results.station_patterns:
            logger.info(f"\n🚉 Station Patterns Learned:")
            pattern_count = len(results.station_patterns)
            logger.info(f"  • Total patterns: {pattern_count}")
            if 'feature_importance' in results.station_patterns:
                top_features = sorted(
                    results.station_patterns['feature_importance'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                logger.info(f"  • Top factors: {', '.join([f[0] for f in top_features])}")
        
        if results.track_patterns:
            logger.info(f"\n🛤️  Track Patterns Learned:")
            if 'cost_analysis' in results.track_patterns:
                logger.info(f"  • Cost models learned")
            if 'engineering_patterns' in results.track_patterns:
                logger.info(f"  • Engineering patterns identified")
        
        # Show key insights
        if results.key_insights:
            logger.info(f"\n💡 Key Insights:")
            for i, insight in enumerate(results.key_insights[:5], 1):
                logger.info(f"  {i}. {insight}")
        
        # Show recommendations
        if results.recommendations:
            logger.info(f"\n🎯 Recommendations:")
            for i, rec in enumerate(results.recommendations[:3], 1):
                logger.info(f"  {i}. {rec}")
        
        # Save models
        model_dir = Path(args.output_dir) / "models" / args.country.lower()
        model_dir.mkdir(parents=True, exist_ok=True)
        
        learner.save_models(str(model_dir))
        logger.info(f"\n💾 Models and patterns saved to {model_dir}")
        
        # Create summary file
        summary_file = model_dir / "learning_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Railway AI Learning Summary\n")
            f.write(f"=========================\n\n")
            f.write(f"Country: {results.country}\n")
            f.write(f"Learning Date: {results.learning_date}\n")
            f.write(f"Learning Time: {results.learning_time:.1f} seconds\n\n")
            f.write(f"Infrastructure Analyzed:\n")
            f.write(f"- Stations: {results.stations_analyzed}\n")
            f.write(f"- Track Segments: {results.track_segments}\n")
            f.write(f"- Railyards: {results.railyards_found}\n\n")
            f.write(f"Quality Metrics:\n")
            f.write(f"- Data Quality: {results.data_quality_score:.1%}\n")
            f.write(f"- Coverage: {results.coverage_completeness:.1%}\n")
            f.write(f"- Confidence: {results.learning_confidence:.1%}\n\n")
            
            if results.key_insights:
                f.write(f"Key Insights:\n")
                for insight in results.key_insights:
                    f.write(f"- {insight}\n")
                f.write("\n")
            
            if results.recommendations:
                f.write(f"Recommendations:\n")
                for rec in results.recommendations:
                    f.write(f"- {rec}\n")
        
        logger.info(f"📄 Summary saved to {summary_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Learning failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def execute_generate_mode(args: argparse.Namespace, config: RailwayConfig) -> int:
    """Execute route generation mode"""
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Generating route plan from {args.input}...")
    
    try:
        # Parse optimization targets
        optimization_targets = [t.strip() for t in args.optimize.split(",")]
        
        # Load constraints if provided
        constraints = None
        if args.constraints:
            import json
            with open(args.constraints, 'r') as f:
                constraints = json.load(f)
        
        # Initialize generator
        generator = RouteGenerator(config=config)
        
        # Execute generation
        route_plan = generator.create_plan(
            input_data=args.input,
            country=args.country,
            optimization_targets=optimization_targets,
            constraints=constraints,
            route_name=args.route_name
        )
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plan_file = output_dir / f"{route_plan.name}_{int(time.time())}.json"
        generator.save_plan(route_plan, str(plan_file))
        
        # Print results summary
        logger.info("✅ Route generation completed!")
        logger.info(f"📏 Total length: {route_plan.total_length_km:.1f} km")
        logger.info(f"🚉 Stations: {len(route_plan.stations)}")
        logger.info(f"💰 Estimated cost: €{route_plan.total_cost:,.0f}")
        logger.info(f"📁 Plan saved to: {plan_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Route generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def execute_test_mode(args: argparse.Namespace, config: RailwayConfig) -> int:
    """Execute testing mode"""
    logger = logging.getLogger(__name__)
    
    if args.benchmark:
        logger.info("🏁 Running benchmark tests...")
    elif args.scenario:
        logger.info(f"🧪 Testing scenario: {args.scenario}")
    else:
        logger.info(f"📋 Running test suite: {args.test_suite}")
    
    try:
        # Initialize tester
        tester = ScenarioTester(config=config)
        
        # Execute tests
        if args.benchmark:
            results = tester.run_benchmark()
        elif args.scenario:
            results = tester.run_scenario(args.scenario)
        else:
            results = tester.run_test_suite(args.test_suite)
        
        # Print results
        logger.info("✅ Testing completed!")
        logger.info(f"✅ Passed: {results.get('passed', 0)}")
        logger.info(f"❌ Failed: {results.get('failed', 0)}")
        logger.info(f"📊 Success rate: {results.get('success_rate', 0):.1%}")
        
        return 0 if results.get('failed', 0) == 0 else 1
        
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def execute_grade_mode(args: argparse.Namespace, config: RailwayConfig) -> int:
    """Execute plan grading mode"""
    logger = logging.getLogger(__name__)
    logger.info(f"📊 Grading plan: {args.plan}")
    
    try:
        # Parse metrics
        metrics = [m.strip() for m in args.metrics.split(",")]
        
        # Initialize grader
        grader = PlanGrader(config=config)
        
        # Execute grading
        grade_report = grader.evaluate_plan(
            plan_file=args.plan,
            metrics=metrics,
            reference_file=args.reference
        )
        
        # Print results
        logger.info("✅ Grading completed!")
        logger.info(f"🎯 Overall score: {grade_report.overall_score:.1f}/100")
        logger.info(f"💰 Cost efficiency: {grade_report.cost_score:.1f}/100")
        logger.info(f"🚄 Technical feasibility: {grade_report.feasibility_score:.1f}/100")
        logger.info(f"🌱 Environmental impact: {grade_report.environmental_score:.1f}/100")
        
        # Save detailed report
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = output_dir / f"grade_report_{int(time.time())}.json"
        grader.save_report(grade_report, str(report_file))
        logger.info(f"📁 Detailed report saved to: {report_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Grading failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def main() -> int:
    """Main entry point"""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Setup logging
    log_level = LogLevel[args.log_level.upper()]
    setup_logging(log_level, args.log_file)
    
    # Validate arguments
    if not validate_arguments(args):
        return 1
    
    # Load configuration
    try:
        if args.config:
            config = RailwayConfig.load_from_file(args.config)
        else:
            config = RailwayConfig()
        
        # Override config with command line arguments
        if args.output_dir:
            config.paths.output_dir = Path(args.output_dir)
        if args.verbose:
            config.logging.level = LogLevel.DEBUG
            
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1
    
    # Execute based on mode
    start_time = time.time()
    
    try:
        if args.mode == "learn":
            result = execute_learn_mode(args, config)
        elif args.mode == "generate":
            result = execute_generate_mode(args, config)
        
        elif args.mode == "test":
            result = execute_test_mode(args, config)
        elif args.mode == "grade":
            result = execute_grade_mode(args, config)
        else:
            print(f"Unknown mode: {args.mode}", file=sys.stderr)
            return 1
        
        # Print execution time
        execution_time = time.time() - start_time
        logger = logging.getLogger(__name__)
        logger.info(f"⏱️  Total execution time: {execution_time:.1f} seconds")
        
        return result
        
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        return 1
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"💥 Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())