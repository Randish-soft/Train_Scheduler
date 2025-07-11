"""
Dagster Orchestrator for BCPC Pipeline
=====================================

Parallel orchestration layer for the existing BCPC pipeline modules.
Uses ThreadPoolExecutor to parallelize terrain analysis calls.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import sys

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from dagster import (
    asset, 
    AssetExecutionContext,
    Config,
    get_dagster_logger,
    Output,
    MaterializeResult,
)

# Import your existing pipeline modules
try:
    from data_loader import DataLoader, CityData
    from terrain_analysis import TerrainAnalyzer
    from demand_analysis import DemandAnalyzer  
    from cost_analysis import CostAnalyzer
    from station_placement import StationOptimizer
    from route_optimizer import RouteOptimizer
    from route_mapping import RouteMapper
    from visualizer import PipelineVisualizer
    
    logger = logging.getLogger(__name__)
    logger.info("✅ Successfully imported all BCPC modules")
    
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Failed to import BCPC modules: {e}")
    logger.info("📁 Current directory: " + str(current_dir))
    logger.info("📁 Src directory: " + str(src_dir))
    logger.info("📁 Python path: " + str(sys.path))
    
    # Create dummy classes for testing
    class DataLoader:
        def load_csv(self, path): return []
        def validate_city_data(self, cities): return cities
    
    class CityData:
        def __init__(self): pass
    
    class TerrainAnalyzer:
        def __init__(self, **kwargs): pass
        def analyze_route_terrain(self, **kwargs): return None

class BCPCConfig(Config):
    """Configuration for BCPC Pipeline"""
    csv_file_path: str = "input/lebanon_cities_2024.csv"
    output_dir: str = "output"
    max_workers: int = 4  # Parallel terrain analysis workers
    verbose: bool = True

# ===== BASIC TEST ASSETS =====

@asset
def test_environment(context: AssetExecutionContext, config: BCPCConfig):
    """Test the environment and file structure"""
    logger = get_dagster_logger()
    logger.info("🚂 Testing BCPC Pipeline Environment")
    
    # Check current working directory
    cwd = Path.cwd()
    logger.info(f"📁 Current directory: {cwd}")
    
    # Check for key files and directories
    checks = {
        "CSV file": Path(config.csv_file_path),
        "Pipeline src": current_dir / "src",
        "Data cache": Path("data/_cache"),
        "Output dir": Path(config.output_dir)
    }
    
    results = {}
    for name, path in checks.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        logger.info(f"{status} {name}: {path} ({'exists' if exists else 'missing'})")
        results[name] = {"path": str(path), "exists": exists}
    
    # List files in src directory if it exists
    src_path = current_dir / "src"
    if src_path.exists():
        src_files = [f.name for f in src_path.glob("*.py")]
        logger.info(f"📄 Found {len(src_files)} Python files in src: {src_files}")
        results["src_files"] = src_files
    
    return results

@asset 
def load_city_data(context: AssetExecutionContext, config: BCPCConfig, test_environment):
    """Load and validate city data"""
    logger = get_dagster_logger()
    logger.info("📊 Loading city data from CSV")
    
    try:
        # Check if CSV file exists
        csv_path = Path(config.csv_file_path)
        if not csv_path.exists():
            logger.error(f"❌ CSV file not found: {csv_path}")
            return {"error": "CSV file not found", "path": str(csv_path)}
        
        # Try to load with pandas first (simpler)
        import pandas as pd
        df = pd.read_csv(csv_path)
        logger.info(f"✅ Loaded CSV with pandas: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"📊 Columns: {list(df.columns)}")
        
        # Convert to simple format for now
        cities_data = []
        for _, row in df.iterrows():
            city_info = {
                'name': row.get('city_name', 'Unknown'),  # Fixed: use city_name column
                'population': row.get('population', 0),
                'latitude': row.get('latitude', 0.0),
                'longitude': row.get('longitude', 0.0),
                'region': row.get('region', row.get('city_id', 'Unknown'))  # Use city_id as region fallback
            }
            cities_data.append(city_info)
        
        logger.info(f"✅ Processed {len(cities_data)} cities")
        return {"cities": cities_data, "count": len(cities_data)}
        
    except Exception as e:
        logger.error(f"❌ Error loading city data: {e}")
        return {"error": str(e)}

@asset
def test_terrain_analysis(context: AssetExecutionContext, config: BCPCConfig, load_city_data):
    """Test terrain analysis with a single city"""
    logger = get_dagster_logger()
    logger.info("🏔️  Testing terrain analysis")
    
    if "error" in load_city_data:
        logger.error("❌ Cannot test terrain - city data failed to load")
        return {"error": "City data not available"}
    
    cities = load_city_data.get("cities", [])
    if not cities:
        logger.error("❌ No cities available for terrain testing")
        return {"error": "No cities available"}
    
    # Test with first city
    test_city = cities[0]
    logger.info(f"🎯 Testing terrain analysis with: {test_city['name']}")
    
    try:
        # Create a simple terrain analyzer
        terrain_analyzer = TerrainAnalyzer(
            cache_dir="data/_cache/terrain",
            preferred_resolution=250.0
        )
        
        # Create a simple route around the city for testing
        from shapely.geometry import Point, LineString
        import math
        
        lat, lon = test_city['latitude'], test_city['longitude']
        radius_deg = 0.01  # Small radius for testing
        
        # Create a simple 4-point route
        coords = [
            (lon - radius_deg, lat - radius_deg),
            (lon + radius_deg, lat - radius_deg),
            (lon + radius_deg, lat + radius_deg),
            (lon - radius_deg, lat + radius_deg),
            (lon - radius_deg, lat - radius_deg)  # Close the loop
        ]
        
        test_route = LineString(coords)
        logger.info(f"📍 Created test route: {len(coords)} points, {test_route.length:.6f} degrees long")
        
        # Analyze terrain (this might take time due to API calls)
        logger.info("⏳ Starting terrain analysis (this may take 30-60 seconds)...")
        terrain_result = terrain_analyzer.analyze_route_terrain(
            route_line=test_route,
            buffer_km=1.0
        )
        
        if terrain_result:
            logger.info(f"✅ Terrain analysis successful!")
            logger.info(f"🏔️  Complexity: {terrain_result.overall_complexity.value}")
            logger.info(f"💰 Cost multiplier: {terrain_result.cost_multiplier:.1f}x")
            logger.info(f"📏 Route length: {terrain_result.elevation_profile.total_length_km:.1f} km")
            
            return {
                "status": "success",
                "city": test_city['name'],
                "complexity": terrain_result.overall_complexity.value,
                "cost_multiplier": terrain_result.cost_multiplier,
                "route_length_km": terrain_result.elevation_profile.total_length_km
            }
        else:
            logger.warning("⚠️  Terrain analysis returned None")
            return {"status": "warning", "message": "Terrain analysis returned None"}
            
    except Exception as e:
        logger.error(f"❌ Terrain analysis failed: {e}")
        return {"status": "error", "error": str(e)}

@asset
def parallel_terrain_test(
    context: AssetExecutionContext, 
    config: BCPCConfig,
    load_city_data
):
    """Test parallel terrain analysis with multiple cities"""
    logger = get_dagster_logger()
    logger.info("⚡ Testing parallel terrain analysis")
    
    if "error" in load_city_data:
        logger.error("❌ Cannot run parallel test - city data failed")
        return {"error": "City data not available"}
    
    cities = load_city_data.get("cities", [])
    if len(cities) < 2:
        logger.warning("⚠️  Need at least 2 cities for parallel testing")
        return {"warning": "Insufficient cities for parallel test"}
    
    # Test with first 3 cities (or all if fewer than 3)
    test_cities = cities[:min(3, len(cities))]
    logger.info(f"🎯 Testing parallel analysis with {len(test_cities)} cities")
    
    def analyze_single_city(city_info):
        """Analyze terrain for a single city"""
        try:
            from shapely.geometry import Point, LineString
            
            terrain_analyzer = TerrainAnalyzer(
                cache_dir="data/_cache/terrain",
                preferred_resolution=250.0
            )
            
            lat, lon = city_info['latitude'], city_info['longitude']
            radius_deg = 0.005  # Small radius for testing
            
            # Create test route
            coords = [
                (lon - radius_deg, lat),
                (lon + radius_deg, lat),
                (lon, lat + radius_deg),
                (lon, lat - radius_deg),
                (lon - radius_deg, lat)
            ]
            
            test_route = LineString(coords)
            result = terrain_analyzer.analyze_route_terrain(
                route_line=test_route,
                buffer_km=1.0
            )
            
            return {
                "city": city_info['name'],
                "status": "success",
                "complexity": result.overall_complexity.value if result else "unknown",
                "cost_multiplier": result.cost_multiplier if result else 1.0
            }
            
        except Exception as e:
            return {
                "city": city_info['name'],
                "status": "error",
                "error": str(e)
            }
    
    # Run parallel analysis
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_to_city = {
            executor.submit(analyze_single_city, city): city['name'] 
            for city in test_cities
        }
        
        for future in as_completed(future_to_city):
            city_name = future_to_city[future]
            try:
                result = future.result()
                results.append(result)
                
                status = "✅" if result['status'] == 'success' else "❌"
                logger.info(f"{status} {city_name}: {result.get('complexity', 'error')}")
                
            except Exception as e:
                logger.error(f"❌ Future failed for {city_name}: {e}")
                results.append({
                    "city": city_name,
                    "status": "error", 
                    "error": str(e)
                })
    
    elapsed_time = time.time() - start_time
    successful = sum(1 for r in results if r['status'] == 'success')
    
    logger.info(f"⚡ Parallel test completed in {elapsed_time:.1f}s")
    logger.info(f"✅ Success rate: {successful}/{len(test_cities)} cities")
    
    return {
        "results": results,
        "total_cities": len(test_cities),
        "successful": successful,
        "elapsed_time": elapsed_time,
        "success_rate": successful / len(test_cities) if test_cities else 0
    }

@asset
def pipeline_summary(
    context: AssetExecutionContext,
    config: BCPCConfig,
    test_environment,
    load_city_data,
    test_terrain_analysis,
    parallel_terrain_test
):
    """Generate summary of pipeline test results"""
    logger = get_dagster_logger()
    logger.info("📋 Generating pipeline summary")
    
    summary = {
        "pipeline_status": "BCPC Parallel Pipeline Test Results",
        "environment_check": test_environment,
        "data_loading": load_city_data,
        "terrain_test": test_terrain_analysis,
        "parallel_test": parallel_terrain_test,
        "config": {
            "csv_path": config.csv_file_path,
            "output_dir": config.output_dir,
            "max_workers": config.max_workers
        }
    }
    
    # Determine overall status
    checks = [
        test_environment.get("CSV file", {}).get("exists", False),
        "error" not in load_city_data,
        test_terrain_analysis.get("status") == "success",
        parallel_terrain_test.get("successful", 0) > 0
    ]
    
    passed_checks = sum(checks)
    overall_status = "🎉 READY" if passed_checks >= 3 else "⚠️  PARTIAL" if passed_checks >= 2 else "❌ NOT READY"
    
    logger.info(f"📊 Overall Status: {overall_status}")
    logger.info(f"✅ Passed checks: {passed_checks}/4")
    
    if passed_checks >= 3:
        logger.info("🚀 Pipeline is ready for full parallel processing!")
        logger.info("🎯 Next: Run the full dagster_orchestrator with all cities")
    else:
        logger.warning("⚠️  Pipeline needs attention before full deployment")
    
    summary["overall_status"] = overall_status
    summary["passed_checks"] = f"{passed_checks}/4"
    
    # Save summary to file
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "pipeline_test_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"📁 Summary saved to {config.output_dir}/pipeline_test_summary.json")
    
    # Generate HTML report
    try:
        # Import here to avoid issues if file doesn't exist
        import importlib.util
        html_generator_path = current_dir / "html_report_generator.py"
        
        if html_generator_path.exists():
            spec = importlib.util.spec_from_file_location("html_report_generator", html_generator_path)
            html_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(html_module)
            
            html_file = html_module.generate_html_report(summary, f"{config.output_dir}/bcpc_report.html")
            logger.info(f"🌐 HTML report generated: {html_file}")
            logger.info("📄 Open the HTML file in your browser to see the interactive report!")
        else:
            logger.info("⚠️ HTML generator not found - creating simple HTML report")
            # Create a simple HTML report inline
            simple_html = f"""
<!DOCTYPE html>
<html>
<head><title>BCPC Pipeline Report</title></head>
<body>
<h1>🚂 BCPC Pipeline Report</h1>
<h2>Summary</h2>
<p>Status: {summary.get('overall_status', 'Unknown')}</p>
<p>Cities: {len(summary.get('data_loading', {}).get('cities', []))}</p>
<p>Terrain Success: {summary.get('parallel_test', {}).get('successful', 0)}</p>
<h2>Raw Data</h2>
<pre>{json.dumps(summary, indent=2, default=str)}</pre>
</body>
</html>
"""
            html_path = Path(config.output_dir) / "bcpc_report.html"
            with open(html_path, 'w') as f:
                f.write(simple_html)
            logger.info(f"📄 Simple HTML report: {html_path}")
            
    except Exception as e:
        logger.warning(f"⚠️ Could not generate HTML report: {e}")
    
    return summary