#!/usr/bin/env python3
"""
Quick setup and run script for BCPC Pipeline with HTML reports
"""

import os
import subprocess
import sys
from pathlib import Path

def create_html_generator():
    """Create the HTML report generator file"""
    
    html_generator_content = '''"""
HTML Report Generator for BCPC Pipeline
Creates beautiful interactive HTML reports
"""

import json
from pathlib import Path
import datetime
from typing import Dict, Any

def generate_html_report(pipeline_results: Dict[str, Any], output_path: str = "output/bcpc_report.html"):
    """Generate a comprehensive HTML report"""
    
    # Extract data
    cities = pipeline_results.get("data_loading", {}).get("cities", [])
    terrain_test = pipeline_results.get("terrain_test", {})
    parallel_test = pipeline_results.get("parallel_test", {})
    config = pipeline_results.get("config", {})
    
    # Calculate stats
    total_population = sum(city.get("population", 0) for city in cities)
    successful_terrain = parallel_test.get("successful", 0)
    total_cities = len(cities)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚂 BCPC Railway Pipeline Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 15px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; font-size: 2.5em; margin-bottom: 30px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .stat-number {{ font-size: 2em; font-weight: bold; }}
        .cities {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }}
        .city-card {{ background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; padding: 20px; border-radius: 10px; }}
        .terrain-results {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
        .terrain-card {{ padding: 15px; border-radius: 8px; text-align: center; }}
        .success {{ background: linear-gradient(135deg, #27ae60, #2ecc71); color: white; }}
        .warning {{ background: linear-gradient(135deg, #f39c12, #e67e22); color: white; }}
        .error {{ background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; }}
        .json-data {{ background: #2c3e50; color: #ecf0f1; padding: 20px; border-radius: 10px; overflow-x: auto; font-family: monospace; margin-top: 20px; }}
        .timestamp {{ text-align: center; color: #7f8c8d; margin-top: 30px; font-style: italic; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚂 BCPC Railway Pipeline Report</h1>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{total_cities}</div>
                <div>Cities Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{total_population:,}</div>
                <div>Total Population</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{successful_terrain}/{total_cities}</div>
                <div>Terrain Success</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{parallel_test.get('elapsed_time', 0):.1f}s</div>
                <div>Processing Time</div>
            </div>
        </div>
        
        <h2>🏙️ Lebanese Cities</h2>
        <div class="cities">"""
    
    # Add city cards
    for city in cities:
        html_content += f"""
            <div class="city-card">
                <h3>{city.get('name', 'Unknown')}</h3>
                <p>👥 Population: {city.get('population', 0):,}</p>
                <p>📍 Coords: {city.get('latitude', 0):.3f}, {city.get('longitude', 0):.3f}</p>
                <p>🏷️ Region: {city.get('region', 'Unknown')}</p>
            </div>"""
    
    html_content += """
        </div>
        
        <h2>🏔️ Terrain Analysis Results</h2>
        <div class="terrain-results">"""
    
    # Add terrain results
    if parallel_test.get('results'):
        for result in parallel_test['results']:
            status_class = result.get('status', 'unknown')
            html_content += f"""
            <div class="terrain-card {status_class}">
                <h3>{result.get('city', 'Unknown')}</h3>
                <p>Status: {result.get('status', 'Unknown').title()}</p>
                <p>Complexity: {result.get('complexity', 'Unknown').title()}</p>
                <p>Cost Factor: {result.get('cost_multiplier', 1.0):.1f}x</p>
            </div>"""
    else:
        html_content += '<p>⚠️ No terrain results available</p>'
    
    html_content += f"""
        </div>
        
        <h2>📊 Complete Results (JSON)</h2>
        <div class="json-data">
            <pre>{json.dumps(pipeline_results, indent=2, default=str)}</pre>
        </div>
        
        <div class="timestamp">
            Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by BCPC Parallel Pipeline
        </div>
    </div>
</body>
</html>"""
    
    # Write HTML file
    output_file = Path(output_path)
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML report generated: {output_file}")
    return str(output_file)

def create_report_from_json(json_path: str = "output/pipeline_test_summary.json", 
                           output_path: str = "output/bcpc_report.html"):
    """Create HTML report from existing JSON results"""
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        html_file = generate_html_report(data, output_path)
        return html_file
        
    except Exception as e:
        print(f"❌ Error generating HTML report: {e}")
        return None
'''
    
    # Write the HTML generator file
    html_gen_path = Path("pipeline/html_report_generator.py")
    with open(html_gen_path, 'w') as f:
        f.write(html_generator_content)
    
    print(f"✅ Created HTML generator: {html_gen_path}")

def main():
    print("🚂 BCPC Pipeline Setup & Run")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("pipeline").exists():
        print("❌ Please run from root directory (where pipeline/ exists)")
        return 1
    
    # Create HTML generator
    print("📄 Creating HTML report generator...")
    create_html_generator()
    
    # Run the pipeline
    print("🚀 Running parallel pipeline with HTML output...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "dagster", "asset", "materialize",
            "--select", "*",
            "--module-name", "pipeline.dagster_orchestrator"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Pipeline completed successfully!")
            
            # Generate HTML from JSON results
            print("📄 Generating HTML report...")
            
            try:
                # Import and use the HTML generator
                sys.path.append('pipeline')
                from html_report_generator import create_report_from_json
                
                html_file = create_report_from_json()
                if html_file:
                    print(f"🌐 HTML Report: {html_file}")
                    print("📱 Open this file in your browser!")
                    
                    # Try to open automatically
                    try:
                        import webbrowser
                        webbrowser.open(f"file://{Path(html_file).absolute()}")
                        print("🌐 Opened in browser automatically")
                    except:
                        print("📱 Manual: Open the HTML file in your browser")
                        
            except Exception as e:
                print(f"⚠️ HTML generation error: {e}")
            
        else:
            print("❌ Pipeline failed!")
            print("Output:", result.stdout[-500:])
            print("Errors:", result.stderr[-500:])
            
    except Exception as e:
        print(f"💥 Error running pipeline: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())