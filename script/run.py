"""
run.py
Quick run script for the entire pipeline
"""
import subprocess
import sys
from pathlib import Path

def run_pipeline():
    """Run the main pipeline"""
    print("🚀 Running Train Scheduler Pipeline...\n")
    
    # Check if we're in the right directory
    if not Path("script/train_pipeline.py").exists():
        print("❌ Error: Must run from project root directory")
        return False
    
    # Run the pipeline
    result = subprocess.run([sys.executable, "script/train_pipeline.py"])
    
    if result.returncode != 0:
        print("\n❌ Pipeline failed!")
        return False
    
    return True

def run_visualization():
    """Run the visualization script"""
    print("\n\n🎨 Generating visualizations...\n")
    
    viz_script = Path("notebooks/visualize_stations.py")
    if not viz_script.exists():
        print("⚠️  Warning: Visualization script not found")
        return False
    
    result = subprocess.run([sys.executable, str(viz_script)])
    
    if result.returncode == 0:
        print("\n✅ Visualizations generated successfully!")
        print("📊 Open output/station_map.html in your browser to view the interactive map")
        return True
    else:
        print("\n⚠️  Visualization failed, but pipeline outputs are ready")
        return False

def main():
    """Run everything"""
    print("="*60)
    print("🚂 TRAIN SCHEDULER WITH POPULATION-BASED STATION OPTIMIZATION")
    print("="*60)
    
    # Run pipeline
    if not run_pipeline():
        sys.exit(1)
    
    # Ask if user wants visualizations
    print("\n" + "="*60)
    response = input("\n📊 Generate visualizations? (y/n): ").strip().lower()
    
    if response == 'y':
        run_visualization()
    
    print("\n" + "="*60)
    print("🎉 All done! Check the output/ directory for results")
    print("="*60)

if __name__ == "__main__":
    main()