#!/usr/bin/env python3
"""
Automated pipeline runner with flow visualization
Shows real-time progress and materializes assets in order
"""

import subprocess
import sys
import time
import json
from pathlib import Path

def run_dagster_command(cmd_args, description):
    """Run a dagster command and show progress"""
    print(f"\n🚀 {description}")
    print("=" * 60)
    
    cmd = [sys.executable, "-m", "dagster"] + cmd_args
    print(f"📋 Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            print("STDERR:", result.stderr[-500:])
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT (5 minutes)")
        return False
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False

def materialize_assets_sequentially():
    """Materialize assets one by one to see flow"""
    print("🎯 Sequential Asset Materialization")
    print("This will show the dependency flow clearly")
    
    # Define asset sequence with dependencies
    asset_sequence = [
        ("test_environment", "Environment Check"),
        ("load_city_data", "Load Lebanon Cities"),
        ("test_terrain_analysis", "Single City Terrain Test"),
        ("parallel_terrain_test", "Parallel Terrain Test"),
        ("pipeline_summary", "Generate Summary")
    ]
    
    results = {}
    
    for asset_name, description in asset_sequence:
        print(f"\n📦 Materializing: {asset_name}")
        
        success = run_dagster_command([
            "asset", "materialize",
            "--select", asset_name,
            "--module-name", "pipeline.dagster_orchestrator"
        ], f"Materializing {description}")
        
        results[asset_name] = success
        
        if success:
            print(f"✅ {asset_name} completed successfully")
            time.sleep(2)  # Brief pause to see progress
        else:
            print(f"❌ {asset_name} failed - stopping pipeline")
            break
    
    return results

def materialize_all_parallel():
    """Materialize all assets at once (shows full dependency graph)"""
    print("🚀 Parallel Asset Materialization")
    print("This will show the full dependency graph execution")
    
    success = run_dagster_command([
        "asset", "materialize",
        "--select", "*",
        "--module-name", "pipeline.dagster_orchestrator"
    ], "Materializing All Assets (Parallel Dependencies)")
    
    return success

def start_ui_with_auto_refresh():
    """Start UI and provide instructions for flow visualization"""
    print("\n🌐 Starting Dagster UI with Flow Visualization")
    print("=" * 60)
    print("🎯 To see the flow diagram:")
    print("   1. Open http://localhost:3000")
    print("   2. Click 'Assets' → 'View as Graph'")
    print("   3. Click 'Runs' → Latest run → 'View in Launchpad'")
    print("   4. Watch real-time execution in the dependency graph!")
    print("=" * 60)
    
    # Start the UI
    try:
        subprocess.run([
            sys.executable, "-m", "dagster", "dev",
            "--module-name", "pipeline.dagster_orchestrator",
            "--host", "0.0.0.0", "--port", "3000"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dagster UI stopped")

def main():
    print("🚂 BCPC Pipeline Auto-Runner")
    print("=" * 60)
    
    if not Path("pipeline/dagster_orchestrator.py").exists():
        print("❌ Please run this from the root directory (where pipeline/ exists)")
        return 1
    
    print("🎯 Choose execution mode:")
    print("1. 📦 Sequential - See each step (recommended for first run)")
    print("2. ⚡ Parallel - Full pipeline at once (faster)")  
    print("3. 🌐 Start UI only (manual control)")
    print("4. 🔄 Auto-run + UI (start pipeline then open UI)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n📦 Running Sequential Materialization...")
        results = materialize_assets_sequentially()
        
        # Print summary
        print("\n📊 EXECUTION SUMMARY:")
        print("=" * 40)
        for asset, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  {asset:<25} {status}")
        
        successful = sum(results.values())
        total = len(results)
        print(f"\n🎯 Overall: {successful}/{total} assets successful")
        
        if successful == total:
            print("🎉 All assets completed successfully!")
            print("📊 Check output/pipeline_test_summary.json for results")
        
    elif choice == "2":
        print("\n⚡ Running Parallel Materialization...")
        success = materialize_all_parallel()
        
        if success:
            print("🎉 Parallel pipeline completed successfully!")
            print("📊 Check output/pipeline_test_summary.json for results")
        else:
            print("❌ Parallel pipeline failed")
            
    elif choice == "3":
        print("\n🌐 Starting UI for manual control...")
        start_ui_with_auto_refresh()
        
    elif choice == "4":
        print("\n🔄 Auto-run + UI Mode")
        
        # Start pipeline in background
        print("🚀 Starting parallel pipeline...")
        
        # Use subprocess.Popen to start pipeline in background
        pipeline_cmd = [
            sys.executable, "-m", "dagster", "asset", "materialize",
            "--select", "*",
            "--module-name", "pipeline.dagster_orchestrator"
        ]
        
        print("📋 Starting background pipeline...")
        pipeline_process = subprocess.Popen(pipeline_cmd)
        
        # Give it a moment to start
        time.sleep(3)
        
        print("🌐 Starting UI to monitor progress...")
        print("🎯 Open http://localhost:3000 → Runs → Latest run to see live progress!")
        
        try:
            # Start UI
            subprocess.run([
                sys.executable, "-m", "dagster", "dev",
                "--module-name", "pipeline.dagster_orchestrator", 
                "--host", "0.0.0.0", "--port", "3000"
            ])
        except KeyboardInterrupt:
            print("\n🛑 Stopping UI and pipeline...")
            pipeline_process.terminate()
            
    else:
        print("❌ Invalid choice")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())