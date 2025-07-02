"""
install_deps.py
Quick dependency installer for the Train Scheduler project
"""
import subprocess
import sys

def install_dependencies():
    """Install all required dependencies"""
    print("📦 Installing required dependencies for Train Scheduler...\n")
    
    # Core dependencies that are missing
    essential_packages = [
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.0.0"
    ]
    
    # Install essentials first
    print("🔧 Installing essential packages...")
    for package in essential_packages:
        print(f"   Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\n✅ Essential dependencies installed!")
    
    # Ask about optional packages
    print("\n" + "="*50)
    response = input("Install all dependencies from requirements.txt? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\n📦 Installing all dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "Model/requirements.txt"])
        print("\n✅ All dependencies installed!")
    
    print("\n🎉 Setup complete! You can now run: python script/train_pipeline.py")

if __name__ == "__main__":
    install_dependencies()