# Simple Guide to Run the BCPC Railway AI Project

## Quick Setup (5 minutes)

### 1. Open Terminal and Go to Project
```bash
cd Train_Scheduler/bcpc_ai
```

### 2. Install Python Packages
```bash
pip install torch numpy pandas scikit-learn pyyaml flask matplotlib
```

### 3. Create Folders
```bash
mkdir -p data models logs artifacts
```

## Run the Project (3 Options)

### Option A: Generate a Railway Plan (Easiest)
```bash
python predict.py --country lebanon
```
This creates a railway infrastructure plan for Lebanon and saves it as `predictions.json`

### Option B: Train a Model
```bash
python train.py --epochs 5
```
This trains a simple model with mock data (takes ~2 minutes)

### Option C: Start API Server
```bash
python serve.py --port 8000
```
Then open browser to `http://localhost:8000/api/v1/health`

## What Each Command Does

- **predict.py** - Generates railway plans for countries without trains
- **train.py** - Learns from countries with good railway systems  
- **evaluate.py** - Tests how well the model works
- **serve.py** - Starts a web server for predictions

## If Something Doesn't Work

**Error: "ModuleNotFoundError"**
```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

**Error: "No such file or directory"**
```bash
mkdir -p data models logs artifacts
```

**Error: "Model file not found"**
- This is fine! The system works with mock data if no models exist yet

That's it! The system will use mock data for testing, so you can see how it works without needing real railway data.