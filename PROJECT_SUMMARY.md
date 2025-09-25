# BCPC AI Pipeline - Project Summary & Setup Guide

## Project Overview
**BCPC (Bring Cities Back to the People)** - An AI-powered railway planning and optimization system that learns from existing railway networks in countries with good coverage (Belgium, Switzerland, Netherlands, Germany, France) to predict and optimize railway infrastructure for countries with limited/no coverage (Lebanon, Egypt, Morocco, Jordan).

## Directory Structure Created
```
bcpc_ai/
├── src/
│   ├── data_pipeline/        # Data handling and processing
│   ├── model_architecture/   # Neural network models
│   ├── training_loops/       # Training infrastructure
│   └── evaluation_suite/     # Model evaluation tools
├── configs/                  # Configuration files
│   ├── data/                # Data pipeline configs
│   ├── model/               # Model architecture configs
│   └── training/            # Training schedule configs
├── data/
│   ├── train/               # Training data (good coverage countries)
│   └── test/                # Test data (limited coverage countries)
└── models/                  # Saved models and checkpoints
```

## Key Components Implemented

### 1. Data Pipeline (`src/data_pipeline/`)
- **data_loader.py**: Loads railway data (OSM railways, stations, timetables, terrain, costs, passenger flow)
- **feature_extractor.py**: Extracts ML features from raw data (line curvature, station centrality, terrain gradients, etc.)
- **data_splitter.py**: Handles train/test/validation splits with multiple strategies (temporal, spatial, by country)
- **data_validator.py**: Validates data integrity and quality

### 2. Model Architecture (`src/model_architecture/`)
- **base_model.py**: Base neural network architectures (MLP, Graph, Attention, VAE models)
- **route_predictor.py**: Predicts optimal railway routes using LSTM and reinforcement learning
- **nimby_analyzer.py**: Analyzes "Not In My Back Yard" resistance and suggests solutions
- **cost_estimator.py**: Estimates construction costs, time, and ROI
- **timetable_optimizer.py**: Optimizes train schedules and resolves conflicts

### 3. Training Infrastructure (`src/training_loops/`)
- **trainer.py**: Main training loop with checkpointing, early stopping, and logging

### 4. Configuration Files (`configs/`)
- **train_config.yaml**: Data preprocessing and feature selection settings
- **architecture.yaml**: Model architecture specifications
- **schedule.yaml**: Training hyperparameters and optimization settings

### 5. Main Scripts
- **train.py**: Entry point for model training
- **evaluate.py**: Model evaluation script
- **predict.py**: Inference on new countries
- **serve.py**: API server for model deployment

## Setup Instructions for New Session

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision pandas geopandas scikit-learn numpy shapely scipy pyyaml tqdm wandb
```

### 2. Data Preparation
```bash
# Navigate to project
cd bcpc_ai

# Create necessary directories if not exists
mkdir -p data/train data/test models/checkpoints logs

# Place your data files in appropriate folders:
# - OSM railway data → data/train/{country}/railways.geojson
# - Station data → data/train/stations/{country}_stations.csv
# - Timetables → data/train/timetables/{country}_timetables.csv
# - Terrain data → data/train/terrain/{country}_elevation.npy
# - Cost data → data/train/costs/{country}_costs.csv
```

### 3. Training the Model
```bash
# Basic training
python train.py --epochs 100 --batch-size 32

# With custom configs
python train.py --config configs/training/schedule.yaml \
                --model-config configs/model/architecture.yaml \
                --epochs 200
```

### 4. Key Features of the System

**Learning Phase:**
- Extracts railway patterns from countries with existing infrastructure
- Learns relationships between terrain, population, and optimal routes
- Identifies station placement patterns and track specifications

**Inference Phase:**
- References similar geographical/demographic patterns
- Handles NIMBY (community resistance) issues
- Optimizes for cost, time, and social impact
- Generates complete railway plans with timetables

**Special Considerations:**
- Terrain analysis for gradient optimization
- Heritage site preservation
- Land value estimation
- Multi-modal integration planning
- Dense urban area solutions (tunnels, elevated tracks)

## Quick Start for New Chat Session

If starting fresh, explain you're working on the BCPC railway planning AI system that:
1. Has a train/test split between countries with good vs limited rail coverage
2. Uses deep learning to predict optimal railway routes and schedules
3. Main code is in `bcpc_ai/` folder with modular architecture
4. Already has data pipeline, models, and training infrastructure set up
5. Currently needs [specify what you need help with next]

## Next Steps Typically Include:
- Adding real OSM data fetching
- Implementing the actual inference pipeline
- Creating visualization components
- Building the web interface/API
- Fine-tuning models on specific country pairs
- Adding reinforcement learning for route optimization

## Important Notes:
- Models handle both technical (route optimization) and social (NIMBY) aspects
- System is designed to be modular - each component can be improved independently
- Configuration-driven approach allows easy experimentation
- Focus on practical deployment for developing nations

This summary should help you quickly restart in a new session. The core architecture is established and ready for data ingestion and training.