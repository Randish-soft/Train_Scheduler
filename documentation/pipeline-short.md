# Railway Planning ML Pipeline

## Training Countries
Belgium, Netherlands, Japan, France, Spain, China, UK (mainline + London Underground)

## Training Phase

### Data Collection (I/O Intensive)
1. **Geographic Layers**
   - Land use (residential, commercial, industrial, agricultural)
   - Existing rail tracks and stations
   - Protected areas and obstacles
   - Water bodies and property boundaries

2. **Terrain Data**
   - Elevation models (DEM)
   - Slope calculations
   - Geological data (soil types, seismic zones)

3. **Socioeconomic Data**
   - Population density heatmaps
   - Origin-destination demand matrices
   - Economic activity centers

4. **Infrastructure Data**
   - Existing timetables and service frequency
   - Station types and capacity
   - Track specifications (gauge, elevation profiles)
   - Construction cost data per terrain type

### Model Training (CPU/GPU Intensive)
**Objective**: Multi-model system for railway optimization

1. **Route Pathfinding Model**
   - Input: Start/end points, constraint layers
   - Output: Optimal path coordinates
   - Method: Graph Neural Network or Reinforcement Learning

2. **Station Placement Model**
   - Input: Route + population data
   - Output: Station locations and types
   - Method: Classification

3. **Cost Estimation Model**
   - Input: Route features (terrain, length, obstacles)
   - Output: Construction cost prediction
   - Method: Regression

4. **Save trained models** (pickle/ONNX format)

## Inference Phase

### Input Processing
1. Load target country/region data
2. Check for existing railway infrastructure
3. Validate input constraints (budget, timeline, environmental limits)

### Route Generation
1. **ML Inference**: Generate 3-5 candidate routes using trained models
2. **Engineering Validation**: Filter by hard constraints
   - Maximum grade limits (typically 3-4% for conventional rail)
   - Minimum curve radii
   - Bridge/tunnel feasibility
3. **Multi-Criteria Ranking**: Score routes by:
   - Construction cost
   - Travel time
   - Environmental impact
   - Ridership potential

### Output & Visualization
1. Present top-ranked routes with tradeoff analysis
2. Interactive map showing:
   - Proposed track path with elevation profile
   - Station locations
   - Cost breakdown
   - Construction phases
3. Export engineering specifications

## Continuous Improvement
- New country data flagged for manual review
- Periodic retraining with validated real-world projects
- Transfer learning for similar geographic contexts