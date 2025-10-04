# BCPC: Bring Cities Back to the People
## Complete Technical Documentation v2.0

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Learning Phase](#learning-phase)
5. [Inference Phase](#inference-phase)
6. [Technical Specifications](#technical-specifications)
7. [Implementation Guide](#implementation-guide)
8. [API Reference](#api-reference)
9. [Case Studies](#case-studies)
10. [Appendices](#appendices)

---

## 1. Executive Summary

### 1.1 Project Overview
BCPC is an intelligent railway planning system that learns from existing railway networks worldwide to design, optimize, and recommend new train lines and schedules for countries with varying levels of rail infrastructure coverage—from nations with comprehensive networks to those with minimal or no existing rail systems.

### 1.2 Core Objectives
- Analyze existing railway infrastructure across multiple countries and continents
- Learn patterns from successful railway implementations globally
- Generate feasible railway proposals with cost estimates
- Optimize station placement and route efficiency
- Create realistic timetables for proposed lines
- Leverage existing railway infrastructure when available
- Address socio-political challenges (NIMBY issues, land acquisition)
- Provide solutions for countries at all development stages

### 1.3 Key Capabilities
- **Automated Route Planning**: Generate optimal railway routes considering terrain, population, and existing infrastructure
- **Existing Infrastructure Integration**: Utilize existing rail lines, stations, and corridors when beneficial
- **Cost Estimation**: Provide detailed construction cost breakdowns with inflation-adjusted historical data
- **Timetable Generation**: Create realistic train schedules based on line characteristics
- **Multi-Country Learning**: Transfer knowledge across different railway systems, geographical contexts, and development levels
- **Constraint Handling**: Navigate archaeological sites, high land values, environmental concerns, and political barriers
- **Adaptive Design**: Scale solutions from developing nations to advanced rail networks

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                     │
│  (Web Interface, API Endpoints, Visualization Dashboard)    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                       │
│         (Request Handler, Queue Manager, Validator)          │
└─────────────────────────────────────────────────────────────┘
                              │
                 ─────────────┴─────────────
                 │                         │
                 ▼                         ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│    LEARNING ENGINE       │  │   INFERENCE ENGINE       │
│  - Route Analysis        │  │  - Route Generation      │
│  - Pattern Recognition   │  │  - Existing Line Reuse   │
│  - Feature Extraction    │  │  - Optimization          │
│  - Global Benchmarking   │  │  - Cost Calculation      │
└──────────────────────────┘  └──────────────────────────┘
                 │                         │
                 └─────────────┬───────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       DATA LAYER                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Route DB   │  │  Terrain DB  │  │  Reference DB    │   │
│  │  - Lines    │  │  - Elevation │  │  - Patterns      │   │
│  │  - Stations │  │  - Geology   │  │  - Costs         │   │
│  │  - Tables   │  │  - Land Use  │  │  - Benchmarks    │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   EXTERNAL DATA SOURCES                      │
│  - OpenStreetMap  - Railway APIs  - Elevation APIs           │
│  - Census Data    - Land Registry - Historical Cost Data     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Component Breakdown

#### 2.2.1 Learning Engine
- **Route Analyzer**: Extracts railway geometry, station placement, and operational patterns from global networks
- **Pattern Recognizer**: Identifies design principles (curves, gradients, station spacing) across different regions
- **Feature Extractor**: Converts raw data into structured features for pattern matching
- **Knowledge Base Builder**: Creates reusable templates from learned patterns worldwide
- **Comparative Analyzer**: Benchmarks similar countries/regions (e.g., mountainous vs. flat, developed vs. developing)

#### 2.2.2 Inference Engine
- **Route Generator**: Proposes new railway alignments based on learned patterns
- **Existing Infrastructure Evaluator**: Determines when to reuse existing rail corridors vs. building new
- **Optimizer**: Refines routes for cost, time, accessibility, and integration with existing networks
- **Constraint Solver**: Handles NIMBY issues, archaeological sites, terrain limitations, and political boundaries
- **Timetable Calculator**: Generates realistic schedules that integrate with existing services
- **Upgrade Assessor**: Evaluates whether existing lines should be upgraded vs. building parallel routes

#### 2.2.3 Data Pipeline
- **Ingestion Layer**: Collects data from multiple sources worldwide
- **Transformation Layer**: Normalizes data across different standards (European, American, Asian, etc.)
- **Storage Layer**: Manages structured and geospatial data
- **Cache Layer**: Optimizes frequent queries

---

## 3. Data Pipeline

### 3.1 Data Sources

#### 3.1.1 Railway Data (Global)
- **OpenStreetMap**: Global track geometry, station locations
- **National Railway APIs**: 
  - **Europe**: SNCF (France), NMBS/SNCB (Belgium), NS (Netherlands), DB (Germany), Trenitalia (Italy), Renfe (Spain), SBB (Switzerland), ÖBB (Austria)
  - **Asia**: JR Group (Japan), China Railway, Indian Railways, Korean Rail
  - **Americas**: Amtrak (USA), VIA Rail (Canada), various freight operators
  - **Other**: National operators from Australia, South Africa, etc.
- **GTFS Feeds**: Schedule data in standardized format (global coverage)
- **Railway Gazette International**: Infrastructure project data worldwide
- **UIC (International Union of Railways)**: Global railway statistics

#### 3.1.2 Geospatial Data
- **SRTM (Shuttle Radar Topography Mission)**: Global elevation data (30m resolution)
- **ASTER GDEM**: Alternative global elevation source (30m resolution)
- **Geological Surveys**: Soil composition, seismic data from various national agencies
- **Global Land Use Databases**: Urban planning, protected areas, UNESCO sites

#### 3.1.3 Demographic & Economic Data
- **National Census Bureaus**: Population density, demographics
- **World Bank**: Economic statistics, GDP per capita, development indices
- **Transport Statistics**: Current ridership, modal split from various countries
- **Urban Planning Databases**: City development plans, zoning information

#### 3.1.4 Historical Cost Data
- **International Infrastructure Databases**: Actual construction costs from multiple countries
- **Railway Engineering Journals**: Per-kilometer costs by terrain type and country
- **Inflation Indices**: Country-specific indices to adjust historical costs
- **PPP (Purchasing Power Parity) Data**: For cross-country cost comparisons

### 3.2 Data Schema

#### 3.2.1 Railway Line Schema
```json
{
  "line_id": "BE-IC-01",
  "line_name": "Oostende - Eupen",
  "country": "Belgium",
  "region": "Europe",
  "operator": "NMBS/SNCB",
  "sections": [
    {
      "section_id": "BE-IC-01-S1",
      "start_station": "Oostende",
      "end_station": "Leuven",
      "distance_km": 115.3,
      "geometry": "LINESTRING(...)",
      "speed_class": "medium_speed",
      "max_speed_kmh": 140,
      "track_type": "ballasted",
      "gauge_mm": 1435,
      "electrification": "25kV_AC",
      "construction_type": [
        {"type": "at_grade", "percentage": 85}

def adjust_costs_by_region(base_cost, region, terrain_type):
    """Adjust construction costs based on region and terrain"""
    regional_multiplier = cost_multipliers.get(region, 1.0)
    
    # Terrain adjustments apply universally
    terrain_multipliers = {
        'flat': 1.0,
        'rolling': 1.2,
        'mountainous': 1.8,
        'urban_dense': 2.5,
        'desert': 1.1,
        'swamp': 1.4
    }
    
    terrain_multiplier = terrain_multipliers.get(terrain_type, 1.0)
    
    return base_cost * regional_multiplier * terrain_multiplier
```

**Inflation Adjustment (Multi-Currency)**:
```python
def adjust_for_inflation(historical_cost, year_spent, currency, target_year=2025):
    """Adjust historical costs to present value across currencies"""
    
    # Use construction price index for specific country
    base_index = get_construction_index(year_spent, currency)
    current_index = get_construction_index(target_year, currency)
    
    inflation_factor = current_index / base_index
    adjusted_cost_local = historical_cost * inflation_factor
    
    # Convert to common currency (EUR or USD)
    exchange_rate = get_exchange_rate(currency, 'EUR', target_year)
    adjusted_cost_eur = adjusted_cost_local * exchange_rate
    
    return {
        'local_currency': adjusted_cost_local,
        'eur': adjusted_cost_eur,
        'currency': currency
    }
```

**Cost Database Structure**:
```json
{
  "project_id": "CH-GOTTH-BT",
  "name": "Gotthard Base Tunnel",
  "country": "Switzerland",
  "region": "western_europe",
  "year_completed": 2016,
  "length_km": 57.1,
  "total_cost_original": 12200000000,
  "currency": "CHF",
  "total_cost_eur_2025": 11800000000,
  "cost_per_km_eur": 206654000,
  "breakdown": {
    "tunnels_km": 57.1,
    "tunnel_cost_per_km": 206654000,
    "tunnel_type": "twin_bore_TBM"
  },
  "terrain_type": "alpine_mountainous",
  "max_speed_kmh": 250,
  "comparable_projects": [
    "JP-SEIKAN-TUNNEL",
    "UK-CHANNEL-TUNNEL",
    "ES-GUADARRAMA-TUNNEL"
  ],
  "lessons_learned": [
    "TBM tunneling in varied geology",
    "Long construction period (17 years)",
    "Extensive safety systems required"
  ]
}
```

### 4.2 Intelligent Logging System

#### 4.2.1 Dense Map Reference Lines

**Concept**: Routes that share significant track sections should reference a common "base route" to avoid redundant storage and enable efficient reuse during inference.

**Global Examples**:

**Example 1: European Hub Pattern**
```
Major Hub: Brussels
Lines passing through:
- Oostende → Eupen
- Antwerp → Charleroi
- Amsterdam → Paris
- Liège → Lille
- London → Cologne

Dense Reference: "Brussels-North-Central-South-Corridor"
All lines reference this corridor for their Brussels section.
```

**Example 2: Asian Mega-Hub Pattern**
```
Major Hub: Tokyo Station
Lines converging:
- Tokaido Shinkansen (Osaka direction)
- Tohoku Shinkansen (Sendai direction)
- Joetsu Shinkansen (Niigata direction)
- Chuo Line
- Yamanote Line integration

Dense Reference: "Tokyo-Central-Station-Complex"
Separate references for each approach corridor
```

**Example 3: North American Hub Pattern**
```
Major Hub: Chicago Union Station
Lines passing through:
- California Zephyr (SF-Chicago)
- Southwest Chief (LA-Chicago)
- Empire Builder (Seattle-Chicago)
- Lake Shore Limited (Boston/NYC-Chicago)

Dense Reference: "Chicago-Union-Station-Approach"
```

**Example 4: Developing Country Corridor**
```
Major Corridor: Nairobi-Mombasa SGR (Kenya)
Potential branches:
- Nairobi → Kampala extension
- Nairobi → Addis Ababa connection
- Mombasa → Dar es Salaam coastal route

Dense Reference: "Nairobi-Mombasa-SGR-Mainline"
Future lines can reference and extend this corridor
```

**Implementation**:
```python
class DenseMapReferenceManager:
    def __init__(self):
        self.reference_sections = {}
        self.global_hubs = {}
    
    def identify_shared_sections(self, all_lines, country=None):
        """Find track sections used by multiple lines
        
        Can analyze globally or by country
        """
        section_usage = defaultdict(list)
        
        # Filter by country if specified
        if country:
            all_lines = [l for l in all_lines if l.country == country]
        
        for line in all_lines:
            for section in line.sections:
                section_hash = self.hash_section(section.geometry)
                section_usage[section_hash].append({
                    'line_id': line.id,
                    'country': line.country,
                    'operator': line.operator
                })
        
        # Create reference sections for heavily used tracks
        for section_hash, usage_list in section_usage.items():
            if len(usage_list) >= 3:  # Used by 3+ lines
                self.create_reference_section(section_hash, usage_list)
    
    def create_reference_section(self, section_hash, usage_list):
        """Create a reusable reference section"""
        reference_id = f"REF-{section_hash[:8]}"
        
        # Determine if this is a major hub
        countries = set(u['country'] for u in usage_list)
        if len(usage_list) >= 10:
            hub_type = "mega_hub"
        elif len(usage_list) >= 5:
            hub_type = "major_hub"
        elif len(countries) > 1:
            hub_type = "international_corridor"
        else:
            hub_type = "regional_hub"
        
        self.reference_sections[reference_id] = {
            'hash': section_hash,
            'lines_using': usage_list,
            'hub_type': hub_type,
            'geometry': self.get_section_geometry(section_hash),
            'characteristics': self.extract_characteristics(section_hash),
            'upgrade_potential': self.assess_upgrade_potential(usage_list)
        }
        return reference_id
    
    def assess_upgrade_potential(self, usage_list):
        """Determine if shared section is candidate for upgrade"""
        return {
            'high_traffic': len(usage_list) >= 8,
            'international': len(set(u['country'] for u in usage_list)) > 1,
            'capacity_constrained': self.check_capacity(usage_list),
            'recommendation': 'upgrade_to_high_speed' if len(usage_list) >= 10 else 'maintain'
        }
```

**Storage Optimization Examples**:

**Example: Two lines sharing infrastructure**
```json
{
  "line_1": {
    "line_id": "BE-IC-01",
    "name": "Oostende-Eupen",
    "sections": [
      {"from": "Oostende", "to": "Bruges", "geometry": "LINESTRING(...)"},
      {"from": "Bruges", "to": "Leuven", "reference": "REF-BRU-LEU-01"},
      {"from": "Leuven", "to": "Eupen", "geometry": "LINESTRING(...)"}
    ]
  },
  "line_2": {
    "line_id": "BE-IC-05",
    "name": "Blankenberge-Genk",
    "sections": [
      {"from": "Blankenberge", "to": "Bruges", "geometry": "LINESTRING(...)"},
      {"from": "Bruges", "to": "Leuven", "reference": "REF-BRU-LEU-01"},
      {"from": "Leuven", "to": "Genk", "geometry": "LINESTRING(...)"}
    ]
  },
  "reference_sections": {
    "REF-BRU-LEU-01": {
      "name": "Bruges-Leuven Corridor",
      "geometry": "LINESTRING(...)",
      "length_km": 85,
      "characteristics": {
        "speed_class": "medium_speed",
        "max_speed": 140,
        "tracks": "double_track",
        "electrification": "25kV_AC"
      },
      "used_by": ["BE-IC-01", "BE-IC-05", "BE-IC-08", "BE-IC-12"],
      "upgrade_status": "planned_high_speed_upgrade_2028"
    }
  }
}
```

#### 4.2.2 Metadata Management

**Line Metadata (Enhanced for Global Learning)**:
```json
{
  "line_id": "JP-TOKAIDO-SHINKANSEN",
  "learning_date": "2025-03-15",
  "country": "Japan",
  "region": "asia",
  "development_level": "advanced",
  "data_quality": {
    "geometry_accuracy": "high",
    "schedule_completeness": 0.98,
    "cost_data_available": true,
    "terrain_data_quality": "excellent"
  },
  "pattern_tags": [
    "very_high_speed",
    "coastal_mountainous_mix",
    "extensive_tunneling",
    "very_high_frequency",
    "elevated_urban_sections"
  ],
  "references_sections": ["REF-TOKYO-YOKOHAMA-01", "REF-NAGOYA-STATION-01"],
  "referenced_by": ["JP-SANYO-SHINKANSEN"],
  "comparable_lines": [
    "FR-TGV-SUD-EST",
    "ES-AVE-MADRID-BARCELONA",
    "CN-BEIJING-SHANGHAI-HSR"
  ],
  "learning_priority": "high",
  "global_significance": {
    "pioneering_technology": true,
    "ridership_millions_per_year": 165,
    "operational_excellence": "world_leading",
    "lessons_applicable_to": ["all_high_speed_projects"]
  }
}
```

**Country/Region Context Metadata**:
```json
{
  "country": "Switzerland",
  "region": "western_europe",
  "railway_maturity": "very_high",
  "total_network_km": 5200,
  "electrification_percentage": 99,
  "construction_cost_index": 1.2,
  "preferred_construction_methods": [
    "tunnel_preference",
    "low_gradient_optimization",
    "high_quality_standards"
  ],
  "design_philosophy": {
    "max_gradient_standard": 1.2,
    "preferred_tunnel_method": "TBM",
    "station_design": "integrated_with_urban_planning",
    "environmental_priority": "very_high"
  },
  "comparable_countries": ["Austria", "Norway", "Japan"],
  "learning_value": {
    "mountainous_terrain": "world_leading",
    "tunnel_engineering": "world_leading",
    "precision_operations": "world_leading",
    "applicable_to": ["mountainous_regions_globally"]
  }
}
```

### 4.3 Pattern Recognition

#### 4.3.1 Design Pattern Extraction (Global)

**Patterns Learned Across Regions**:

1. **Station Spacing Patterns**
   ```python
   def extract_station_spacing_pattern(line, regional_context):
       """Learn typical station spacing for line type and region"""
       spacings = []
       urban_spacings = []
       rural_spacings = []
       
       for i in range(len(line.stations) - 1):
           distance = calculate_distance(
               line.stations[i].location,
               line.stations[i+1].location
           )
           population_density = get_population_density(
               line.stations[i].location,
               line.stations[i+1].location
           )
           
           spacings.append(distance)
           
           if population_density > 1000:
               urban_spacings.append(distance)
           else:
               rural_spacings.append(distance)
       
       return {
           'line_type': line.type,
           'country': line.country,
           'region': regional_context['region'],
           'median_spacing_km': np.median(spacings),
           'urban_spacing_km': np.median(urban_spacings) if urban_spacings else None,
           'rural_spacing_km': np.median(rural_spacings) if rural_spacings else None,
           'comparison_to_regional_norm': compare_to_regional_average(spacings, regional_context)
       }
   ```
   
   **Global Station Spacing Results**:
   
   **Very High-Speed (250+ km/h)**:
   - Japan Shinkansen: 50-70 km
   - French TGV: 60-90 km  
   - Spanish AVE: 50-80 km
   - Chinese HSR: 40-60 km (more stations for political reasons)
   
   **High-Speed (200-250 km/h)**:
   - European Intercity: 30-50 km
   - Korean KTX: 30-45 km
   
   **Medium-Speed (120-200 km/h)**:
   - European Regional Express: 15-30 km
   - US Amtrak corridors: 20-40 km
   - Australian intercity: 25-40 km
   
   **Standard Speed (80-120 km/h)**:
   - European Regional: 5-15 km
   - Indian Railways: 10-25 km
   - African lines: 15-40 km (larger spacing, fewer resources)
   
   **Urban/Suburban (40-80 km/h)**:
   - European S-Bahn: 2-4 km
   - Japanese commuter: 1.5-3 km
   - US commuter rail: 3-6 km

2. **Curve Radius Patterns by Speed Class**
   ```python
   global_curve_standards = {
       'very_high_speed': {
           'min_radius_m': 4000,
           'preferred_radius_m': 7000,
           'examples': ['Shinkansen', 'TGV', 'ICE']
       },
       'high_speed': {
           'min_radius_m': 2500,
           'preferred_radius_m': 4000,
           'examples': ['Intercity Europe', 'KTX']
       },
       'medium_speed': {
           'min_radius_m': 1000,
           'preferred_radius_m': 1500,
           'examples': ['Regional express']
       },
       'standard_speed': {
           'min_radius_m': 300,
           'preferred_radius_m': 600,
           'examples': ['Branch lines', 'mountain railways']
       }
   }
   ```

3. **Gradient Handling Philosophies**
   
   **European Continental Approach**:
   - Prefer long detours to maintain <2% gradients
   - Extensive tunneling through mountains
   - Example: Gotthard Base Tunnel (1.2% max)
   
   **Swiss Philosophy**: Maximum grade 1.2-2.6% (Alps)
   **German Philosophy**: Maximum grade 1.25-2% (rolling hills)
   **Dutch Philosophy**: Essentially flat (0-0.5%)
   
   **Japanese Approach**:
   - Accept higher gradients (up to 3%) with powerful trains
   - Balance tunneling with surface routes
   - Example: Tokaido Shinkansen through mountains
   
   **Chinese Approach**:
   - Extensive viaducts in flat/rolling terrain
   - Deep tunnels through mountains
   - Gradients typically <2%
   
   **North American Freight Heritage**:
   - Historically accepted 2-3% gradients
   - Modern passenger: prefer <1.5%
   - Example: California HSR design <1.5%
   
   **Developing Country Pragmatism**:
   - Accept 3-5% gradients to reduce costs
   - Slower speeds acceptable
   - Examples: Mountain railways in India, Peru, Kenya
   
   ```python
   def classify_gradient_philosophy(country_lines):
       """Determine regional approach to gradients"""
       all_gradients = []
       tunnel_avoidance_ratio = []
       
       for line in country_lines:
           max_grad = line.terrain_profile['max_gradient']
           all_gradients.append(max_grad)
           
           # Calculate if line took longer route to avoid gradient
           direct_distance = calculate_direct_distance(line)
           actual_distance = line.total_length_km
           detour_ratio = actual_distance / direct_distance
           
           if detour_ratio > 1.15:  # 15%+ longer route
               tunnel_avoidance_ratio.append(detour_ratio)
       
       avg_max_gradient = np.mean(all_gradients)
       avg_detour = np.mean(tunnel_avoidance_ratio) if tunnel_avoidance_ratio else 1.0
       
       if avg_max_gradient < 1.5 and avg_detour > 1.2:
           return "gradient_minimization_priority"
       elif avg_max_gradient < 2.5 and avg_detour < 1.1:
           return "balanced_approach"
       elif avg_max_gradient > 3.0:
           return "cost_constrained_acceptance"
       else:
           return "moderate_gradient_tolerance"
   ```

4. **Urban Entry Strategies**
   
   **Asian Dense City Pattern** (Tokyo, Hong Kong, Singapore):
   - Transition to elevated 10-15 km from center
   - Underground in city core
   - Elevated: 40-60% of urban route
   - Underground: 30-50% of urban route
   
   **European Historic City Pattern** (Paris, Rome, Vienna):
   - Maintain at-grade until 3-5 km from center
   - Underground terminal stations
   - Preserve historic city centers
   
   **North American Suburban Pattern**:
   - At-grade through suburbs
   - Grade separated crossings
   - Limited tunneling due to cost constraints
   
   **Developing City Pattern** (Nairobi, Lagos, Jakarta):
   - Primarily elevated to reduce land acquisition
   - At-grade in less dense areas
   - Minimal tunneling (cost prohibitive)

5. **Electrification Patterns**
   
   **Global Electrification Standards**:
   ```json
   {
     "overhead_25kV_AC": {
       "regions": ["Europe", "China", "Japan (Shinkansen)", "Most modern lines"],
       "advantages": ["Efficient", "Standard for HSR"],
       "cost_per_km": "€1-2M"
     },
     "overhead_15kV_AC": {
       "regions": ["Germany", "Austria", "Switzerland", "Norway", "Sweden"],
       "advantages": ["Compatible with legacy systems"],
       "cost_per_km": "€1-2M"
     },
     "overhead_1.5kV_DC": {
       "regions": ["Netherlands", "Japan (legacy)", "Parts of France"],
       "advantages": ["Historical standard"],
       "cost_per_km": "€1-1.5M"
     },
     "third_rail": {
       "regions": ["UK Southern", "NYC Subway", "Many metros"],
       "advantages": ["Lower infrastructure", "Good for tunnels"],
       "limitations": ["Safety concerns surface", "Speed limited"],
       "cost_per_km": "€0.5-1M"
     },
     "diesel": {
       "regions": ["North America (majority)", "Australia", "Developing nations"],
       "advantages": ["No infrastructure cost", "Flexibility"],
       "limitations": ["Higher operating cost", "Emissions"]
     }
   }
   ```

#### 4.3.2 Cost Prediction Models (Global)

**Machine Learning Approach with Regional Factors**:
```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd

def train_global_cost_model(historical_projects):
    """Train model to predict construction costs globally
    
    Accounts for regional variations and project characteristics
    """
    
    features = []
    targets = []
    regions = []
    
    for project in historical_projects:
        feature_vector = [
            project['length_km'],
            project['tunnel_percentage'],
            project['viaduct_percentage'],
            project['urban_percentage'],
            project['max_speed_kmh'],
            project['terrain_difficulty_score'],  # 1-10 scale
            project['year_completed'],
            project['stations_count'],
            project['avg_station_complexity'],  # 1-5 scale
            get_labor_cost_index(project['country']),
            get_regulatory_complexity(project['country']),  # 1-5 scale
            get_land_acquisition_difficulty(project['country']),  # 1-5 scale
            int(project['high_speed']),  # Boolean as int
            int(project['electrified'])  # Boolean as int
        ]
        features.append(feature_vector)
        targets.append(project['cost_per_km'])
        regions.append(project['region'])
    
    # Create DataFrame for better handling
    df = pd.DataFrame(features, columns=[
        'length_km', 'tunnel_pct', 'viaduct_pct', 'urban_pct',
        'max_speed', 'terrain_difficulty', 'year', 'stations',
        'station_complexity', 'labor_index', 'regulatory_complexity',
        'land_difficulty', 'high_speed', 'electrified'
    ])
    df['region'] = regions
    
    # One-hot encode regions
    df = pd.get_dummies(df, columns=['region'])
    
    # Split data
    X = df.drop('region', axis=1, errors='ignore')
    y = targets
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train ensemble model
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    )
    
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        random_state=42
    )
    
    rf_model.fit(X_scaled, y)
    gb_model.fit(X_scaled, y)
    
    # Create ensemble
    def ensemble_predict(features):
        rf_pred = rf_model.predict(features)
        gb_pred = gb_model.predict(features)
        return (rf_pred + gb_pred) / 2
    
    return {
        'model': ensemble_predict,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'feature_importance': rf_model.feature_importances_
    }

def predict_project_cost(project_spec, trained_model):
    """Predict cost for a new project"""
    
    # Prepare features
    features = prepare_features(project_spec)
    scaled_features = trained_model['scaler'].transform([features])
    
    # Get prediction
    predicted_cost_per_km = trained_model['model'](scaled_features)[0]
    
    # Calculate confidence interval
    uncertainty = calculate_uncertainty(project_spec, trained_model)
    
    total_cost = predicted_cost_per_km * project_spec['length_km']
    
    return {
        'cost_per_km': predicted_cost_per_km,
        'total_cost': total_cost,
        'confidence_interval': {
            'lower': total_cost * (1 - uncertainty),
            'upper': total_cost * (1 + uncertainty)
        },
        'breakdown': estimate_cost_breakdown(project_spec, predicted_cost_per_km)
    }
```

**Cost Patterns by Project Type**:
```python
typical_cost_ranges = {
    'basic_freight_line_developing': {
        'cost_per_km_usd': (2_000_000, 5_000_000),
        'examples': ['African SGR projects', 'South American freight'],
        'characteristics': ['Single track', 'Diesel', 'Basic signaling']
    },
    'regional_passenger_line': {
        'cost_per_km_usd': (5_000_000, 15_000_000),
        'examples': ['European regional', 'US commuter rail'],
        'characteristics': ['Double track', 'Electrified', 'Modern signaling']
    },
    'intercity_medium_speed': {
        'cost_per_km_usd': (10_000_000, 25_000_000),
        'examples': ['UK intercity', 'Australian interstate'],
        'characteristics': ['Double track', 'Electrified', 'Advanced signaling', '160-200 km/h']
    },
    'high_speed_flat_terrain': {
        'cost_per_km_usd': (20_000_000, 40_000_000),
        'examples': ['China HSR (plains)', 'Spanish AVE (meseta)'],
        'characteristics': ['Dedicated HSR', '250-350 km/h', 'Primarily viaducts/at-grade']
    },
    'high_speed_mixed_terrain': {
        'cost_per_km_usd': (35_000_000, 70_000_000),
        'examples': ['French TGV', 'Italian TAV', 'California HSR'],
        'characteristics': ['Mixed tunnels/viaducts', '250-300 km/h', 'Urban sections']
    },
    'high_speed_mountainous': {
        'cost_per_km_usd': (60_000_000, 120_000_000),
        'examples': ['Swiss base tunnels', 'Japanese Alps routes'],
        'characteristics': ['Extensive tunneling', '200-250 km/h', 'Complex geology']
    },
    'urban_metro': {
        'cost_per_km_usd': (100_000_000, 300_000_000),
        'examples': ['Paris Metro extensions', 'NYC Subway', 'London Crossrail'],
        'characteristics': ['Fully underground', 'Dense urban area', 'Frequent stations']
    }
}
```

---

## 5. Inference Phase

### 5.1 Query Processing & Route Analysis

**User Input Processing**:

```python
class QueryProcessor:
    def __init__(self, learning_database):
        self.db = learning_database
        self.existing_networks = self.db.get_all_networks()
    
    def process_query(self, user_query):
        """Process user request for new railway line
        
        Example queries:
        - "Create express line from Lille to Antwerp"
        - "Connect Nairobi to Kampala"
        - "Extend metro to airport"
        """
        
        # Parse query
        parsed = self.parse_query(user_query)
        
        # Extract key information
        origin = parsed['origin']
        destination = parsed['destination']
        service_type = parsed['service_type']  # express, regional, metro, etc.
        country = parsed['country']
        budget = parsed.get('budget', None)
        
        # Check existing infrastructure
        existing_analysis = self.analyze_existing_infrastructure(
            origin, destination, country
        )
        
        return {
            'origin': origin,
            'destination': destination,
            'service_type': service_type,
            'country': country,
            'budget': budget,
            'existing_infrastructure': existing_analysis
        }
```

### 5.2 Existing Infrastructure Analysis

**Critical Component: Determining Whether to Reuse Existing Lines**

```python
class ExistingInfrastructureEvaluator:
    """Evaluates existing railway infrastructure for reuse potential"""
    
    def analyze_existing_infrastructure(self, origin, destination, country):
        """Comprehensive analysis of existing rail between two points"""
        
        # Find all existing rail connections
        existing_routes = self.find_existing_routes(origin, destination)
        
        if not existing_routes:
            return {
                'exists': False,
                'recommendation': 'build_new',
                'reason': 'no_existing_infrastructure'
            }
        
        # Analyze each existing route
        analysis_results = []
        for route in existing_routes:
            evaluation = self.evaluate_route_reusability(route)
            analysis_results.append(evaluation)
        
        # Determine best approach
        best_approach = self.determine_optimal_approach(analysis_results)
        
        return best_approach
    
    def evaluate_route_reusability(self, existing_route):
        """Evaluate if existing route can be reused or needs upgrade"""
        
        metrics = {
            'current_capacity': self.calculate_current_capacity(existing_route),
            'current_speed': existing_route.max_speed_kmh,
            'track_condition': self.assess_track_condition(existing_route),
            'electrification': existing_route.electrification_status,
            'signaling_system': existing_route.signaling_system,
            'route_directness': self.calculate_directness(existing_route),
            'station_locations': self.evaluate_stations(existing_route)
        }
        
        # Decision matrix
        if metrics['current_capacity'] < 0.7:  # Below 70% capacity
            if metrics['current_speed'] >= 160 and metrics['route_directness'] > 0.85:
                return {
                    'approach': 'reuse_existing',
                    'modifications': ['add_trains', 'minor_upgrades'],
                    'estimated_cost_vs_new': 0.1,  # 10% of new construction
                    'timeline_months': 6
                }
            elif metrics['track_condition'] == 'good' and metrics['current_speed'] >= 100:
                return {
                    'approach': 'upgrade_existing',
                    'modifications': ['track_upgrades', 'electrification', 'modern_signaling'],
                    'estimated_cost_vs_new': 0.3,  # 30% of new construction
                    'timeline_months': 18,
                    'speed_improvement': '100 → 160 km/h'
                }
        
        elif metrics['current_capacity'] >= 0.9:  # Near capacity
            if metrics['route_directness'] > 0.9:
                return {
                    'approach': 'parallel_tracks',
                    'modifications': ['add_parallel_tracks_to_existing_corridor'],
                    'estimated_cost_vs_new': 0.6,  # 60% of entirely new route
                    'timeline_months': 36,
                    'benefits': ['doubles_capacity', 'uses_existing_right_of_way']
                }
            else:
                return {
                    'approach': 'build_new_direct_route',
                    'modifications': ['new_alignment', 'keep_existing_for_local_service'],
                    'estimated_cost_vs_new': 1.0,  # Full new construction
                    'timeline_months': 60,
                    'benefits': ['faster_direct_service', 'separates_express_and_local']
                }
        
        # Existing route is inadequate
        else:
            if metrics['route_directness'] < 0.7:
                return {
                    'approach': 'build_new_route',
                    'reason': 'existing_route_too_indirect',
                    'keep_existing': 'for_local_service',
                    'estimated_cost_vs_new': 1.0,
                    'timeline_months': 60
                }
            elif metrics['track_condition'] == 'poor':
                # Compare upgrade cost vs new construction
                upgrade_cost = self.estimate_upgrade_cost(existing_route)
                new_cost = self.estimate_new_construction_cost(existing_route.origin, existing_route.destination)
                
                if upgrade_cost < new_cost * 0.7:
                    return {
                        'approach': 'major_upgrade',
                        'modifications': ['complete_track_renewal', 'realignment_sections'],
                        'estimated_cost_vs_new': 0.7,
                        'timeline_months': 48
                    }
                else:
                    return {
                        'approach': 'build_new_route',
                        'reason': 'upgrade_not_cost_effective',
                        'keep_existing': 'decommission_or_preserve_for_freight',
                        'estimated_cost_vs_new': 1.0,
                        'timeline_months': 60
                    }
    
    def determine_optimal_approach(self, analysis_results):
        """Select best approach from multiple existing routes"""
        
        if not analysis_results:
            return {'recommendation': 'build_new', 'exists': False}
        
        # Sort by cost-effectiveness
        sorted_options = sorted(
            analysis_results,
            key=lambda x: x.get('estimated_cost_vs_new', 1.0)
        )
        
        best_option = sorted_options[0]
        
        return {
            'exists': True,
            'recommendation': best_option['approach'],
            'best_route': best_option,
            'alternatives': sorted_options[1:3] if len(sorted_options) > 1 else [],
            'comparative_analysis': self.compare_options(sorted_options)
        }
    
    def calculate_directness(self, route):
        """Calculate how direct a route is (1.0 = perfectly straight)"""
        straight_line_distance = haversine_distance(
            route.origin_coords,
            route.destination_coords
        )
        actual_distance = route.total_length_km
        return straight_line_distance / actual_distance
    
    def calculate_current_capacity(self, route):
        """Estimate capacity utilization of existing route"""
        
        # Get current timetable
        current_trains_per_day = len(route.daily_services)
        theoretical_max_capacity = self.calculate_theoretical_capacity(route)
        
        utilization = current_trains_per_day / theoretical_max_capacity
        return utilization
    
    def calculate_theoretical_capacity(self, route):
        """Calculate theoretical maximum trains per day"""
        
        # Based on signaling system and track configuration
        if route.signaling_system == 'ETCS_Level_2':
            headway_minutes = 3
        elif route.signaling_system == 'modern_block':
            headway_minutes = 5
        else:
            headway_minutes = 10
        
        # Operating hours (typically 18-20 hours/day)
        operating_hours = 18
        trains_per_hour = 60 / headway_minutes
        
        # Double for double track
        if route.track_count >= 2:
            trains_per_hour *= 2
        
        return trains_per_hour * operating_hours
```

**Real-World Examples of Infrastructure Reuse Decisions**:

```python
example_scenarios = {
    'scenario_1_lille_antwerp': {
        'query': 'Express line Lille to Antwerp',
        'existing_infrastructure': {
            'route_1': {
                'name': 'Via Brussels (current intercity)',
                'distance_km': 145,
                'max_speed_kmh': 140,
                'journey_time_min': 90,
                'capacity_utilization': 0.75,
                'directness': 0.76,
                'track_condition': 'good'
            }
        },
        'analysis_result': {
            'recommendation': 'upgrade_existing_plus_new_direct',
            'approach': {
                'phase_1': {
                    'action': 'upgrade_existing_route',
                    'improvements': [
                        'increase_speed_to_160_kmh',
                        'add_express_services_skipping_intermediate_stops',
                        'upgrade_signaling'
                    ],
                    'cost': '€200M',
                    'timeline': '18 months',
                    'result': 'journey_time_reduced_to_70_min'
                },
                'phase_2_optional': {
                    'action': 'build_direct_high_speed_line',
                    'specs': [
                        'new_direct_alignment_110km',
                        '250_kmh_design_speed',
                        'shares_stations_in_Lille_and_Antwerp'
                    ],
                    'cost': '€3.5B',
                    'timeline': '60 months',
                    'result': 'journey_time_35_min'
                }
            },
            'recommendation_rationale': 'Phase 1 provides quick wins at low cost. Phase 2 only justified if demand exceeds 5M passengers/year.'
        }
    },
    
    'scenario_2_nairobi_kampala': {
        'query': 'Connect Nairobi to Kampala',
        'existing_infrastructure': {
            'route_1': {
                'name': 'Old Kenya-Uganda Railway (meter gauge)',
                'distance_km': 520,
                'max_speed_kmh': 45,
                'journey_time_hours': 18,
                'capacity_utilization': 0.3,
                'directness': 0.82,
                'track_condition': 'poor',
                'gauge': '1000mm'
            }
        },
        'analysis_result': {
            'recommendation': 'build_new_standard_gauge',
            'approach': {
                'action': 'new_standard_gauge_railway',
                'rationale': [
                    'existing_meter_gauge_incompatible_with_modern_SGR',
                    'track_condition_poor_requiring_complete_replacement',
                    'upgrade_cost_similar_to_new_construction',
                    'new_SGR_enables_freight_integration_with_Mombasa_SGR'
                ],
                'specs': [
                    'standard_gauge_1435mm',
                    'design_speed_120_kmh',
                    'single_track_with_passing_loops',
                    'diesel_electric_operation'
                ],
                'cost': '€4.2B',
                'timeline': '72 months',
                'result': 'journey_time_8_hours'
            },
            'existing_route_disposition': {
                'action': 'preserve_for_heritage_tourism',
                'rationale': 'historic_colonial_era_railway'
            }
        }
    },
    
    'scenario_3_london_heathrow': {
        'query': 'Better rail connection London to Heathrow',
        'existing_infrastructure': {
            'route_1': {
                'name': 'Heathrow Express',
                'distance_km': 25,
                'max_speed_kmh': 160,
                'journey_time_min': 15,
                'capacity_utilization': 0.85,
                'directness': 0.92,
                'track_condition': 'excellent',
                'dedicated': True
            },
            'route_2': {
                'name': 'Piccadilly Line (Underground)',
                'distance_km': 28,
                'max_speed_kmh': 60,
                'journey_time_min': 50,
                'capacity_utilization': 0.95,
                'directness': 0.75,
                'track_condition': 'good',
                'dedicated': False
            }
        },
        'analysis_result': {
            'recommendation': 'integrate_with_crossrail_existing_infrastructure',
            'approach': {
                'action': 'utilize_crossrail_elizabeth_line',
                'implementation': [
                    'completed_2022',
                    'uses_existing_heathrow_express_tracks',
                    'provides_through_service_across_london',
                    'increases_capacity_without_new_construction'
                ],
                'cost': '€0 (already built)',
                'result': 'additional_service_pattern_using_existing_infrastructure'
            },
            'recommendation_rationale': 'Perfect example of leveraging existing infrastructure with new service patterns'
        }
    },
    
    'scenario_4_madrid_barcelona': {
        'query': 'High-speed link Madrid to Barcelona',
        'existing_infrastructure': {
            'route_1': {
                'name': 'Conventional line via Zaragoza',
                'distance_km': 625,
                'max_speed_kmh': 120,
                'journey_time_hours': 7,
                'capacity_utilization': 0.6,
                'directness': 0.78,
                'track_condition': 'fair'
            }
        },
        'analysis_result': {
            'recommendation': 'build_new_high_speed_keep_conventional',
            'approach': {
                'action': 'new_high_speed_line_built',
                'specs': [
                    'entirely_new_alignment',
                    'standard_gauge_1435mm',
                    'design_speed_300_kmh',
                    'distance_615km_more_direct'
                ],
                'cost': '€7.1B (actual)',
                'completion': '2008',
                'result': 'journey_time_2h30min',
                'ridership': '5M_passengers_per_year'
            },
            'existing_route_disposition': {
                'action': 'retained_for_regional_and_freight',
                'benefits': [
                    'serves_intermediate_cities',
                    'freight_traffic',
                    'backup_route'
                ]
            },
            'outcome': {
                'success': 'high',
                'modal_shift': 'captured_70_percent_of_air_market',
                'justification': 'existing_route_too_slow_and_indirect_for_this_corridor'
            }
        }
    }
}
```

### 5.3 Referencing System

**How Inference Uses Learned Reference Patterns**:

```python
class ReferencingEngine:
    """Uses learned patterns to design new railways"""
    
    def __init__(self, learning_database):
        self.db = learning_database
        self.reference_library = self.db.get_reference_patterns()
    
    def find_comparable_projects(self, new_project_spec):
        """Find similar existing projects to use as reference
        
        Matching criteria:
        - Similar terrain
        - Similar country development level
        - Similar distance/population served
        - Similar service type
        """
        
        candidates = []
        
        for reference_project in self.reference_library:
            similarity_score = self.calculate_similarity(
                new_project_spec,
                reference_project
            )
            
            if similarity_score > 0.7:  # 70% similarity threshold
                candidates.append({
                    'project': reference_project,
                    'similarity': similarity_score,
                    'applicable_patterns': self.extract_applicable_patterns(
                        reference_project,
                        new_project_spec
                    )
                })
        
        # Sort by similarity
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        
        return candidates[:5]  # Top 5 matches
    
    def calculate_similarity(self, spec_a, spec_b):
        """Calculate similarity score between two projects"""
        
        scores = []
        
        # Terrain similarity
        terrain_score = self.compare_terrain(spec_a['terrain'], spec_b['terrain'])
        scores.append(('terrain', terrain_score, 0.25))
        
        # Distance similarity
        distance_ratio = min(spec_a['distance_km'], spec_b['distance_km']) / max(spec_a['distance_km'], spec_b['distance_km'])
        scores.append(('distance', distance_ratio, 0.15))
        
        # Speed class similarity
        speed_score = 1.0 if spec_a['speed_class'] == spec_b['speed_class'] else 0.5
        scores.append(('speed', speed_score, 0.2))
        
        # Development level similarity
        dev_score = self.compare_development_levels(
            spec_a['country'],
            spec_b['country']
        )
        scores.append(('development', dev_score, 0.2))
        
        # Population density similarity
        pop_score = self.compare_population_densities(spec_a, spec_b)
        scores.append(('population', pop_score, 0.2))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in scores)
        
        return total_score
    
    def compare_terrain(self, terrain_a, terrain_b):
        """Compare terrain types (0.0 to 1.0)"""
        
        terrain_similarity_matrix = {
            ('flat', 'flat'): 1.0,
            ('flat', 'rolling'): 0.7,
            ('flat', 'mountainous'): 0.3,
            ('rolling', 'rolling'): 1.0,
            ('rolling', 'mountainous'): 0.6,
            ('mountainous', 'mountainous'): 1.0,
            ('coastal', 'coastal'): 1.0,
            ('coastal', 'flat'): 0.8,
            ('desert', 'desert'): 1.0,
            ('desert', 'flat'): 0.7
        }
        
        key = (terrain_a, terrain_b)
        reverse_key = (terrain_b, terrain_a)
        
        return terrain_similarity_matrix.get(key, 
               terrain_similarity_matrix.get(reverse_key, 0.5))
    
    def extract_applicable_patterns(self, reference_project, new_project_spec):
        """Extract specific design patterns applicable to new project"""
        
        patterns = {}
        
        # Station spacing pattern
        if reference_project['terrain'] == new_project_spec['terrain']:
            patterns['station_spacing'] = {
                'recommended_urban_km': reference_project['urban_station_spacing'],
                'recommended_rural_km': reference_project['rural_station_spacing'],
                'rationale': f"Based on {reference_project['name']}"
            }
        
        # Gradient handling
        if reference_project['terrain_difficulty'] >= new_project_spec['terrain_difficulty']:
            patterns['gradient_approach'] = {
                'max_gradient': reference_project['max_gradient_percent'],
                'tunnel_percentage': reference_project['tunnel_percentage'],
                'approach': reference_project['gradient_philosophy'],
                'rationale': f"Similar terrain successfully handled in {reference_project['name']}"
            }
        
        # Construction method
        patterns['construction_methods'] = {
            'at_grade_percentage': reference_project['at_grade_pct'],
            'elevated_percentage': reference_project['elevated_pct'],
            'tunnel_percentage': reference_project['tunnel_pct'],
            'rationale': f"Construction mix proven effective in {reference_project['country']}"
        }
        
        # Cost benchmarking
        patterns['cost_benchmark'] = {
            'reference_cost_per_km': reference_project['cost_per_km'],
            'adjusted_for_region': self.adjust_cost_for_region(
                reference_project['cost_per_km'],
                reference_project['country'],
                new_project_spec['country']
            ),
            'confidence': 'high' if reference_project['similarity'] > 0.85 else 'medium'
        }
        
        return patterns

def reference_selection_example():
    """Example: Lebanon railway project finds references"""
    
    new_project = {
        'name': 'Beirut-Tripoli Railway',
        'country': 'Lebanon',
        'region': 'middle_east',
        'terrain': 'coastal_mountainous',
        'distance_km': 85,
        'speed_class': 'medium_speed',
        'target_speed': 140,
        'development_level': 'developing',
        'budget_constraint': 'high',
        'challenges': ['dense_urban', 'narrow_coastal_corridor', 'mountain_proximity', 'NIMBY']
    }
    
    engine = ReferencingEngine(learning_database)
    references = engine.find_comparable_projects(new_project)
    
    return {
        'top_references': [
            {
                'project': 'Italian Riviera Lines (Genoa-La Spezia)',
                'similarity': 0.89,
                'why_similar': [
                    'coastal_mountainous_terrain',
                    'narrow_corridor_between_sea_and_mountains',
                    'mix_of_tunnels_viaducts_and_at_grade',
                    'similar_distance',
                    'serves_coastal_cities'
                ],
                'applicable_lessons': [
                    'extensive_tunneling_through_headlands',
                    'elevated_sections_along_coast',
                    'station_placement_in_city_centers',
                    'typical_tunnel_ratio_40_percent'
                ]
            },
            {
                'project': 'Athens Suburban Railway (Coastal Section)',
                'similarity': 0.82,
                'why_similar': [
                    'Mediterranean_climate',
                    'coastal_route',
                    'similar_development_level',
                    'integration_with_existing_urban_areas'
                ],
                'applicable_lessons': [
                    'at_grade_through_less_dense_areas',
                    'elevated_urban_sections',
                    'integration_with_metro_systems'
                ]
            },
            {
                'project': 'Nice-Monaco Railway',
                'similarity': 0.85,
                'why_similar': [
                    'french_riviera_similar_terrain',
                    'very_narrow_coastal_strip',
                    'high_land_values',
                    'environmental_sensitivity'
                ],
                'applicable_lessons': [
                    'underground_stations_in_constrained_areas',
                    'extensive_use_of_tunnels_60_percent',
                    'minimal_visual_impact_design'
                ]
            }
        ],
        'synthesized_recommendations': {
            'construction_approach': {
                'tunnels': '45-50% of route',
                'elevated': '25-30% of route',
                'at_grade': '20-25% of route',
                'rationale': 'Average of successful Mediterranean coastal projects'
            },
            'estimated_cost_range': {
                'min': '€35M per km',
                'max': '€55M per km',
                'total_range': '€3.0B - €4.7B',
                'basis': 'Adjusted from Italian and Greek coastal railway costs'
            },
            'timeline': {
                'design': '18-24 months',
                'construction': '60-72 months',
                'total': '78-96 months'
            }
        }
    }
```

### 5.4 Terrain Projections

**Applying Terrain Data to New Route Planning**:

```python
class TerrainProjectionEngine:
    """Projects terrain data onto proposed routes"""
    
    def project_terrain(self, route_geometry, resolution_m=50):
        """Generate elevation profile for proposed route"""
        
        # Sample points along route
        sample_points = self.generate_sample_points(route_geometry, resolution_m)
        
        # Fetch elevation for each point
        elevations = []
        for point in sample_points:
            elevation = self.fetch_elevation(point.lat, point.lon)
            elevations.append({
                'distance_m': point.distance_from_start,
                'elevation_m': elevation,
                'lat': point.lat,
                'lon': point.lon
            })
        
        # Calculate gradients
        gradients = self.calculate_gradients(elevations)
        
        # Identify challenging sections
        challenges = self.identify_challenges(elevations, gradients)
        
        return {
            'elevation_profile': elevations,
            'gradients': gradients,
            'challenges': challenges,
            'statistics': self.calculate_terrain_statistics(elevations, gradients)
        }
    
    def identify_challenges(self, elevations, gradients):
        """Identify sections requiring special construction"""
        
        challenges = []
        
        for i, gradient in enumerate(gradients):
            elevation_change = abs(elevations[i+1]['elevation_m'] - elevations[i]['elevation_m'])
            
            if abs(gradient['percent']) > 3.0:
                challenges.append({
                    'location_km': elevations[i]['distance_m'] / 1000,
                    'type': 'steep_gradient',
                    'gradient_percent': gradient['percent'],
                    'recommended_solution': 'tunnel' if elevation_change > 80 else 'extensive_earthworks',
                    'estimated_cost_multiplier': 2.5 if elevation_change > 80 else 1.8
                })
            
            elif elevation_change > 100 and gradient['percent'] > 2.0:
                challenges.append({
                    'location_km': elevations[i]['distance_m'] / 1000,
                    'type': 'major_elevation_change',
                    'elevation_change_m': elevation_change,
                    'recommended_solution': 'viaduct_or_tunnel',
                    'estimated_cost_multiplier': 2.0
                })
        
        return challenges
```

### 5.5 Train Station Positioning & Accessibility

**Station Placement Algorithm**:

```python
class StationPlacementEngine:
    """Determines optimal station locations"""
    
    def position_stations(self, route_geometry, country_context, service_type):
        """Generate optimal station positions along route"""
        
        # Get reference station spacing for this service type
        reference_spacing = self.get_reference_spacing(service_type, country_context)
        
        # Get population density along route
        population_data = self.get_population_along_route(route_geometry)
        
        # Get existing transport nodes
        existing_nodes = self.get_existing_transport_nodes(route_geometry)
        
        # Generate candidate station locations
        candidates = self.generate_station_candidates(
            route_geometry,
            population_data,
            existing_nodes,
            reference_spacing
        )
        
        # Optimize station selection
        selected_stations = self.optimize_station_selection(candidates, service_type)
        
        # Determine station types and construction methods
        station_specs = []
        for station in selected_stations:
            spec = self.specify_station(station, route_geometry, country_context)
            station_specs.append(spec)
        
        return station_specs
    
    def specify_station(self, station_location, route_geometry, country_context):
        """Determine station specifications"""
        
        population_catchment = self.calculate_catchment(station_location)
        land_use = self.get_land_use(station_location)
        terrain = self.get_terrain(station_location)
        
        # Determine construction type
        if land_use['urban_density'] > 5000 and land_use['available_space'] < 5000:
            # Dense urban area, limited space
            construction_type = 'underground'
            platform_count = 2
            estimated_cost = '€80M'
        
        elif land_use['urban_density'] > 3000:
            # Urban area with some space
            if terrain['elevation_vs_surroundings'] > 5:
                construction_type = 'elevated'
                platform_count = 2
                estimated_cost = '€25M'
            else:
                construction_type = 'at_grade_with_overpass'
                platform_count = 2
                estimated_cost = '€15M'
        
        else:
            # Suburban or rural
            construction_type = 'at_grade'
            platform_count = 2
            estimated_cost = '€8M'
        
        # Accessibility features (based on country standards)
        accessibility = self.determine_accessibility_features(country_context)
        
        # Integration with other transport
        integration = self.plan_transport_integration(station_location)
        
        return {
            'location': station_location,
            'name': self.suggest_station_name(station_location),
            'construction_type': construction_type,
            'platform_count': platform_count,
            'platform_length_m': 200 if service_type == 'regional' else 400,
            'estimated_cost': estimated_cost,
            'catchment_population_15min': population_catchment['15min'],
            'catchment_population_30min': population_catchment['30min'],
            'accessibility_features': accessibility,
            'integration': integration
        }
    
    def plan_transport_integration(self, station_location):
        """Plan integration with other transport modes"""
        
        nearby_bus_routes = self.find_nearby_bus_routes(station_location, radius_m=500)
        nearby_metro = self.find_nearby_metro(station_location, radius_m=300)
        
        integration_plan = {
            'bus': {
                'existing_routes': len(nearby_bus_routes),
                'recommendation': 'create_bus_terminal' if len(nearby_bus_routes) > 5 else 'bus_stops',
                'estimated_cost': '€2M' if len(nearby_bus_routes) > 5 else '€200k'
            },
            'metro': {
                'nearby_metro_station': nearby_metro['name'] if nearby_metro else None,
                'walking_distance_m': nearby_metro['distance'] if nearby_metro else None,
                'recommendation': 'covered_walkway' if nearby_metro and nearby_metro['distance'] < 200 else 'signage'
            },
            'park_and_ride': {
                'recommended': True if station_location['urban_density'] < 2000 else False,
                'capacity': 500 if station_location['urban_density'] < 2000 else 0,
                'estimated_cost': '€2.5M' if station_location['urban_density'] < 2000 else '€0'
            },
            'bike_parking': {
                'recommended_capacity': min(200, int(station_location['population_nearby'] / 100)),
                'estimated_cost': '€50k'
            }
        }
        
        return integration_plan
```

### 5.6 Speculative Rail Plotting

**Route Generation Algorithm**:

```python
class RouteGenerationEngine:
    """Generates optimal railway alignments"""
    
    def generate_route(self, origin, destination, service_type, constraints, learned_patterns):
        """Generate railway route considering all factors"""
        
        # Generate multiple candidate routes
        candidates = []
        
        # Candidate 1: Most direct route
        direct_route = self.generate_direct_route(origin, destination)
        candidates.append(self.evaluate_route(direct_route, 'direct', constraints, learned_patterns))
        
        # Candidate 2: Terrain-optimized route
        terrain_route = self.generate_terrain_optimized_route(origin, destination, learned_patterns)
        candidates.append(self.evaluate_route(terrain_route, 'terrain_optimized', constraints, learned_patterns))
        
        # Candidate 3: Population-optimized route (serves more cities)
        pop_route = self.generate_population_optimized_route(origin, destination)
        candidates.append(self.evaluate_route(pop_route, 'population_optimized', constraints, learned_patterns))
        
        # Candidate 4: Existing corridor route (if available)
        if self.existing_corridor_available(origin, destination):
            corridor_route = self.generate_corridor_route(origin, destination)
            candidates.append(self.evaluate_route(corridor_route, 'existing_corridor', constraints, learned_patterns))
        
        # Score and rank candidates
        ranked_candidates = self.rank_routes(candidates, constraints)
        
        # Return best route with alternatives
        return {
            'recommended': ranked_candidates[0],
            'alternatives': ranked_candidates[1:3],
            'comparison': self.compare_routes(ranked_candidates)
        }
    
    def generate_terrain_optimized_route(self, origin, destination, learned_patterns):
        """Generate route that minimizes terrain challenges"""
        
        # Get terrain data
        terrain = self.get_terrain_between_points(origin, destination)
        
        # Apply learned gradient philosophy from similar projects
        gradient_philosophy = learned_patterns['gradient_approach']
        max_acceptable_gradient = gradient_philosophy['max_gradient']
        
        # Use A* algorithm with terrain-aware cost function
        route = self.astar_terrain_aware(
            origin,
            destination,
            terrain,
            max_gradient=max_acceptable_gradient
        )
        
        return route
    
    def astar_terrain_aware(self, start, goal, terrain, max_gradient):
        """A* pathfinding with terrain cost weighting
            ,
        {"type": "bridge", "percentage": 8},
        {"type": "tunnel", "percentage": 7}
      ],
      "terrain_profile": {
        "avg_gradient": 0.8,
        "max_gradient": 2.1,
        "elevation_gain_m": 120,
        "terrain_type": "flat_coastal"
      }
    }
  ],
  "stations": [...],
  "timetable": {...},
  "cost_data": {...},
  "comparable_lines": ["NL-IC-Rotterdam-Amsterdam", "DK-IC-Copenhagen-Aarhus"]
}
```

#### 3.2.2 Station Schema
```json
{
  "station_id": "STN-12345",
  "name": "Central Station",
  "location": {"lat": 51.0357, "lon": 3.7103},
  "country": "Country Name",
  "station_type": "major_hub",
  "platforms": 14,
  "platform_config": [
    {"platform_id": 1, "track_numbers": [1, 2], "length_m": 450},
    {"platform_id": 2, "track_numbers": [3, 4], "length_m": 450}
  ],
  "accessibility": {
    "elevator": true,
    "tactile_paving": true,
    "wheelchair_accessible": true
  },
  "construction_type": "at_grade",
  "connections": {
    "bus_lines": 15,
    "tram_lines": 4,
    "metro_lines": 2,
    "park_and_ride_capacity": 500
  },
  "catchment_area": {
    "walking_15min_population": 45000,
    "cycling_15min_population": 120000,
    "transit_30min_population": 280000
  },
  "existing_infrastructure_quality": "good",
  "upgrade_potential": "high"
}
```

#### 3.2.3 Terrain Data Schema
```json
{
  "tile_id": "N51E003",
  "bounds": {
    "north": 51.5,
    "south": 51.0,
    "east": 3.5,
    "west": 3.0
  },
  "resolution_m": 30,
  "elevation_data": "...",
  "geology": {
    "primary_type": "alluvial_deposits",
    "soil_bearing_capacity_kpa": 150,
    "groundwater_depth_m": 2.5,
    "seismic_zone": "low"
  },
  "land_use": {
    "urban": 35,
    "agricultural": 50,
    "forest": 10,
    "water": 5
  },
  "constraints": [
    {
      "type": "archaeological_site",
      "protection_level": "high",
      "geometry": "POLYGON(...)"
    }
  ]
}
```

### 3.3 Data Processing Pipeline

#### 3.3.1 Ingestion Stage
```
Raw Data → Validation → Format Conversion → Initial Storage
```

**Process Steps:**
1. **Fetch**: Pull data from APIs/databases on schedule
2. **Validate**: Check completeness, format, coordinate systems
3. **Normalize**: Convert different standards (track gauge, signaling systems, etc.)
4. **Convert**: Transform to internal format (GeoJSON, PostgreSQL/PostGIS)
5. **Store**: Save to appropriate database with metadata

#### 3.3.2 Enrichment Stage
```
Basic Data → Enrichment → Enhanced Data
```

**Enrichment Types:**
- **Geometric Enrichment**: Calculate curves, gradients, distances
- **Topological Enrichment**: Identify connections, junctions, crossovers
- **Statistical Enrichment**: Compute averages, distributions, outliers
- **Contextual Enrichment**: Add demographic, economic context
- **Comparative Enrichment**: Link to similar lines in other countries

#### 3.3.3 Feature Extraction Stage
```
Enhanced Data → Feature Engineering → Feature Vectors
```

**Extracted Features:**
- **Line Characteristics**: Average speed, curvature, gradient
- **Station Patterns**: Spacing, size distribution, type classification
- **Network Topology**: Hub identification, connectivity metrics
- **Cost Patterns**: Per-km costs by terrain, construction type, and country
- **Regional Patterns**: Design preferences by country/region (e.g., European preference for at-grade vs. Japanese preference for elevated)

---

## 4. Learning Phase

### 4.1 Train Line Plotting

#### 4.1.1 Route Geometry Extraction

**Objective**: Convert raw railway data from any country into structured geometric representations.

**Process**:
1. **Data Acquisition**
   - Query OpenStreetMap for `railway=rail` ways globally
   - Fetch official railway network shapefiles from national databases
   - Cross-reference multiple sources for accuracy
   - Handle different coordinate systems (WGS84, local systems)

2. **Geometry Processing**
   - Convert to consistent coordinate system (WGS84)
   - Simplify geometry while preserving key features (Douglas-Peucker algorithm)
   - Identify straight sections, curves, and critical points
   - Handle different track gauges (standard 1435mm, broad, narrow, etc.)

3. **Segmentation**
   - Split lines at stations and junctions
   - Create section objects with start/end points
   - Calculate section lengths and bearings

**Code Example**:
```python
def extract_line_geometry(line_id, country, data_source):
    """Extract and process railway line geometry from any country"""
    
    # Fetch raw geometry
    raw_geometry = fetch_railway_data(line_id, country, data_source)
    
    # Handle country-specific data formats
    geometry = normalize_geometry(raw_geometry, country)
    
    # Convert to LineString
    line = shapely.geometry.LineString(geometry)
    
    # Simplify while preserving topology
    simplified = line.simplify(tolerance=5, preserve_topology=True)
    
    # Split at stations
    stations = get_stations_on_line(line_id)
    sections = split_line_at_points(simplified, stations)
    
    return {
        'line_id': line_id,
        'country': country,
        'geometry': simplified,
        'sections': sections,
        'total_length_km': line.length / 1000,
        'gauge': get_track_gauge(line_id),
        'comparable_lines': find_similar_lines_globally(simplified, country)
    }
```

#### 4.1.2 Line Type Classification

**Speed Classes (Global Standards)**:
- **Very High Speed**: 250-320 km/h (Shinkansen, TGV, ICE, etc.)
- **High Speed**: 200-250 km/h (Many European and Asian corridors)
- **Medium Speed**: 120-200 km/h (Intercity services globally)
- **Standard Speed**: 80-120 km/h (Regional trains, varied terrain)
- **Low Speed**: 40-80 km/h (Branch lines, mountainous terrain, developing regions)
- **Urban Speed**: 30-60 km/h (Urban rail, trams, metros)

**Classification Algorithm**:
```python
def classify_line_section(section_data, country_context):
    """Classify railway section by operational characteristics
    
    Takes into account country-specific standards and development level
    """
    
    max_speed = section_data['max_speed_kmh']
    curvature = calculate_curvature(section_data['geometry'])
    gradient = section_data['terrain_profile']['max_gradient']
    development_level = country_context['railway_maturity']
    
    # Rule-based classification with regional adjustments
    if max_speed >= 250 and curvature < 0.5 and gradient < 1.5:
        return "very_high_speed"
    elif max_speed >= 200 and curvature < 0.8 and gradient < 2.0:
        return "high_speed"
    elif max_speed >= 120 and curvature < 1.5 and gradient < 2.5:
        return "medium_speed"
    elif gradient > 3.0 or curvature > 2.5:
        return "mountain_climbing"
    elif development_level == "developing" and max_speed < 80:
        return "basic_service"
    else:
        return "standard_speed"
```

#### 4.1.3 Train Table Extraction

**Objective**: Extract operational schedules globally to understand service patterns across different countries.

**Data Points Collected**:
- Departure/arrival times at each station
- Dwell time at stations
- Journey time between stations
- Service frequency (trains per hour/day)
- Peak vs off-peak patterns
- Integration with other transport modes

**Example Train Table Structure**:
```json
{
  "train_id": "IC532",
  "operator": "National Operator",
  "line": "City A - City B",
  "country": "Country Name",
  "service_type": "intercity",
  "schedule": [
    {"station": "City A Central", "arrival": null, "departure": "08:03", "dwell_sec": 0},
    {"station": "Town B", "arrival": "08:18", "departure": "08:20", "dwell_sec": 120},
    {"station": "City C", "arrival": "08:40", "departure": "08:42", "dwell_sec": 120},
    {"station": "Capital Station", "arrival": "09:15", "departure": "09:18", "dwell_sec": 180},
    {"station": "City D", "arrival": "09:35", "departure": "09:37", "dwell_sec": 120},
    {"station": "City E", "arrival": "10:15", "departure": "10:18", "dwell_sec": 180},
    {"station": "Town F", "arrival": "10:35", "departure": "10:37", "dwell_sec": 120},
    {"station": "City B Terminal", "arrival": "10:52", "departure": null, "dwell_sec": 0}
  ],
  "total_journey_time_min": 169,
  "frequency_peak": "every_30_min",
  "frequency_offpeak": "every_60_min"
}
```

**Cross-Country Analysis**:
- **Japan**: Very high frequency (3-5 min headways on busy lines)
- **Europe**: Clockface scheduling (departures at same minute each hour)
- **North America**: Lower frequency, longer distances between stops
- **Developing Regions**: Limited frequency, may be daily rather than hourly

#### 4.1.4 Route Formatting & Categorization

**Track Construction Types (Global)**:

1. **At-Grade (Directly on Ground)**
   - Most common globally, cost-effective
   - Requires level crossings or grade separation
   - Typical cost: €3-15M per km (varies by country labor costs)

2. **Ballasted Track**
   - Traditional stone ballast bed
   - Easier maintenance, better drainage
   - Preferred for speeds < 200 km/h worldwide

3. **Slab Track (Concrete)**
   - Used for high-speed lines globally
   - Lower maintenance, higher initial cost
   - Standard for Japanese Shinkansen, European HSR

4. **Elevated/Viaduct**
   - Crosses valleys, wetlands, urban areas
   - Common in space-constrained Asian cities
   - Typical cost: €20-80M per km (location dependent)

5. **Tunnel**
   - Mountainous terrain, urban cores
   - Highest cost but shortest routes
   - Typical cost: €50-200M per km (geology and country dependent)
   - Examples: Swiss Alpine tunnels, Japanese mountain routes, urban metros

6. **Bridge**
   - River crossings, valleys
   - Cost depends on span length and local conditions
   - Typical cost: €15-60M per km

**Regional Construction Preferences**:
- **Western Europe**: Mix of at-grade and elevated, extensive tunneling in Alps
- **Japan**: Extensive elevated sections in urban areas, tunnels in mountains
- **China**: Rapid construction using viaducts for flat terrain
- **Switzerland**: Focus on base tunnels through mountains
- **Scandinavia**: At-grade with rock cuttings, minimal tunneling costs due to stable geology
- **Developing Nations**: Primarily at-grade to minimize costs

**Categorization Process**:
```python
def categorize_construction_type(section, terrain_data, country_context):
    """Determine construction type for route section
    
    Considers local construction practices and economic factors
    """
    
    elevation_profile = get_elevation_along_line(section.geometry)
    land_use = get_land_use(section.geometry)
    urban_density = get_population_density(section.geometry)
    construction_costs = country_context['construction_cost_index']
    local_practice = country_context['preferred_construction_methods']
    
    construction_breakdown = []
    
    for i, segment in enumerate(section.segments):
        elevation_change = abs(elevation_profile[i+1] - elevation_profile[i])
        
        if land_use[i] == 'water':
            construction_breakdown.append({
                'segment': i,
                'type': 'bridge',
                'reason': 'water_crossing',
                'cost_factor': construction_costs['bridge']
            })
        elif elevation_change > 50 and terrain_data[i]['slope'] > 15:
            # Check local preference: tunnel vs. switchback
            if 'tunnel_preference' in local_practice:
                construction_breakdown.append({
                    'segment': i,
                    'type': 'tunnel',
                    'reason': 'steep_terrain',
                    'cost_factor': construction_costs['tunnel']
                })
            else:
                construction_breakdown.append({
                    'segment': i,
                    'type': 'switchback_climbing',
                    'reason': 'steep_terrain_budget_constrained',
                    'cost_factor': construction_costs['at_grade'] * 1.5
                })
        elif urban_density[i] > 3000:
            # Asian cities often prefer elevated, European cities prefer underground
            if country_context['region'] == 'asia' and 'elevated_preference' in local_practice:
                construction_type = 'elevated'
            else:
                construction_type = 'tunnel' if construction_costs['tunnel'] < construction_costs['elevated'] * 2 else 'elevated'
            
            construction_breakdown.append({
                'segment': i,
                'type': construction_type,
                'reason': 'urban_area',
                'cost_factor': construction_costs[construction_type]
            })
        else:
            construction_breakdown.append({
                'segment': i,
                'type': 'at_grade',
                'reason': 'standard',
                'cost_factor': construction_costs['at_grade']
            })
    
    return construction_breakdown
```

#### 4.1.5 Terrain Overlay

**Objective**: Understand how terrain influences route design decisions across different geographical and cultural contexts.

**Process**:
1. **Elevation Profile Generation**
   - Sample elevation every 50m along route
   - Calculate gradients between points
   - Identify peaks, valleys, plateaus, mountain passes

2. **Gradient Analysis**
   ```python
   def analyze_gradients(elevation_profile, distances, country_standards):
       """Calculate gradient statistics for route
       
       Consider country-specific gradient tolerances
       """
       gradients = []
       for i in range(len(elevation_profile) - 1):
           rise = elevation_profile[i+1] - elevation_profile[i]
           run = distances[i+1] - distances[i]
           gradient_percent = (rise / run) * 100
           gradients.append(gradient_percent)
       
       # Compare to local standards
       max_acceptable = country_standards['max_gradient']
       steep_threshold = country_standards['steep_threshold']
       
       return {
           'max_gradient': max(gradients),
           'avg_gradient': np.mean(gradients),
           'steep_sections': [i for i, g in enumerate(gradients) if abs(g) > steep_threshold],
           'exceeds_standard': any(abs(g) > max_acceptable for g in gradients),
           'comparison_to_standard': max(gradients) / max_acceptable
       }
   ```

3. **Terrain-Route Correlation**
   - Compare actual route with "straight-line" alternative
   - Identify detours taken to avoid steep grades
   - Learn regional terrain avoidance patterns

**Regional Terrain Handling Patterns**:

**Example 1: Flat Terrain (Netherlands, Denmark, North German Plain)**
```
Pattern: Direct routes prioritized
- Minimal elevation change (<50m over 100km)
- Routes follow shortest paths
- Focus on avoiding water crossings
- Station placement based on population centers only
```

**Example 2: Rolling Hills (France, Southern England, US Midwest)**
```
Pattern: Moderate detours to reduce gradients
- Will add 10-20% distance to keep gradients below 1.5%
- Uses cuttings and embankments to smooth terrain
- Follows valley floors where possible
```

**Example 3: Mountains (Switzerland, Austria, Japan, Andes)**
```
Swiss Approach:
- Base tunnels through mountains (Gotthard, Lötschberg)
- Accepts high construction costs for optimal gradients (<1.2%)
- Minimizes operating costs over lifetime

Japanese Approach:
- Extensive tunneling (60-80% of route in mountains)
- Accepts 3% gradients in some sections
- Balances construction vs. operating costs

Developing Country Approach (Andes, Himalayas):
- Switchback designs to reduce tunnel costs
- Gradients up to 4-5% accepted
- Longer routes, slower speeds, lower construction costs
```

**Example 4: Coastal Routes (Mediterranean, Pacific Coast)**
```
Pattern: Follows coastline with strategic tunneling
- Hugs shoreline to serve coastal cities
- Tunnels through headlands/promontories
- Elevated sections over beaches/cliffs
- Example: Italian Riviera lines, California Coast proposals
```

#### 4.1.6 Platform Logging & Crossovers

**Platform Configuration Analysis (Global)**:

**Data Collected**:
- Number of platforms per station
- Track assignments (through tracks vs. terminating tracks)
- Crossover locations (where trains can switch tracks)
- Platform length and curvature
- Platform height standards (varies by country/region)

**Global Station Typologies**:

1. **Mega-Hubs** (Tokyo Station, Paris Gare du Nord, London King's Cross, Grand Central NYC)
   - 15+ platforms
   - Multiple terminal and through platforms
   - Extensive crossover networks
   - Dedicated platforms for different services (local, express, high-speed)
   - Underground and surface levels

2. **Major Hubs** (Most capital city stations)
   - 8-15 platforms
   - Mix of terminal and through platforms
   - Strategic crossovers for operational flexibility
   - Integration with urban transit

3. **Regional Hubs**
   - 4-8 platforms
   - Primarily through platforms
   - Basic crossover facilities
   - Some terminating services

4. **Standard Stations**
   - 2-4 platforms
   - Simple island or side platforms
   - Limited crossovers
   - Through services only

5. **Basic Stops** (Common in developing regions)
   - 1-2 platforms
   - No crossovers
   - Minimal facilities

**Regional Platform Standards**:
- **Europe**: Typically 55-76cm height, 200-400m length
- **Japan**: 110-130cm height (very high), 200-300m length
- **North America**: 1220mm (48") high-level or 200mm (8") low-level, 200-600m length
- **China**: Standardizing on 125cm height for HSR

**Platform Stopping Logic**:
```python
def validate_platform_assignment(train_schedule, station_config, country_standards):
    """Check if platform assignments follow logical patterns
    
    Considers country-specific operational practices
    """
    
    for stop in train_schedule:
        station = station_config[stop['station']]
        platform = stop['platform_number']
        direction = stop['direction']
        train_length = stop['train_length_m']
        service_type = stop['service_type']
        
        # Check if platform is suitable for direction
        if direction == 'north' and platform in station['southbound_platforms']:
            return {
                'valid': False,
                'reason': 'Platform assigned for wrong direction'
            }
        
        # Check if train length fits platform (with safety margin)
        if train_length > station['platforms'][platform]['length'] - 10:
            return {
                'valid': False,
                'reason': 'Platform too short for train'
            }
        
        # Check service type compatibility
        if service_type == 'high_speed' and platform not in station['high_speed_platforms']:
            return {
                'valid': False,
                'reason': 'High-speed train on incompatible platform'
            }
        
        # Regional practice checks
        if country_standards['requires_platform_screen_doors'] and not station['platforms'][platform]['has_screen_doors']:
            return {
                'valid': False,
                'reason': 'Platform lacks required safety features'
            }
    
    return {'valid': True}
```

#### 4.1.7 Cost Logging

**Historical Cost Data Collection (Global)**:

**Data Sources**:
- Railway Gazette International
- National audit office reports worldwide
- World Bank infrastructure reports
- Academic infrastructure studies
- Company annual reports
- Regional development banks (EIB, ADB, AfDB, IDB)

**Cost Categories (Adjusted by Country)**:
1. **Engineering & Design**: 5-15% of construction (higher in developing countries due to foreign expertise)
2. **Land Acquisition**: 
   - Developed urban areas: €5-100M per km
   - Developed rural areas: €0.5-5M per km
   - Developing countries: €0.1-2M per km (but often with complications)
3. **Track Construction**: 
   - **At-grade**: 
     - Developed countries: €5-15M per km
     - Developing countries: €2-8M per km
   - **Elevated**: 
     - Developed countries: €25-80M per km
     - Developing countries: €15-40M per km
   - **Tunnel**: 
     - Soft ground: €50-100M per km
     - Rock (TBM): €80-150M per km
     - Rock (drill & blast in low-cost regions): €30-80M per km
4. **Electrification**: 
   - €1-3M per km (relatively consistent globally)
5. **Signaling**: 
   - Basic: €1-2M per km
   - ETCS/Advanced: €2-5M per km
6. **Stations**: 
   - Basic stop: €0.5-2M
   - Small station: €5-15M
   - Medium station: €15-50M
   - Major hub: €50-500M
   - Mega project (Grand Paris Express stations): €200M-€1B each
7. **Rolling Stock**: €3-10M per train unit (separate operational budget)

**Regional Cost Variations**:
```python
cost_multipliers = {
    'western_europe': 1.0,  # Baseline
    'northern_europe': 1.2,
    'southern_europe': 0.8,
    'eastern_europe': 0.5,
    'north_america': 1.5,  # High due to regulatory complexity
    'japan': 1.3,
    'china': 0.4,  # Efficient mass construction
    'india': 0.3,
    'southeast_asia': 0.4,
    'middle_east': 0.9,
    'latin_america': 0.5,
    'africa': 0.4
}