# File: railway_ai/config.py
"""
Railway AI Configuration Management
Centralized configuration for all railway intelligence components.
"""
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

class TrainType(Enum):
    S_BAHN = "S"
    REGIONAL_BAHN = "RB"
    REGIONAL_EXPRESS = "RE"
    INTERCITY = "IC"
    INTERCITY_EXPRESS = "ICE"
    EUROCITY = "EC"
    EUROSTAR = "EST"
    FREIGHT = "FREIGHT"

class OptimizationTarget(Enum):
    COST = "cost"
    TIME = "time"
    RIDERSHIP = "ridership"
    ENVIRONMENTAL = "environmental"
    FEASIBILITY = "feasibility"

@dataclass
class PathConfig:
    """File and directory paths configuration"""
    data_dir: Path = Path("data")
    input_dir: Path = Path("data/input")
    output_dir: Path = Path("data/outputs")
    models_dir: Path = Path("data/models")
    cache_dir: Path = Path("data/cache")
    logs_dir: Path = Path("logs")
    
    def __post_init__(self):
        """Ensure paths are Path objects"""
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, str):
                setattr(self, field_name, Path(value))

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "railway_ai"
    username: str = "railway_user"
    password: str = ""
    connection_timeout: int = 30
    pool_size: int = 5

@dataclass
class APIConfig:
    """External API configuration"""
    osm_overpass_url: str = "http://overpass-api.de/api/interpreter"
    elevation_api_url: str = "https://api.open-elevation.com/api/v1/lookup"
    weather_api_key: str = ""
    geocoding_api_key: str = ""
    request_timeout: int = 30
    max_retries: int = 3
    rate_limit_requests_per_minute: int = 60

@dataclass
class MLConfig:
    """Machine learning configuration"""
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    hyperparameter_tuning: bool = False
    model_cache_enabled: bool = True
    feature_selection_enabled: bool = True
    ensemble_methods: bool = True
    auto_scaling: bool = True
    
    # Model-specific parameters
    random_forest_n_estimators: int = 100
    gradient_boosting_learning_rate: float = 0.1
    clustering_max_clusters: int = 10
    pca_n_components: Optional[int] = None

@dataclass
class TrainingConfig:
    """Model training configuration"""
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    learning_rate: float = 0.001
    validation_split: float = 0.2
    save_best_model: bool = True
    model_checkpoint_frequency: int = 10

@dataclass
class EngineeringStandards:
    """Railway engineering standards and constraints"""
    # Gradient limits by train type (percentage)
    max_gradients: Dict[str, float] = None
    
    # Minimum curve radii (meters)
    min_curve_radii: Dict[str, int] = None
    
    # Speed limits (km/h)
    max_speeds: Dict[str, int] = None
    
    # Platform requirements
    min_platform_length: int = 120  # meters
    max_platform_length: int = 400  # meters
    standard_platform_heights: List[float] = None  # meters
    
    # Track specifications
    standard_gauge: int = 1435  # mm
    min_track_spacing: float = 4.0  # meters
    max_cant: int = 150  # mm
    
    # Safety parameters
    min_signal_spacing: int = 1000  # meters
    emergency_brake_distance: int = 1500  # meters
    
    def __post_init__(self):
        """Initialize default values"""
        if self.max_gradients is None:
            self.max_gradients = {
                "ICE": 2.5, "IC": 3.5, "RE": 4.0, "RB": 4.0, 
                "S": 4.0, "FREIGHT": 2.0, "EC": 3.0, "EST": 2.5
            }
        
        if self.min_curve_radii is None:
            self.min_curve_radii = {
                "ICE": 3500, "IC": 1000, "RE": 600, "RB": 400,
                "S": 300, "FREIGHT": 400, "EC": 1200, "EST": 4000
            }
        
        if self.max_speeds is None:
            self.max_speeds = {
                "ICE": 320, "IC": 200, "RE": 160, "RB": 120,
                "S": 120, "FREIGHT": 120, "EC": 200, "EST": 320
            }
        
        if self.standard_platform_heights is None:
            self.standard_platform_heights = [0.55, 0.76, 0.96]  # meters

@dataclass
class CostParameters:
    """Cost estimation parameters"""
    # Construction costs per km (EUR)
    track_costs: Dict[str, int] = None
    
    # Station costs (EUR)
    station_base_cost: int = 5_000_000
    platform_cost_per_meter: int = 5_000
    station_complexity_multipliers: Dict[str, float] = None
    
    # Operational costs per km (EUR/year)
    operational_costs: Dict[str, float] = None
    
    # Regional cost factors
    regional_multipliers: Dict[str, float] = None
    
    # Environmental mitigation costs
    noise_barrier_cost_per_km: int = 500_000
    wildlife_crossing_cost: int = 2_000_000
    wetland_mitigation_per_hectare: int = 50_000
    
    def __post_init__(self):
        """Initialize default cost parameters"""
        if self.track_costs is None:
            self.track_costs = {
                "surface": 2_000_000,
                "elevated": 8_000_000,
                "tunnel": 25_000_000,
                "bridge": 12_000_000,
                "cutting": 4_000_000,
                "embankment": 3_000_000
            }
        
        if self.station_complexity_multipliers is None:
            self.station_complexity_multipliers = {
                "local": 1.0, "regional": 1.5, "intercity": 2.0, 
                "major": 3.0, "international": 4.0
            }
        
        if self.operational_costs is None:
            self.operational_costs = {
                "S": 8.50, "RB": 6.20, "RE": 7.80, "IC": 12.40,
                "ICE": 18.60, "EC": 15.20, "EST": 25.80, "FREIGHT": 4.50
            }
        
        if self.regional_multipliers is None:
            self.regional_multipliers = {
                "DE": 1.0, "FR": 1.1, "CH": 1.8, "UK": 1.4,
                "BE": 1.2, "NL": 1.3, "AT": 1.1, "IT": 0.9
            }

@dataclass
class EnvironmentalConfig:
    """Environmental impact and mitigation configuration"""
    protected_areas_buffer_km: float = 1.0
    noise_sensitive_areas_buffer_km: float = 0.5
    wetlands_buffer_km: float = 0.2
    wildlife_corridors_min_width_m: int = 50
    
    # Emission factors
    electric_train_co2_g_per_pkm: float = 14  # grams CO2 per passenger-km
    diesel_train_co2_g_per_pkm: float = 41
    car_co2_g_per_pkm: float = 120
    
    # Environmental scoring weights
    habitat_fragmentation_weight: float = 0.3
    noise_impact_weight: float = 0.25
    air_quality_weight: float = 0.2
    water_impact_weight: float = 0.15
    visual_impact_weight: float = 0.1

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    console_enabled: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 5
    log_sql_queries: bool = False

@dataclass
class PerformanceConfig:
    """Performance and optimization configuration"""
    enable_multiprocessing: bool = True
    max_workers: Optional[int] = None  # None = auto-detect CPU count
    memory_limit_gb: Optional[float] = None
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    batch_processing_size: int = 1000
    
    # Optimization settings
    optimization_timeout_minutes: int = 60
    max_route_alternatives: int = 5
    station_placement_iterations: int = 100
    genetic_algorithm_generations: int = 50

@dataclass
class ValidationConfig:
    """Validation and testing configuration"""
    enable_cross_validation: bool = True
    validation_split: float = 0.2
    test_scenarios_dir: Path = Path("tests/scenarios")
    benchmark_tolerance_percent: float = 5.0
    
    # Quality thresholds
    min_model_accuracy: float = 0.8
    min_r2_score: float = 0.7
    max_prediction_error_percent: float = 15.0
    
    def __post_init__(self):
        if isinstance(self.test_scenarios_dir, str):
            self.test_scenarios_dir = Path(self.test_scenarios_dir)

@dataclass
class RailwayConfig:
    """Main configuration class containing all settings"""
    paths: PathConfig = None
    database: DatabaseConfig = None
    api: APIConfig = None
    ml: MLConfig = None
    training: TrainingConfig = None
    engineering: EngineeringStandards = None
    costs: CostParameters = None
    environmental: EnvironmentalConfig = None
    logging: LoggingConfig = None
    performance: PerformanceConfig = None
    validation: ValidationConfig = None
    
    # Global settings
    version: str = "1.0.0"
    debug_mode: bool = False
    country_default: str = "DE"
    language: str = "en"
    timezone: str = "UTC"
    
    def __post_init__(self):
        """Initialize default configurations"""
        if self.paths is None:
            self.paths = PathConfig()
        if self.database is None:
            self.database = DatabaseConfig()
        if self.api is None:
            self.api = APIConfig()
        if self.ml is None:
            self.ml = MLConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.engineering is None:
            self.engineering = EngineeringStandards()
        if self.costs is None:
            self.costs = CostParameters()
        if self.environmental is None:
            self.environmental = EnvironmentalConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
        if self.validation is None:
            self.validation = ValidationConfig()
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.paths.data_dir,
            self.paths.input_dir,
            self.paths.output_dir,
            self.paths.models_dir,
            self.paths.cache_dir,
            self.paths.logs_dir,
            self.validation.test_scenarios_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_to_file(self, file_path: Union[str, Path]):
        """Save configuration to file (JSON or YAML)"""
        file_path = Path(file_path)
        config_dict = self.to_dict()
        
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'RailwayConfig':
        """Load configuration from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Load data based on file extension
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            with open(file_path, 'r') as f:
                config_data = json.load(f)
        
        return cls.from_dict(config_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        def convert_value(value):
            if isinstance(value, Path):
                return str(value)
            elif isinstance(value, Enum):
                return value.value
            elif hasattr(value, '__dict__'):
                return {k: convert_value(v) for k, v in asdict(value).items()}
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [convert_value(item) for item in value]
            else:
                return value
        
        return convert_value(asdict(self))
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RailwayConfig':
        """Create configuration from dictionary"""
        # Helper function to convert paths
        def convert_paths(data, path_fields):
            if isinstance(data, dict):
                for key, value in data.items():
                    if key in path_fields and isinstance(value, str):
                        data[key] = Path(value)
                    elif isinstance(value, dict):
                        convert_paths(value, path_fields)
        
        # Convert path strings to Path objects
        path_fields = {
            'data_dir', 'input_dir', 'output_dir', 'models_dir', 
            'cache_dir', 'logs_dir', 'test_scenarios_dir'
        }
        convert_paths(config_dict, path_fields)
        
        # Convert enum strings back to enums
        if 'logging' in config_dict and 'level' in config_dict['logging']:
            level_str = config_dict['logging']['level']
            if isinstance(level_str, str):
                config_dict['logging']['level'] = LogLevel(level_str)
        
        # Create nested configuration objects
        config_kwargs = {}
        
        if 'paths' in config_dict:
            config_kwargs['paths'] = PathConfig(**config_dict['paths'])
        
        if 'database' in config_dict:
            config_kwargs['database'] = DatabaseConfig(**config_dict['database'])
        
        if 'api' in config_dict:
            config_kwargs['api'] = APIConfig(**config_dict['api'])
        
        if 'ml' in config_dict:
            config_kwargs['ml'] = MLConfig(**config_dict['ml'])
        
        if 'training' in config_dict:
            config_kwargs['training'] = TrainingConfig(**config_dict['training'])
        
        if 'engineering' in config_dict:
            config_kwargs['engineering'] = EngineeringStandards(**config_dict['engineering'])
        
        if 'costs' in config_dict:
            config_kwargs['costs'] = CostParameters(**config_dict['costs'])
        
        if 'environmental' in config_dict:
            config_kwargs['environmental'] = EnvironmentalConfig(**config_dict['environmental'])
        
        if 'logging' in config_dict:
            config_kwargs['logging'] = LoggingConfig(**config_dict['logging'])
        
        if 'performance' in config_dict:
            config_kwargs['performance'] = PerformanceConfig(**config_dict['performance'])
        
        if 'validation' in config_dict:
            config_kwargs['validation'] = ValidationConfig(**config_dict['validation'])
        
        # Add global settings
        global_fields = ['version', 'debug_mode', 'country_default', 'language', 'timezone']
        for field in global_fields:
            if field in config_dict:
                config_kwargs[field] = config_dict[field]
        
        return cls(**config_kwargs)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate engineering standards
        for train_type in ["ICE", "IC", "S"]:
            if train_type not in self.engineering.max_gradients:
                issues.append(f"Missing gradient limit for train type: {train_type}")
            
            if train_type not in self.engineering.min_curve_radii:
                issues.append(f"Missing curve radius for train type: {train_type}")
        
        # Validate cost parameters
        if not self.costs.track_costs:
            issues.append("Track costs not configured")
        
        # Validate API configuration
        if not self.api.osm_overpass_url:
            issues.append("OSM Overpass URL not configured")
        
        # Validate ML configuration
        if self.ml.test_size <= 0 or self.ml.test_size >= 1:
            issues.append(f"Invalid test_size: {self.ml.test_size} (must be between 0 and 1)")
        
        if self.ml.cv_folds < 2:
            issues.append(f"Invalid cv_folds: {self.ml.cv_folds} (must be >= 2)")
        
        # Validate performance settings
        if self.performance.max_workers is not None and self.performance.max_workers <= 0:
            issues.append(f"Invalid max_workers: {self.performance.max_workers} (must be > 0)")
        
        return issues
    
    def get_train_specs(self, train_type: str) -> Dict[str, Any]:
        """Get complete specifications for a train type"""
        return {
            'max_gradient': self.engineering.max_gradients.get(train_type, 3.5),
            'min_curve_radius': self.engineering.min_curve_radii.get(train_type, 800),
            'max_speed': self.engineering.max_speeds.get(train_type, 160),
            'operational_cost': self.costs.operational_costs.get(train_type, 10.0)
        }
    
    def get_regional_cost_factor(self, country_code: str) -> float:
        """Get cost multiplier for a specific country"""
        return self.costs.regional_multipliers.get(country_code.upper(), 1.0)
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"RailwayConfig(version={self.version}, country_default={self.country_default})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"RailwayConfig(version='{self.version}', "
                f"debug_mode={self.debug_mode}, "
                f"paths={self.paths}, "
                f"ml_enabled={self.ml is not None})")

# Configuration factory functions

def create_development_config() -> RailwayConfig:
    """Create configuration optimized for development"""
    config = RailwayConfig()
    config.debug_mode = True
    config.logging.level = LogLevel.DEBUG
    config.ml.hyperparameter_tuning = False
    config.performance.optimization_timeout_minutes = 10
    config.validation.enable_cross_validation = False
    return config

def create_production_config() -> RailwayConfig:
    """Create configuration optimized for production"""
    config = RailwayConfig()
    config.debug_mode = False
    config.logging.level = LogLevel.INFO
    config.ml.hyperparameter_tuning = True
    config.ml.ensemble_methods = True
    config.performance.enable_multiprocessing = True
    config.validation.enable_cross_validation = True
    return config

def create_testing_config() -> RailwayConfig:
    """Create configuration optimized for testing"""
    config = RailwayConfig()
    config.debug_mode = True
    config.logging.level = LogLevel.WARNING
    config.ml.random_forest_n_estimators = 10  # Faster training
    config.performance.optimization_timeout_minutes = 5
    config.performance.max_route_alternatives = 2
    config.validation.benchmark_tolerance_percent = 10.0
    return config

# Load default configuration
def get_default_config() -> RailwayConfig:
    """Get default configuration"""
    return RailwayConfig()

# Example usage
if __name__ == "__main__":
    # Create and save example configuration
    config = create_development_config()
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid!")
    
    # Save to file
    config.save_to_file("example_config.yaml")
    print("Configuration saved to example_config.yaml")
    
    # Load from file
    loaded_config = RailwayConfig.load_from_file("example_config.yaml")
    print(f"Loaded configuration: {loaded_config}")
    
    # Show train specifications
    ice_specs = config.get_train_specs("ICE")
    print(f"ICE specifications: {ice_specs}")
    
    # Show regional cost factor
    swiss_factor = config.get_regional_cost_factor("CH")
    print(f"Swiss cost factor: {swiss_factor}x")