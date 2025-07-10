"""
BCPC Pipeline: Train Selection Module

This module handles train type selection based on route characteristics,
demand patterns, and cost optimization. It includes data from Wikipedia
scraping and standard train specifications.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import requests
from bs4 import BeautifulSoup
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class TrainType(Enum):
    """Enumeration of train types."""
    HIGH_SPEED = "high_speed"
    CONVENTIONAL = "conventional"
    REGIONAL = "regional"
    LIGHT_RAIL = "light_rail"
    METRO = "metro"

@dataclass
class TrainSpecification:
    """Complete train specification data."""
    name: str
    manufacturer: str
    train_type: TrainType
    max_speed_kmh: int
    operating_speed_kmh: int
    capacity_seated: int
    capacity_standing: int
    capacity_total: int
    length_meters: float
    power_kw: int
    energy_consumption_kwh_per_km: float
    purchase_cost_millions: float
    operating_cost_per_km: float
    maintenance_cost_per_year: float
    acceleration_ms2: float
    braking_distance_m: float
    track_gauge_mm: int = 1435  # Standard gauge
    electrification_voltage: Optional[int] = None
    minimum_curve_radius_m: int = 300
    maximum_grade_percent: float = 3.5
    noise_level_db: float = 75.0
    passenger_comfort_score: float = 7.0  # 1-10 scale
    reliability_score: float = 8.0  # 1-10 scale
    environmental_score: float = 7.0  # 1-10 scale

@dataclass
class TrainSelectionResult:
    """Result of train selection analysis."""
    selected_train: TrainSpecification
    alternative_trains: List[TrainSpecification]
    selection_criteria: Dict[str, float]
    performance_metrics: Dict[str, Any]
    cost_analysis: Dict[str, float]
    suitability_score: float

class TrainSelector:
    """Handles train selection and catalog management."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize the train selector with catalog data."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize train catalog
        self.train_catalog = self._load_train_catalog()
        
        # Selection criteria weights
        self.criteria_weights = {
            'speed_suitability': 0.25,
            'capacity_match': 0.20,
            'cost_efficiency': 0.20,
            'terrain_suitability': 0.15,
            'energy_efficiency': 0.10,
            'reliability': 0.10
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            'min_speed_for_hsr': 200,  # km/h
            'min_capacity_utilization': 0.6,
            'max_cost_per_passenger_km': 0.50,  # USD
            'max_energy_consumption': 15.0  # kWh/km
        }
    
    def get_available_trains(self) -> List[TrainSpecification]:
        """Get all available train specifications."""
        try:
            if not self.train_catalog:
                logger.info("Train catalog empty, loading default specifications")
                self.train_catalog = self._load_default_train_specifications()
            
            logger.info(f"Available trains: {len(self.train_catalog)} specifications loaded")
            return self.train_catalog
            
        except Exception as e:
            logger.error(f"Error getting available trains: {e}")
            return self._load_default_train_specifications()
    
    def select_optimal_train(self, route_data: Dict[str, Any], demand_data: Dict[str, Any]) -> TrainSelectionResult:
        """
        Select the optimal train for a given route and demand profile.
        
        Args:
            route_data: Route characteristics (distance, terrain, etc.)
            demand_data: Demand characteristics (passengers, peak hours, etc.)
            
        Returns:
            TrainSelectionResult with selected train and analysis
        """
        try:
            logger.info("Starting train selection analysis")
            
            # Extract route characteristics
            route_distance = route_data.get('distance_km', 0)
            max_grade = route_data.get('max_grade_percent', 0)
            min_curve_radius = route_data.get('min_curve_radius_m', 500)
            terrain_difficulty = route_data.get('terrain_difficulty', 'medium')
            
            # Extract demand characteristics
            peak_demand = demand_data.get('peak_hourly_demand', 0)
            annual_demand = demand_data.get('annual_demand', 0)
            average_occupancy = demand_data.get('average_occupancy_rate', 0.7)
            
            # Analyze each train option
            train_scores = []
            for train in self.train_catalog:
                try:
                    score = self._evaluate_train_suitability(
                        train, route_data, demand_data
                    )
                    train_scores.append((train, score))
                    
                except Exception as e:
                    logger.warning(f"Error evaluating train {train.name}: {e}")
                    continue
            
            if not train_scores:
                raise ValueError("No suitable trains found for route requirements")
            
            # Sort by suitability score
            train_scores.sort(key=lambda x: x[1]['total_score'], reverse=True)
            
            # Select best train
            selected_train, best_score = train_scores[0]
            alternative_trains = [train for train, _ in train_scores[1:6]]  # Top 5 alternatives
            
            # Perform detailed analysis for selected train
            performance_metrics = self._calculate_performance_metrics(
                selected_train, route_data, demand_data
            )
            
            cost_analysis = self._calculate_cost_analysis(
                selected_train, route_data, demand_data
            )
            
            result = TrainSelectionResult(
                selected_train=selected_train,
                alternative_trains=alternative_trains,
                selection_criteria=best_score,
                performance_metrics=performance_metrics,
                cost_analysis=cost_analysis,
                suitability_score=best_score['total_score']
            )
            
            logger.info(f"Selected train: {selected_train.name} (score: {best_score['total_score']:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Train selection failed: {e}")
            raise
    
    def _evaluate_train_suitability(self, train: TrainSpecification, route_data: Dict, demand_data: Dict) -> Dict[str, float]:
        """Evaluate how suitable a train is for the given route and demand."""
        try:
            scores = {}
            
            # Speed suitability
            scores['speed_suitability'] = self._evaluate_speed_suitability(
                train, route_data.get('distance_km', 0)
            )
            
            # Capacity match
            scores['capacity_match'] = self._evaluate_capacity_match(
                train, demand_data.get('peak_hourly_demand', 0)
            )
            
            # Cost efficiency
            scores['cost_efficiency'] = self._evaluate_cost_efficiency(
                train, route_data, demand_data
            )
            
            # Terrain suitability
            scores['terrain_suitability'] = self._evaluate_terrain_suitability(
                train, route_data
            )
            
            # Energy efficiency
            scores['energy_efficiency'] = self._evaluate_energy_efficiency(train)
            
            # Reliability
            scores['reliability'] = train.reliability_score / 10.0
            
            # Calculate weighted total score
            total_score = sum(
                scores[criterion] * weight 
                for criterion, weight in self.criteria_weights.items()
            )
            
            scores['total_score'] = total_score
            return scores
            
        except Exception as e:
            logger.error(f"Error evaluating train suitability: {e}")
            return {'total_score': 0.0}
    
    def _evaluate_speed_suitability(self, train: TrainSpecification, distance_km: float) -> float:
        """Evaluate speed suitability based on route distance."""
        try:
            # Speed requirements based on distance
            if distance_km < 50:
                # Short distance - speed less important
                optimal_speed_range = (80, 160)
            elif distance_km < 200:
                # Medium distance - moderate speed
                optimal_speed_range = (120, 200)
            elif distance_km < 500:
                # Long distance - high speed beneficial
                optimal_speed_range = (160, 300)
            else:
                # Very long distance - high speed essential
                optimal_speed_range = (200, 350)
            
            train_speed = train.operating_speed_kmh
            min_optimal, max_optimal = optimal_speed_range
            
            if min_optimal <= train_speed <= max_optimal:
                return 1.0
            elif train_speed < min_optimal:
                # Penalty for being too slow
                return max(0.3, train_speed / min_optimal)
            else:
                # Minor penalty for being faster than needed (overkill)
                return max(0.8, max_optimal / train_speed)
                
        except Exception as e:
            logger.warning(f"Error evaluating speed suitability: {e}")
            return 0.5
    
    def _evaluate_capacity_match(self, train: TrainSpecification, peak_demand: int) -> float:
        """Evaluate how well train capacity matches demand."""
        try:
            if peak_demand <= 0:
                return 0.5  # Default score if no demand data
            
            # Calculate required capacity (with comfort buffer)
            required_capacity = peak_demand * 1.2  # 20% buffer
            
            # Evaluate capacity match
            capacity_ratio = train.capacity_total / required_capacity
            
            if 0.8 <= capacity_ratio <= 1.5:
                # Good capacity match
                return 1.0
            elif capacity_ratio < 0.8:
                # Insufficient capacity
                return capacity_ratio / 0.8
            else:
                # Overcapacity (wasteful but not critical)
                return max(0.6, 1.5 / capacity_ratio)
                
        except Exception as e:
            logger.warning(f"Error evaluating capacity match: {e}")
            return 0.5
    
    def _evaluate_cost_efficiency(self, train: TrainSpecification, route_data: Dict, demand_data: Dict) -> float:
        """Evaluate cost efficiency of the train."""
        try:
            # Calculate cost per passenger-km
            annual_demand = demand_data.get('annual_demand', 100000)
            route_distance = route_data.get('distance_km', 100)
            
            if annual_demand <= 0 or route_distance <= 0:
                return 0.5
            
            # Annual operating cost
            annual_km = annual_demand * route_distance / train.capacity_total
            operating_cost = annual_km * train.operating_cost_per_km + train.maintenance_cost_per_year
            
            # Cost per passenger-km
            cost_per_passenger_km = operating_cost / (annual_demand * route_distance)
            
            # Evaluate against threshold
            threshold = self.performance_thresholds['max_cost_per_passenger_km']
            if cost_per_passenger_km <= threshold:
                return 1.0
            else:
                # Penalty for high cost
                return max(0.2, threshold / cost_per_passenger_km)
                
        except Exception as e:
            logger.warning(f"Error evaluating cost efficiency: {e}")
            return 0.5
    
    def _evaluate_terrain_suitability(self, train: TrainSpecification, route_data: Dict) -> float:
        """Evaluate train suitability for terrain conditions."""
        try:
            max_grade = route_data.get('max_grade_percent', 0)
            min_curve_radius = route_data.get('min_curve_radius_m', 1000)
            terrain_difficulty = route_data.get('terrain_difficulty', 'medium')
            
            score = 1.0
            
            # Grade suitability
            if max_grade > train.maximum_grade_percent:
                score *= max(0.3, train.maximum_grade_percent / max_grade)
            
            # Curve radius suitability
            if min_curve_radius < train.minimum_curve_radius_m:
                score *= max(0.4, min_curve_radius / train.minimum_curve_radius_m)
            
            # Terrain difficulty modifier
            terrain_modifiers = {
                'flat': 1.0,
                'rolling': 0.9,
                'hilly': 0.8,
                'mountainous': 0.6
            }
            
            score *= terrain_modifiers.get(terrain_difficulty, 0.8)
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"Error evaluating terrain suitability: {e}")
            return 0.7
    
    def _evaluate_energy_efficiency(self, train: TrainSpecification) -> float:
        """Evaluate train energy efficiency."""
        try:
            consumption = train.energy_consumption_kwh_per_km
            threshold = self.performance_thresholds['max_energy_consumption']
            
            if consumption <= threshold:
                # Reward efficient trains
                return 1.0 - (consumption / threshold) * 0.3
            else:
                # Penalty for high consumption
                return max(0.2, threshold / consumption)
                
        except Exception as e:
            logger.warning(f"Error evaluating energy efficiency: {e}")
            return 0.7
    
    def _calculate_performance_metrics(self, train: TrainSpecification, route_data: Dict, demand_data: Dict) -> Dict[str, Any]:
        """Calculate detailed performance metrics for selected train."""
        try:
            route_distance = route_data.get('distance_km', 0)
            annual_demand = demand_data.get('annual_demand', 0)
            
            # Travel time calculation
            avg_speed = train.operating_speed_kmh * 0.85  # Account for stops and acceleration
            travel_time_hours = route_distance / avg_speed if avg_speed > 0 else 0
            
            # Service frequency calculation
            daily_demand = annual_demand / 365 if annual_demand > 0 else 0
            trains_per_day = max(1, daily_demand / train.capacity_total) if train.capacity_total > 0 else 1
            service_frequency_hours = 24 / trains_per_day if trains_per_day > 0 else 24
            
            # Capacity utilization
            capacity_utilization = min(1.0, daily_demand / (trains_per_day * train.capacity_total)) if trains_per_day * train.capacity_total > 0 else 0
            
            # Environmental impact
            daily_energy_consumption = trains_per_day * route_distance * train.energy_consumption_kwh_per_km
            annual_co2_emissions = daily_energy_consumption * 365 * 0.4  # kg CO2 (0.4 kg/kWh grid average)
            
            return {
                'travel_time_hours': round(travel_time_hours, 2),
                'average_speed_kmh': round(avg_speed, 1),
                'trains_per_day': int(trains_per_day),
                'service_frequency_hours': round(service_frequency_hours, 2),
                'capacity_utilization': round(capacity_utilization, 3),
                'daily_energy_consumption_kwh': round(daily_energy_consumption, 1),
                'annual_co2_emissions_kg': round(annual_co2_emissions, 0),
                'passenger_comfort_score': train.passenger_comfort_score,
                'reliability_score': train.reliability_score,
                'noise_level_db': train.noise_level_db
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _calculate_cost_analysis(self, train: TrainSpecification, route_data: Dict, demand_data: Dict) -> Dict[str, float]:
        """Calculate comprehensive cost analysis for selected train."""
        try:
            route_distance = route_data.get('distance_km', 0)
            annual_demand = demand_data.get('annual_demand', 0)
            
            # Fleet size calculation
            daily_demand = annual_demand / 365 if annual_demand > 0 else 0
            trains_needed = max(1, daily_demand / train.capacity_total) if train.capacity_total > 0 else 1
            
            # Capital costs
            fleet_purchase_cost = trains_needed * train.purchase_cost_millions * 1_000_000
            
            # Annual operating costs
            annual_km = trains_needed * route_distance * 365 * 2  # Round trips
            annual_operating_cost = annual_km * train.operating_cost_per_km
            annual_maintenance_cost = trains_needed * train.maintenance_cost_per_year
            annual_energy_cost = annual_km * train.energy_consumption_kwh_per_km * 0.12  # $0.12/kWh
            
            total_annual_cost = annual_operating_cost + annual_maintenance_cost + annual_energy_cost
            
            # Cost per passenger and per km
            cost_per_passenger = total_annual_cost / annual_demand if annual_demand > 0 else 0
            cost_per_passenger_km = cost_per_passenger / route_distance if route_distance > 0 else 0
            
            # Break-even analysis
            revenue_per_passenger = route_distance * 0.10  # $0.10 per km average fare
            annual_revenue = annual_demand * revenue_per_passenger
            net_profit = annual_revenue - total_annual_cost
            payback_period_years = fleet_purchase_cost / max(1, net_profit) if net_profit > 0 else 999
            
            return {
                'fleet_size': int(trains_needed),
                'fleet_purchase_cost': fleet_purchase_cost,
                'annual_operating_cost': annual_operating_cost,
                'annual_maintenance_cost': annual_maintenance_cost,
                'annual_energy_cost': annual_energy_cost,
                'total_annual_cost': total_annual_cost,
                'cost_per_passenger': cost_per_passenger,
                'cost_per_passenger_km': cost_per_passenger_km,
                'annual_revenue': annual_revenue,
                'net_annual_profit': net_profit,
                'payback_period_years': min(payback_period_years, 50)  # Cap at 50 years
            }
            
        except Exception as e:
            logger.error(f"Error calculating cost analysis: {e}")
            return {}
    
    def _load_train_catalog(self) -> List[TrainSpecification]:
        """Load train catalog from cache or create default."""
        cache_file = self.cache_dir / "train_catalog.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    catalog_data = json.load(f)
                return [self._dict_to_train_spec(spec) for spec in catalog_data]
            except Exception as e:
                logger.warning(f"Could not load train catalog from cache: {e}")
        
        # Create default catalog and save to cache
        catalog = self._load_default_train_specifications()
        self._save_train_catalog(catalog)
        return catalog
    
    def _save_train_catalog(self, catalog: List[TrainSpecification]):
        """Save train catalog to cache."""
        cache_file = self.cache_dir / "train_catalog.json"
        
        try:
            catalog_data = [self._train_spec_to_dict(train) for train in catalog]
            with open(cache_file, 'w') as f:
                json.dump(catalog_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save train catalog to cache: {e}")
    
    def _train_spec_to_dict(self, train: TrainSpecification) -> Dict:
        """Convert TrainSpecification to dictionary."""
        return {
            'name': train.name,
            'manufacturer': train.manufacturer,
            'train_type': train.train_type.value,
            'max_speed_kmh': train.max_speed_kmh,
            'operating_speed_kmh': train.operating_speed_kmh,
            'capacity_seated': train.capacity_seated,
            'capacity_standing': train.capacity_standing,
            'capacity_total': train.capacity_total,
            'length_meters': train.length_meters,
            'power_kw': train.power_kw,
            'energy_consumption_kwh_per_km': train.energy_consumption_kwh_per_km,
            'purchase_cost_millions': train.purchase_cost_millions,
            'operating_cost_per_km': train.operating_cost_per_km,
            'maintenance_cost_per_year': train.maintenance_cost_per_year,
            'acceleration_ms2': train.acceleration_ms2,
            'braking_distance_m': train.braking_distance_m,
            'track_gauge_mm': train.track_gauge_mm,
            'electrification_voltage': train.electrification_voltage,
            'minimum_curve_radius_m': train.minimum_curve_radius_m,
            'maximum_grade_percent': train.maximum_grade_percent,
            'noise_level_db': train.noise_level_db,
            'passenger_comfort_score': train.passenger_comfort_score,
            'reliability_score': train.reliability_score,
            'environmental_score': train.environmental_score
        }
    
    def _dict_to_train_spec(self, data: Dict) -> TrainSpecification:
        """Convert dictionary to TrainSpecification."""
        return TrainSpecification(
            name=data['name'],
            manufacturer=data['manufacturer'],
            train_type=TrainType(data['train_type']),
            max_speed_kmh=data['max_speed_kmh'],
            operating_speed_kmh=data['operating_speed_kmh'],
            capacity_seated=data['capacity_seated'],
            capacity_standing=data['capacity_standing'],
            capacity_total=data['capacity_total'],
            length_meters=data['length_meters'],
            power_kw=data['power_kw'],
            energy_consumption_kwh_per_km=data['energy_consumption_kwh_per_km'],
            purchase_cost_millions=data['purchase_cost_millions'],
            operating_cost_per_km=data['operating_cost_per_km'],
            maintenance_cost_per_year=data['maintenance_cost_per_year'],
            acceleration_ms2=data['acceleration_ms2'],
            braking_distance_m=data['braking_distance_m'],
            track_gauge_mm=data.get('track_gauge_mm', 1435),
            electrification_voltage=data.get('electrification_voltage'),
            minimum_curve_radius_m=data.get('minimum_curve_radius_m', 300),
            maximum_grade_percent=data.get('maximum_grade_percent', 3.5),
            noise_level_db=data.get('noise_level_db', 75.0),
            passenger_comfort_score=data.get('passenger_comfort_score', 7.0),
            reliability_score=data.get('reliability_score', 8.0),
            environmental_score=data.get('environmental_score', 7.0)
        )
    
    def _load_default_train_specifications(self) -> List[TrainSpecification]:
        """Load default train specifications when no catalog is available."""
        logger.info("Loading default train specifications")
        
        default_trains = [
            # High-Speed Trains
            TrainSpecification(
                name="TGV 2N2 Euroduplex",
                manufacturer="Alstom",
                train_type=TrainType.HIGH_SPEED,
                max_speed_kmh=320,
                operating_speed_kmh=280,
                capacity_seated=556,
                capacity_standing=100,
                capacity_total=656,
                length_meters=200.0,
                power_kw=8800,
                energy_consumption_kwh_per_km=12.5,
                purchase_cost_millions=35.0,
                operating_cost_per_km=15.0,
                maintenance_cost_per_year=2_500_000,
                acceleration_ms2=0.7,
                braking_distance_m=3500,
                electrification_voltage=25000,
                minimum_curve_radius_m=4000,
                maximum_grade_percent=3.5,
                noise_level_db=72,
                passenger_comfort_score=9.0,
                reliability_score=9.2,
                environmental_score=8.5
            ),
            
            TrainSpecification(
                name="ICE 3",
                manufacturer="Siemens",
                train_type=TrainType.HIGH_SPEED,
                max_speed_kmh=330,
                operating_speed_kmh=300,
                capacity_seated=444,
                capacity_standing=80,
                capacity_total=524,
                length_meters=200.0,
                power_kw=8000,
                energy_consumption_kwh_per_km=11.8,
                purchase_cost_millions=40.0,
                operating_cost_per_km=16.5,
                maintenance_cost_per_year=2_800_000,
                acceleration_ms2=0.8,
                braking_distance_m=3200,
                electrification_voltage=15000,
                minimum_curve_radius_m=3500,
                maximum_grade_percent=4.0,
                noise_level_db=70,
                passenger_comfort_score=9.2,
                reliability_score=9.0,
                environmental_score=8.8
            ),
            
            TrainSpecification(
                name="Shinkansen N700S",
                manufacturer="Kawasaki/Hitachi",
                train_type=TrainType.HIGH_SPEED,
                max_speed_kmh=320,
                operating_speed_kmh=285,
                capacity_seated=1323,
                capacity_standing=200,
                capacity_total=1523,
                length_meters=400.0,
                power_kw=17200,
                energy_consumption_kwh_per_km=18.2,
                purchase_cost_millions=150.0,
                operating_cost_per_km=25.0,
                maintenance_cost_per_year=5_000_000,
                acceleration_ms2=0.71,
                braking_distance_m=4000,
                electrification_voltage=25000,
                minimum_curve_radius_m=4000,
                maximum_grade_percent=3.5,
                noise_level_db=68,
                passenger_comfort_score=9.5,
                reliability_score=9.8,
                environmental_score=9.0
            ),
            
            # Conventional High-Speed
            TrainSpecification(
                name="Pendolino ETR 600",
                manufacturer="Alstom",
                train_type=TrainType.CONVENTIONAL,
                max_speed_kmh=250,
                operating_speed_kmh=200,
                capacity_seated=459,
                capacity_standing=50,
                capacity_total=509,
                length_meters=187.0,
                power_kw=5500,
                energy_consumption_kwh_per_km=9.8,
                purchase_cost_millions=28.0,
                operating_cost_per_km=12.0,
                maintenance_cost_per_year=1_800_000,
                acceleration_ms2=0.9,
                braking_distance_m=2500,
                electrification_voltage=3000,
                minimum_curve_radius_m=1500,
                maximum_grade_percent=6.0,
                noise_level_db=74,
                passenger_comfort_score=8.5,
                reliability_score=8.7,
                environmental_score=8.2
            ),
            
            TrainSpecification(
                name="Velaro E",
                manufacturer="Siemens",
                train_type=TrainType.HIGH_SPEED,
                max_speed_kmh=350,
                operating_speed_kmh=310,
                capacity_seated=404,
                capacity_standing=60,
                capacity_total=464,
                length_meters=200.0,
                power_kw=8800,
                energy_consumption_kwh_per_km=13.2,
                purchase_cost_millions=45.0,
                operating_cost_per_km=18.0,
                maintenance_cost_per_year=3_200_000,
                acceleration_ms2=0.75,
                braking_distance_m=3800,
                electrification_voltage=25000,
                minimum_curve_radius_m=3500,
                maximum_grade_percent=3.5,
                noise_level_db=69,
                passenger_comfort_score=9.0,
                reliability_score=8.8,
                environmental_score=8.7
            ),
            
            # Regional Trains
            TrainSpecification(
                name="Stadler FLIRT",
                manufacturer="Stadler",
                train_type=TrainType.REGIONAL,
                max_speed_kmh=160,
                operating_speed_kmh=120,
                capacity_seated=230,
                capacity_standing=180,
                capacity_total=410,
                length_meters=75.0,
                power_kw=2400,
                energy_consumption_kwh_per_km=6.5,
                purchase_cost_millions=15.0,
                operating_cost_per_km=8.0,
                maintenance_cost_per_year=800_000,
                acceleration_ms2=1.2,
                braking_distance_m=1200,
                electrification_voltage=15000,
                minimum_curve_radius_m=150,
                maximum_grade_percent=7.0,
                noise_level_db=76,
                passenger_comfort_score=7.5,
                reliability_score=8.5,
                environmental_score=8.0
            ),
            
            TrainSpecification(
                name="Siemens Desiro",
                manufacturer="Siemens",
                train_type=TrainType.REGIONAL,
                max_speed_kmh=160,
                operating_speed_kmh=110,
                capacity_seated=174,
                capacity_standing=126,
                capacity_total=300,
                length_meters=55.0,
                power_kw=1500,
                energy_consumption_kwh_per_km=5.8,
                purchase_cost_millions=12.0,
                operating_cost_per_km=7.2,
                maintenance_cost_per_year=650_000,
                acceleration_ms2=1.1,
                braking_distance_m=1100,
                electrification_voltage=15000,
                minimum_curve_radius_m=100,
                maximum_grade_percent=8.0,
                noise_level_db=77,
                passenger_comfort_score=7.2,
                reliability_score=8.3,
                environmental_score=7.8
            ),
            
            # Light Rail
            TrainSpecification(
                name="Alstom Citadis",
                manufacturer="Alstom",
                train_type=TrainType.LIGHT_RAIL,
                max_speed_kmh=80,
                operating_speed_kmh=50,
                capacity_seated=68,
                capacity_standing=182,
                capacity_total=250,
                length_meters=32.0,
                power_kw=480,
                energy_consumption_kwh_per_km=3.2,
                purchase_cost_millions=5.0,
                operating_cost_per_km=4.5,
                maintenance_cost_per_year=250_000,
                acceleration_ms2=1.5,
                braking_distance_m=400,
                electrification_voltage=750,
                minimum_curve_radius_m=25,
                maximum_grade_percent=10.0,
                noise_level_db=78,
                passenger_comfort_score=6.8,
                reliability_score=8.0,
                environmental_score=8.5
            ),
            
            # Diesel Trains for non-electrified routes
            TrainSpecification(
                name="Stadler GTW DMU",
                manufacturer="Stadler",
                train_type=TrainType.REGIONAL,
                max_speed_kmh=140,
                operating_speed_kmh=100,
                capacity_seated=120,
                capacity_standing=80,
                capacity_total=200,
                length_meters=41.7,
                power_kw=560,
                energy_consumption_kwh_per_km=8.5,  # Diesel equivalent
                purchase_cost_millions=8.0,
                operating_cost_per_km=9.5,
                maintenance_cost_per_year=450_000,
                acceleration_ms2=1.0,
                braking_distance_m=800,
                electrification_voltage=None,  # Diesel
                minimum_curve_radius_m=75,
                maximum_grade_percent=6.0,
                noise_level_db=82,
                passenger_comfort_score=6.5,
                reliability_score=7.8,
                environmental_score=6.0
            ),
            
            # Maglev (future technology)
            TrainSpecification(
                name="Shanghai Maglev",
                manufacturer="Siemens/ThyssenKrupp",
                train_type=TrainType.HIGH_SPEED,
                max_speed_kmh=430,
                operating_speed_kmh=350,
                capacity_seated=574,
                capacity_standing=50,
                capacity_total=624,
                length_meters=153.0,
                power_kw=18000,
                energy_consumption_kwh_per_km=22.0,
                purchase_cost_millions=80.0,
                operating_cost_per_km=35.0,
                maintenance_cost_per_year=6_000_000,
                acceleration_ms2=1.2,
                braking_distance_m=5000,
                electrification_voltage=None,  # Magnetic levitation
                minimum_curve_radius_m=8000,
                maximum_grade_percent=4.0,
                noise_level_db=65,
                passenger_comfort_score=9.8,
                reliability_score=7.5,
                environmental_score=7.0
            )
        ]
        
        logger.info(f"Loaded {len(default_trains)} default train specifications")
        return default_trains
    
    def scrape_wikipedia_train_data(self) -> List[Dict[str, Any]]:
        """
        Scrape train data from Wikipedia (as mentioned in the paper).
        This is a simplified implementation for demonstration.
        """
        logger.info("Attempting to scrape Wikipedia train data")
        
        try:
            # URLs for train data on Wikipedia
            urls = [
                "https://en.wikipedia.org/wiki/List_of_high-speed_trains",
                "https://en.wikipedia.org/wiki/List_of_named_passenger_trains"
            ]
            
            scraped_data = []
            
            for url in urls:
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Look for tables with train data
                    tables = soup.find_all('table', class_='wikitable')
                    
                    for table in tables:
                        rows = table.find_all('tr')[1:]  # Skip header
                        
                        for row in rows[:10]:  # Limit to first 10 for demo
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 3:
                                # Extract basic information
                                train_info = {
                                    'name': cells[0].get_text(strip=True),
                                    'country': cells[1].get_text(strip=True) if len(cells) > 1 else '',
                                    'speed': self._extract_speed(cells[2].get_text(strip=True)) if len(cells) > 2 else 0
                                }
                                
                                if train_info['name'] and train_info['speed'] > 0:
                                    scraped_data.append(train_info)
                    
                    time.sleep(1)  # Be respectful to Wikipedia
                    
                except requests.RequestException as e:
                    logger.warning(f"Failed to scrape {url}: {e}")
                    continue
            
            logger.info(f"Scraped {len(scraped_data)} train entries from Wikipedia")
            return scraped_data
            
        except Exception as e:
            logger.error(f"Wikipedia scraping failed: {e}")
            return []
    
    def _extract_speed(self, speed_text: str) -> int:
        """Extract speed value from text."""
        try:
            import re
            # Look for patterns like "320 km/h" or "200mph"
            speed_match = re.search(r'(\d+)\s*(?:km/h|mph)', speed_text.lower())
            if speed_match:
                speed = int(speed_match.group(1))
                # Convert mph to km/h if needed
                if 'mph' in speed_text.lower():
                    speed = int(speed * 1.60934)
                return speed
            return 0
        except:
            return 0
    
    def get_trains_by_type(self, train_type: TrainType) -> List[TrainSpecification]:
        """Get all trains of a specific type."""
        return [train for train in self.train_catalog if train.train_type == train_type]
    
    def get_trains_by_speed_range(self, min_speed: int, max_speed: int) -> List[TrainSpecification]:
        """Get trains within a specific speed range."""
        return [
            train for train in self.train_catalog 
            if min_speed <= train.operating_speed_kmh <= max_speed
        ]
    
    def get_trains_by_capacity_range(self, min_capacity: int, max_capacity: int) -> List[TrainSpecification]:
        """Get trains within a specific capacity range."""
        return [
            train for train in self.train_catalog 
            if min_capacity <= train.capacity_total <= max_capacity
        ]
    
    def compare_trains(self, train_names: List[str]) -> Dict[str, Any]:
        """Compare multiple trains side by side."""
        trains = [train for train in self.train_catalog if train.name in train_names]
        
        if not trains:
            return {'error': 'No trains found with specified names'}
        
        comparison = {
            'trains': [train.name for train in trains],
            'specifications': {}
        }
        
        # Compare key specifications
        specs_to_compare = [
            'max_speed_kmh', 'operating_speed_kmh', 'capacity_total',
            'purchase_cost_millions', 'operating_cost_per_km',
            'energy_consumption_kwh_per_km', 'reliability_score'
        ]
        
        for spec in specs_to_compare:
            comparison['specifications'][spec] = [getattr(train, spec) for train in trains]
        
        return comparison