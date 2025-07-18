# File: model/intelligence/train_classifier.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math

class TrainCategory(Enum):
    S_BAHN = "S"           # Suburban rail (S1, S2, etc.)
    REGIONAL = "RE"        # Regional Express
    REGIONAL_BAHN = "RB"   # Regional Bahn (slower regional)
    INTERCITY = "IC"       # Intercity
    INTERCITY_EXPRESS = "ICE"  # High-speed intercity
    EUROCITY = "EC"        # International intercity
    EUROSTAR = "EST"       # High-speed international
    FREIGHT = "FREIGHT"    # Cargo trains
    TRAM = "TRAM"         # Light rail/tram (filtered out)
    METRO = "METRO"       # Underground (filtered out)

@dataclass
class TrainSpecification:
    category: TrainCategory
    max_speed_kmh: int
    typical_speed_kmh: int
    acceleration_ms2: float
    power_kw: int
    capacity_passengers: int
    frequency_peak_min: int    # Minutes between trains in peak
    frequency_offpeak_min: int # Minutes between trains off-peak
    station_dwell_time_sec: int
    max_gradient_percent: float
    min_curve_radius_m: int
    electrification_required: bool
    platform_length_m: int
    operational_cost_per_km: float
    reasons: List[str]

@dataclass
class RouteAnalysis:
    total_distance_km: float
    max_gradient_percent: float
    avg_gradient_percent: float
    min_curve_radius_m: int
    station_count: int
    avg_station_spacing_km: float
    urban_percentage: float
    international_route: bool
    expected_demand_passengers_per_hour: int
    stops_per_100km: int

class TrainClassifier:
    def __init__(self):
        self.train_classifier = RandomForestClassifier(n_estimators=150, random_state=42)
        self.speed_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.capacity_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.trained = False
        self.train_specifications = self._initialize_train_specs()
        self.learned_patterns = {}
        
    def _initialize_train_specs(self) -> Dict[TrainCategory, TrainSpecification]:
        """Initialize standard train specifications based on real European trains"""
        return {
            TrainCategory.S_BAHN: TrainSpecification(
                category=TrainCategory.S_BAHN,
                max_speed_kmh=120,
                typical_speed_kmh=60,
                acceleration_ms2=1.2,
                power_kw=2400,
                capacity_passengers=600,
                frequency_peak_min=5,
                frequency_offpeak_min=10,
                station_dwell_time_sec=30,
                max_gradient_percent=4.0,
                min_curve_radius_m=300,
                electrification_required=True,
                platform_length_m=140,
                operational_cost_per_km=8.50,
                reasons=["high_frequency", "urban_suburban", "short_distances"]
            ),
            
            TrainCategory.REGIONAL_BAHN: TrainSpecification(
                category=TrainCategory.REGIONAL_BAHN,
                max_speed_kmh=120,
                typical_speed_kmh=75,
                acceleration_ms2=0.8,
                power_kw=2000,
                capacity_passengers=300,
                frequency_peak_min=30,
                frequency_offpeak_min=60,
                station_dwell_time_sec=60,
                max_gradient_percent=4.0,
                min_curve_radius_m=400,
                electrification_required=False,  # Can be diesel
                platform_length_m=120,
                operational_cost_per_km=6.20,
                reasons=["local_service", "frequent_stops", "moderate_speed"]
            ),
            
            TrainCategory.REGIONAL: TrainSpecification(
                category=TrainCategory.REGIONAL,
                max_speed_kmh=160,
                typical_speed_kmh=90,
                acceleration_ms2=0.9,
                power_kw=3200,
                capacity_passengers=400,
                frequency_peak_min=60,
                frequency_offpeak_min=120,
                station_dwell_time_sec=90,
                max_gradient_percent=3.5,
                min_curve_radius_m=600,
                electrification_required=True,
                platform_length_m=160,
                operational_cost_per_km=7.80,
                reasons=["regional_connectivity", "moderate_stops", "medium_distance"]
            ),
            
            TrainCategory.INTERCITY: TrainSpecification(
                category=TrainCategory.INTERCITY,
                max_speed_kmh=200,
                typical_speed_kmh=130,
                acceleration_ms2=0.7,
                power_kw=6400,
                capacity_passengers=500,
                frequency_peak_min=120,
                frequency_offpeak_min=180,
                station_dwell_time_sec=120,
                max_gradient_percent=3.0,
                min_curve_radius_m=1000,
                electrification_required=True,
                platform_length_m=200,
                operational_cost_per_km=12.40,
                reasons=["long_distance", "limited_stops", "high_speed"]
            ),
            
            TrainCategory.INTERCITY_EXPRESS: TrainSpecification(
                category=TrainCategory.INTERCITY_EXPRESS,
                max_speed_kmh=320,
                typical_speed_kmh=200,
                acceleration_ms2=0.6,
                power_kw=9280,
                capacity_passengers=460,
                frequency_peak_min=60,
                frequency_offpeak_min=120,
                station_dwell_time_sec=180,
                max_gradient_percent=2.5,
                min_curve_radius_m=3500,
                electrification_required=True,
                platform_length_m=200,
                operational_cost_per_km=18.60,
                reasons=["very_high_speed", "major_cities_only", "dedicated_infrastructure"]
            ),
            
            TrainCategory.EUROCITY: TrainSpecification(
                category=TrainCategory.EUROCITY,
                max_speed_kmh=200,
                typical_speed_kmh=140,
                acceleration_ms2=0.7,
                power_kw=6400,
                capacity_passengers=480,
                frequency_peak_min=180,
                frequency_offpeak_min=360,
                station_dwell_time_sec=300,  # Longer for border controls
                max_gradient_percent=3.0,
                min_curve_radius_m=1200,
                electrification_required=True,
                platform_length_m=200,
                operational_cost_per_km=15.20,
                reasons=["international_service", "border_crossings", "premium_comfort"]
            ),
            
            TrainCategory.EUROSTAR: TrainSpecification(
                category=TrainCategory.EUROSTAR,
                max_speed_kmh=320,
                typical_speed_kmh=250,
                acceleration_ms2=0.5,
                power_kw=9280,
                capacity_passengers=750,
                frequency_peak_min=120,
                frequency_offpeak_min=180,
                station_dwell_time_sec=600,  # Security checks
                max_gradient_percent=2.5,
                min_curve_radius_m=4000,
                electrification_required=True,
                platform_length_m=400,
                operational_cost_per_km=25.80,
                reasons=["international_high_speed", "security_requirements", "channel_tunnel"]
            ),
            
            TrainCategory.FREIGHT: TrainSpecification(
                category=TrainCategory.FREIGHT,
                max_speed_kmh=120,
                typical_speed_kmh=80,
                acceleration_ms2=0.3,
                power_kw=6400,
                capacity_passengers=0,
                frequency_peak_min=360,  # Less frequent
                frequency_offpeak_min=180,  # More frequent at night
                station_dwell_time_sec=1800,  # 30 min loading/unloading
                max_gradient_percent=2.0,    # Strict for heavy loads
                min_curve_radius_m=400,
                electrification_required=False,
                platform_length_m=750,  # Long freight trains
                operational_cost_per_km=4.50,
                reasons=["cargo_transport", "heavy_loads", "low_gradients_required"]
            )
        }
    
    def learn_from_existing_services(self, service_data: List[Dict]) -> None:
        """Learn train type patterns from existing railway services"""
        if len(service_data) < 30:
            print("Warning: Limited service data for robust learning")
            
        # Extract features and labels for training
        features = []
        labels = []
        
        for service in service_data:
            feature_vector = self._extract_service_features(service)
            if len(feature_vector) == 12:  # Ensure complete feature vector
                features.append(feature_vector)
                labels.append(service.get('train_type', 'REGIONAL'))
        
        if len(features) < 10:
            print("Insufficient valid training data")
            return
        
        # Train classification model
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train classifier
        self.train_classifier.fit(X_train_scaled, y_train_encoded)
        
        # Evaluate
        y_pred = self.train_classifier.predict(X_test_scaled)
        
        # Analyze feature importance
        feature_names = [
            'distance_km', 'max_gradient', 'avg_gradient', 'min_curve_radius',
            'station_count', 'avg_station_spacing', 'urban_percentage', 'international',
            'expected_demand', 'stops_per_100km', 'electrified_percentage', 'avg_speed'
        ]
        
        importance = dict(zip(feature_names, self.train_classifier.feature_importances_))
        self.learned_patterns['feature_importance'] = importance
        
        # Learn service patterns
        self._analyze_service_patterns(service_data)
        
        self.trained = True
        print(f"Trained classifier on {len(features)} services")
    
    def _extract_service_features(self, service: Dict) -> List[float]:
        """Extract features from service data for ML training"""
        return [
            service.get('route_distance_km', 100),
            service.get('max_gradient_percent', 2.0),
            service.get('avg_gradient_percent', 1.0),
            service.get('min_curve_radius_m', 800),
            service.get('station_count', 10),
            service.get('avg_station_spacing_km', 10),
            service.get('urban_percentage', 0.3),
            1.0 if service.get('crosses_border', False) else 0.0,
            service.get('daily_passengers', 5000),
            service.get('stops_per_100km', 8),
            service.get('electrified_percentage', 0.8),
            service.get('avg_commercial_speed_kmh', 80)
        ]
    
    def _analyze_service_patterns(self, services: List[Dict]) -> None:
        """Analyze patterns in existing train services"""
        patterns = {
            'speed_vs_distance': [],
            'frequency_vs_demand': [],
            'gradient_tolerance': {},
            'international_characteristics': []
        }
        
        for service in services:
            train_type = service.get('train_type', 'REGIONAL')
            
            # Speed vs distance relationship
            patterns['speed_vs_distance'].append({
                'distance': service.get('route_distance_km', 0),
                'speed': service.get('avg_commercial_speed_kmh', 0),
                'type': train_type
            })
            
            # Gradient tolerance by train type
            max_gradient = service.get('max_gradient_percent', 0)
            if train_type not in patterns['gradient_tolerance']:
                patterns['gradient_tolerance'][train_type] = []
            patterns['gradient_tolerance'][train_type].append(max_gradient)
            
            # International service characteristics
            if service.get('crosses_border', False):
                patterns['international_characteristics'].append({
                    'type': train_type,
                    'border_count': service.get('border_crossings', 1),
                    'dwell_time': service.get('avg_station_dwell_sec', 120)
                })
        
        self.learned_patterns['service_patterns'] = patterns
    
    def classify_optimal_train_type(self, route_analysis: RouteAnalysis) -> List[TrainSpecification]:
        """Classify optimal train type(s) for a given route"""
        
        # Generate multiple options ranked by suitability
        candidates = []
        
        # Rule-based classification (always available)
        rule_based_candidates = self._rule_based_classification(route_analysis)
        candidates.extend(rule_based_candidates)
        
        # ML-based classification (if trained)
        if self.trained:
            ml_candidates = self._ml_based_classification(route_analysis)
            candidates.extend(ml_candidates)
        
        # Remove duplicates and rank
        unique_candidates = self._deduplicate_candidates(candidates)
        ranked_candidates = self._rank_candidates(unique_candidates, route_analysis)
        
        return ranked_candidates[:3]  # Return top 3 options
    
    def _rule_based_classification(self, route: RouteAnalysis) -> List[TrainSpecification]:
        """Rule-based train type classification using engineering standards"""
        candidates = []
        
        # High-speed long distance (ICE/Eurostar)
        if (route.total_distance_km > 300 and 
            route.max_gradient_percent <= 2.5 and 
            route.stops_per_100km <= 3 and
            route.min_curve_radius_m >= 3000):
            
            if route.international_route:
                spec = self.train_specifications[TrainCategory.EUROSTAR].copy() if route.total_distance_km > 500 else self.train_specifications[TrainCategory.EUROCITY]
            else:
                spec = self.train_specifications[TrainCategory.INTERCITY_EXPRESS]
            
            spec.reasons = self._generate_selection_reasons(route, spec.category)
            candidates.append(spec)
        
        # Intercity service
        if (route.total_distance_km > 100 and 
            route.max_gradient_percent <= 3.5 and 
            route.stops_per_100km <= 8 and
            route.min_curve_radius_m >= 800):
            
            if route.international_route:
                spec = self.train_specifications[TrainCategory.EUROCITY]
            else:
                spec = self.train_specifications[TrainCategory.INTERCITY]
            
            spec.reasons = self._generate_selection_reasons(route, spec.category)
            candidates.append(spec)
        
        # Regional Express
        if (route.total_distance_km > 30 and 
            route.total_distance_km < 200 and
            route.max_gradient_percent <= 4.0 and
            route.stops_per_100km <= 15):
            
            spec = self.train_specifications[TrainCategory.REGIONAL]
            spec.reasons = self._generate_selection_reasons(route, spec.category)
            candidates.append(spec)
        
        # Regional Bahn (local service)
        if (route.total_distance_km < 100 and
            route.stops_per_100km > 10):
            
            spec = self.train_specifications[TrainCategory.REGIONAL_BAHN]
            spec.reasons = self._generate_selection_reasons(route, spec.category)
            candidates.append(spec)
        
        # S-Bahn (suburban)
        if (route.total_distance_km < 80 and
            route.urban_percentage > 0.6 and
            route.expected_demand_passengers_per_hour > 1000 and
            route.stops_per_100km > 15):
            
            spec = self.train_specifications[TrainCategory.S_BAHN]
            spec.reasons = self._generate_selection_reasons(route, spec.category)
            candidates.append(spec)
        
        # Freight consideration
        if route.expected_demand_passengers_per_hour == 0:  # No passenger demand specified
            spec = self.train_specifications[TrainCategory.FREIGHT]
            spec.reasons = ["dedicated_freight_corridor", "no_passenger_service"]
            candidates.append(spec)
        
        return candidates
    
    def _ml_based_classification(self, route: RouteAnalysis) -> List[TrainSpecification]:
        """ML-based train type classification"""
        if not self.trained:
            return []
        
        # Extract features
        features = [
            route.total_distance_km,
            route.max_gradient_percent,
            route.avg_gradient_percent,
            route.min_curve_radius_m,
            route.station_count,
            route.avg_station_spacing_km,
            route.urban_percentage,
            1.0 if route.international_route else 0.0,
            route.expected_demand_passengers_per_hour,
            route.stops_per_100km,
            0.8,  # Assume 80% electrified
            80    # Estimated average speed
        ]
        
        try:
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get predictions with probabilities
            probabilities = self.train_classifier.predict_proba(features_scaled)[0]
            classes = self.label_encoder.classes_
            
            # Create candidates from top predictions
            candidates = []
            for i, prob in enumerate(probabilities):
                if prob > 0.1:  # Only consider predictions with >10% confidence
                    train_type_str = classes[i]
                    try:
                        train_category = TrainCategory(train_type_str)
                        if train_category in self.train_specifications:
                            spec = self.train_specifications[train_category]
                            spec.ml_confidence = prob
                            spec.reasons = self._generate_ml_reasons(route, train_category, prob)
                            candidates.append(spec)
                    except ValueError:
                        continue  # Skip invalid train categories
            
            return candidates
            
        except Exception as e:
            print(f"ML classification failed: {e}")
            return []
    
    def _generate_selection_reasons(self, route: RouteAnalysis, category: TrainCategory) -> List[str]:
        """Generate human-readable reasons for train type selection"""
        reasons = []
        
        if category == TrainCategory.INTERCITY_EXPRESS:
            reasons.append(f"High-speed service: {route.total_distance_km:.0f}km distance suitable for ICE")
            if route.stops_per_100km <= 3:
                reasons.append(f"Limited stops ({route.stops_per_100km:.1f}/100km) enables high speeds")
            if route.max_gradient_percent <= 2.5:
                reasons.append(f"Gentle gradients ({route.max_gradient_percent:.1f}%) support 300+ km/h")
        
        elif category == TrainCategory.INTERCITY:
            reasons.append(f"Intercity distance: {route.total_distance_km:.0f}km ideal for IC service")
            if route.stops_per_100km <= 8:
                reasons.append(f"Moderate stops ({route.stops_per_100km:.1f}/100km) balance speed and accessibility")
        
        elif category == TrainCategory.REGIONAL:
            reasons.append(f"Regional service: {route.total_distance_km:.0f}km connects regional centers")
            if route.avg_station_spacing_km < 15:
                reasons.append(f"Station spacing ({route.avg_station_spacing_km:.1f}km) suits regional travel")
        
        elif category == TrainCategory.S_BAHN:
            reasons.append(f"Urban/suburban: {route.urban_percentage*100:.0f}% urban route")
            if route.expected_demand_passengers_per_hour > 1000:
                reasons.append(f"High frequency needed: {route.expected_demand_passengers_per_hour} pax/hour")
            if route.stops_per_100km > 15:
                reasons.append(f"Dense stops ({route.stops_per_100km:.1f}/100km) serve urban areas")
        
        elif category == TrainCategory.EUROCITY or category == TrainCategory.EUROSTAR:
            if route.international_route:
                reasons.append("International route requires cross-border service")
                reasons.append("Enhanced comfort and border facilities needed")
        
        elif category == TrainCategory.FREIGHT:
            if route.max_gradient_percent <= 2.0:
                reasons.append(f"Low gradients ({route.max_gradient_percent:.1f}%) suitable for heavy freight")
            reasons.append("Dedicated freight infrastructure optimized for cargo")
        
        # Add gradient compliance
        spec = self.train_specifications[category]
        if route.max_gradient_percent <= spec.max_gradient_percent:
            reasons.append(f"Gradient compliant: {route.max_gradient_percent:.1f}% ≤ {spec.max_gradient_percent}%")
        
        return reasons
    
    def _generate_ml_reasons(self, route: RouteAnalysis, category: TrainCategory, confidence: float) -> List[str]:
        """Generate reasons for ML-based classification"""
        reasons = [f"ML prediction: {confidence*100:.1f}% confidence for {category.value}"]
        
        # Add the most important features from learned patterns
        if 'feature_importance' in self.learned_patterns:
            top_features = sorted(
                self.learned_patterns['feature_importance'].items(),
                key=lambda x: x[1], reverse=True
            )[:3]
            
            for feature, importance in top_features:
                reasons.append(f"Key factor: {feature} (importance: {importance:.3f})")
        
        return reasons
    
    def _deduplicate_candidates(self, candidates: List[TrainSpecification]) -> List[TrainSpecification]:
        """Remove duplicate train type candidates"""
        seen_categories = set()
        unique_candidates = []
        
        for candidate in candidates:
            if candidate.category not in seen_categories:
                seen_categories.add(candidate.category)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _rank_candidates(self, candidates: List[TrainSpecification], route: RouteAnalysis) -> List[TrainSpecification]:
        """Rank candidates by suitability score"""
        scored_candidates = []
        
        for candidate in candidates:
            score = self._calculate_suitability_score(candidate, route)
            candidate.suitability_score = score
            scored_candidates.append(candidate)
        
        return sorted(scored_candidates, key=lambda x: x.suitability_score, reverse=True)
    
    def _calculate_suitability_score(self, spec: TrainSpecification, route: RouteAnalysis) -> float:
        """Calculate suitability score for train type on route"""
        score = 0.5  # Base score
        
        # Distance suitability
        if spec.category == TrainCategory.INTERCITY_EXPRESS:
            if route.total_distance_km > 200:
                score += 0.3
            elif route.total_distance_km < 100:
                score -= 0.2
        
        elif spec.category == TrainCategory.S_BAHN:
            if route.total_distance_km < 50 and route.urban_percentage > 0.5:
                score += 0.3
            elif route.total_distance_km > 100:
                score -= 0.3
        
        elif spec.category == TrainCategory.REGIONAL:
            if 50 <= route.total_distance_km <= 150:
                score += 0.2
        
        # Gradient compliance
        if route.max_gradient_percent <= spec.max_gradient_percent:
            score += 0.2
        else:
            score -= 0.4  # Heavy penalty for non-compliance
        
        # Curve radius compliance
        if route.min_curve_radius_m >= spec.min_curve_radius_m:
            score += 0.15
        else:
            score -= 0.2
        
        # Demand matching
        expected_capacity = spec.capacity_passengers * (60 / spec.frequency_peak_min)  # Pax per hour
        demand_ratio = route.expected_demand_passengers_per_hour / max(1, expected_capacity)
        
        if 0.6 <= demand_ratio <= 1.2:  # Sweet spot
            score += 0.2
        elif demand_ratio < 0.3:  # Under-utilized
            score -= 0.1
        elif demand_ratio > 2.0:  # Over-capacity
            score -= 0.15
        
        # Station spacing suitability
        optimal_spacing = {
            TrainCategory.S_BAHN: 2.0,
            TrainCategory.REGIONAL_BAHN: 5.0,
            TrainCategory.REGIONAL: 10.0,
            TrainCategory.INTERCITY: 25.0,
            TrainCategory.INTERCITY_EXPRESS: 50.0,
            TrainCategory.EUROCITY: 30.0,
            TrainCategory.EUROSTAR: 100.0
        }
        
        if spec.category in optimal_spacing:
            ideal_spacing = optimal_spacing[spec.category]
            spacing_deviation = abs(route.avg_station_spacing_km - ideal_spacing) / ideal_spacing
            score += max(0, 0.15 * (1 - spacing_deviation))
        
        # International route bonus
        if route.international_route and spec.category in [TrainCategory.EUROCITY, TrainCategory.EUROSTAR]:
            score += 0.1
        
        # ML confidence bonus (if available)
        if hasattr(spec, 'ml_confidence'):
            score += spec.ml_confidence * 0.1
        
        return max(0, min(1, score))
    
    def optimize_service_parameters(self, train_spec: TrainSpecification, route: RouteAnalysis) -> TrainSpecification:
        """Optimize service parameters for specific route"""
        optimized_spec = train_spec.copy()
        
        # Adjust frequency based on demand
        if route.expected_demand_passengers_per_hour > 0:
            required_trains_per_hour = route.expected_demand_passengers_per_hour / train_spec.capacity_passengers
            
            if required_trains_per_hour > 1:
                # Increase frequency
                optimized_spec.frequency_peak_min = max(5, int(60 / required_trains_per_hour))
                optimized_spec.frequency_offpeak_min = min(120, optimized_spec.frequency_peak_min * 2)
            else:
                # Can reduce frequency
                optimized_spec.frequency_peak_min = min(180, int(60 / max(0.5, required_trains_per_hour)))
        
        # Adjust speed based on route characteristics
        if route.max_gradient_percent > train_spec.max_gradient_percent * 0.8:
            # Reduce speed for steep routes
            speed_reduction = 0.9
            optimized_spec.typical_speed_kmh = int(optimized_spec.typical_speed_kmh * speed_reduction)
        
        if route.min_curve_radius_m < train_spec.min_curve_radius_m * 1.5:
            # Reduce speed for tight curves
            speed_reduction = 0.85
            optimized_spec.typical_speed_kmh = int(optimized_spec.typical_speed_kmh * speed_reduction)
        
        # Adjust dwell time based on station characteristics
        if route.stops_per_100km > 15:  # High-frequency stops
            optimized_spec.station_dwell_time_sec = min(optimized_spec.station_dwell_time_sec, 45)
        elif route.stops_per_100km < 5:  # Express service
            optimized_spec.station_dwell_time_sec = max(optimized_spec.station_dwell_time_sec, 120)
        
        # Update reasons
        optimization_reasons = []
        if optimized_spec.frequency_peak_min != train_spec.frequency_peak_min:
            optimization_reasons.append(f"Frequency optimized: {optimized_spec.frequency_peak_min}min peak")
        if optimized_spec.typical_speed_kmh != train_spec.typical_speed_kmh:
            optimization_reasons.append(f"Speed adjusted: {optimized_spec.typical_speed_kmh}km/h for route conditions")
        
        optimized_spec.reasons.extend(optimization_reasons)
        
        return optimized_spec
    
    def calculate_operational_metrics(self, train_spec: TrainSpecification, route: RouteAnalysis) -> Dict:
        """Calculate operational performance metrics"""
        
        # Journey time calculation
        pure_running_time_hours = route.total_distance_km / train_spec.typical_speed_kmh
        dwell_time_hours = (route.station_count * train_spec.station_dwell_time_sec) / 3600
        total_journey_time_hours = pure_running_time_hours + dwell_time_hours
        
        # Capacity and frequency
        trains_per_hour_peak = 60 / train_spec.frequency_peak_min
        hourly_capacity_peak = trains_per_hour_peak * train_spec.capacity_passengers
        
        # Operating costs
        daily_trains = (14 * trains_per_hour_peak) + (10 * (60 / train_spec.frequency_offpeak_min))  # 14h peak + 10h off-peak
        daily_km = daily_trains * route.total_distance_km * 2  # Round trips
        daily_operating_cost = daily_km * train_spec.operational_cost_per_km
        
        # Load factor estimation
        if route.expected_demand_passengers_per_hour > 0:
            peak_load_factor = route.expected_demand_passengers_per_hour / hourly_capacity_peak
        else:
            peak_load_factor = 0.7  # Assume 70% load factor
        
        # Energy consumption (simplified)
        energy_kwh_per_km = train_spec.power_kw * 0.6 / train_spec.typical_speed_kmh  # Rough estimate
        daily_energy_kwh = daily_km * energy_kwh_per_km
        
        # Environmental impact
        co2_kg_per_kwh = 0.35 if train_spec.electrification_required else 2.5  # Electric vs diesel
        daily_co2_kg = daily_energy_kwh * co2_kg_per_kwh
        
        return {
            'journey_time_hours': total_journey_time_hours,
            'pure_running_time_hours': pure_running_time_hours,
            'dwell_time_hours': dwell_time_hours,
            'commercial_speed_kmh': route.total_distance_km / total_journey_time_hours,
            'hourly_capacity_peak': hourly_capacity_peak,
            'daily_trains': daily_trains,
            'daily_operating_cost_eur': daily_operating_cost,
            'annual_operating_cost_eur': daily_operating_cost * 365,
            'peak_load_factor': min(1.0, peak_load_factor),
            'daily_energy_kwh': daily_energy_kwh,
            'daily_co2_kg': daily_co2_kg,
            'revenue_potential_eur_per_year': self._estimate_revenue_potential(
                route, train_spec, hourly_capacity_peak, peak_load_factor
            ),
            'cost_per_passenger_km': daily_operating_cost / max(1, daily_km * peak_load_factor * train_spec.capacity_passengers / 1000)
        }
    
    def _estimate_revenue_potential(self, route: RouteAnalysis, train_spec: TrainSpecification,
                                  hourly_capacity: int, load_factor: float) -> float:
        """Estimate annual revenue potential"""
        
        # Fare estimation based on train type and distance
        fare_per_km = {
            TrainCategory.S_BAHN: 0.12,          # €0.12/km
            TrainCategory.REGIONAL_BAHN: 0.10,   # €0.10/km  
            TrainCategory.REGIONAL: 0.15,        # €0.15/km
            TrainCategory.INTERCITY: 0.25,       # €0.25/km
            TrainCategory.INTERCITY_EXPRESS: 0.35, # €0.35/km
            TrainCategory.EUROCITY: 0.30,        # €0.30/km
            TrainCategory.EUROSTAR: 0.50,        # €0.50/km
            TrainCategory.FREIGHT: 0.08          # €0.08/km (per ton)
        }
        
        base_fare_per_km = fare_per_km.get(train_spec.category, 0.15)
        avg_fare = base_fare_per_km * route.total_distance_km
        
        # Apply distance discounts for longer journeys
        if route.total_distance_km > 200:
            avg_fare *= 0.8  # 20% discount for long distance
        elif route.total_distance_km > 100:
            avg_fare *= 0.9  # 10% discount for medium distance
        
        # Annual passenger volume
        operating_hours_per_day = 18  # 6 AM to 12 AM
        operating_days_per_year = 365
        
        annual_passengers = (
            hourly_capacity * load_factor * 
            operating_hours_per_day * operating_days_per_year
        )
        
        return annual_passengers * avg_fare
    
    def validate_infrastructure_requirements(self, train_spec: TrainSpecification, 
                                           route: RouteAnalysis) -> Dict:
        """Validate infrastructure requirements for selected train type"""
        
        requirements = {
            'compliant': True,
            'issues': [],
            'recommendations': [],
            'infrastructure_needs': []
        }
        
        # Gradient compliance
        if route.max_gradient_percent > train_spec.max_gradient_percent:
            requirements['compliant'] = False
            requirements['issues'].append(
                f"Max gradient {route.max_gradient_percent:.1f}% exceeds {train_spec.category.value} limit of {train_spec.max_gradient_percent}%"
            )
            requirements['recommendations'].append("Reduce gradients through additional tunneling or route modification")
        
        # Curve radius compliance
        if route.min_curve_radius_m < train_spec.min_curve_radius_m:
            requirements['compliant'] = False
            requirements['issues'].append(
                f"Min curve radius {route.min_curve_radius_m}m below {train_spec.category.value} requirement of {train_spec.min_curve_radius_m}m"
            )
            requirements['recommendations'].append("Increase curve radii or reduce operating speeds")
        
        # Platform length requirements
        if train_spec.platform_length_m > 200:  # Assume existing platforms are 200m
            requirements['infrastructure_needs'].append(
                f"Platform extensions required: {train_spec.platform_length_m}m platforms needed"
            )
        
        # Electrification requirements
        if train_spec.electrification_required:
            requirements['infrastructure_needs'].append("Full route electrification required")
            
            # Estimate electrification cost
            electrification_cost_per_km = 2_000_000  # €2M per km
            total_electrification_cost = route.total_distance_km * electrification_cost_per_km
            requirements['electrification_cost_eur'] = total_electrification_cost
        
        # Signaling requirements
        if train_spec.max_speed_kmh > 160:
            requirements['infrastructure_needs'].append("High-speed signaling system (ETCS Level 2)")
        elif train_spec.max_speed_kmh > 120:
            requirements['infrastructure_needs'].append("Modern signaling system (ETCS Level 1)")
        
        # Maintenance facilities
        if train_spec.category in [TrainCategory.INTERCITY_EXPRESS, TrainCategory.EUROSTAR]:
            requirements['infrastructure_needs'].append("Specialized high-speed maintenance facility")
        
        # Freight-specific requirements
        if train_spec.category == TrainCategory.FREIGHT:
            requirements['infrastructure_needs'].extend([
                "Freight loading/unloading terminals",
                "Extended platform lengths (750m)",
                "Reinforced track for heavy axle loads"
            ])
        
        return requirements
    
    def generate_service_plan(self, train_spec: TrainSpecification, 
                            route: RouteAnalysis) -> Dict:
        """Generate comprehensive service plan"""
        
        # Calculate timetable parameters
        journey_time_min = int((route.total_distance_km / train_spec.typical_speed_kmh) * 60)
        dwell_time_total_min = int((route.station_count * train_spec.station_dwell_time_sec) / 60)
        total_journey_time_min = journey_time_min + dwell_time_total_min
        
        # Calculate fleet requirements
        cycle_time_min = total_journey_time_min * 2 + 30  # Round trip + turnaround
        trains_needed_peak = math.ceil(cycle_time_min / train_spec.frequency_peak_min)
        trains_needed_offpeak = math.ceil(cycle_time_min / train_spec.frequency_offpeak_min)
        
        # Calculate staff requirements (simplified)
        staff_per_train = {
            TrainCategory.S_BAHN: 1,        # Driver only
            TrainCategory.REGIONAL_BAHN: 2, # Driver + conductor
            TrainCategory.REGIONAL: 2,      # Driver + conductor
            TrainCategory.INTERCITY: 4,     # Driver + 3 service staff
            TrainCategory.INTERCITY_EXPRESS: 6, # Driver + 5 service staff
            TrainCategory.EUROCITY: 5,      # Driver + 4 service staff
            TrainCategory.EUROSTAR: 8,      # Driver + 7 service staff
            TrainCategory.FREIGHT: 2        # Driver + conductor
        }.get(train_spec.category, 2)
        
        # Operating shifts per day
        operating_hours = 18
        shift_length = 8
        shifts_per_day = math.ceil(operating_hours / shift_length)
        
        total_staff_needed = trains_needed_peak * staff_per_train * shifts_per_day * 1.3  # 30% overhead
        
        return {
            'timetable': {
                'journey_time_min': total_journey_time_min,
                'running_time_min': journey_time_min,
                'station_time_min': dwell_time_total_min,
                'frequency_peak_min': train_spec.frequency_peak_min,
                'frequency_offpeak_min': train_spec.frequency_offpeak_min,
                'first_departure': "05:30",
                'last_departure': "23:30"
            },
            'fleet_requirements': {
                'trains_needed_peak': trains_needed_peak,
                'trains_needed_offpeak': trains_needed_offpeak,
                'fleet_size_total': trains_needed_peak + 2,  # +2 for maintenance spare
                'cycle_time_min': cycle_time_min,
                'utilization_hours_per_day': operating_hours * 0.85  # 85% utilization
            },
            'staffing': {
                'staff_per_train': staff_per_train,
                'total_operational_staff': int(total_staff_needed),
                'drivers_needed': int(trains_needed_peak * shifts_per_day * 1.3),
                'service_staff_needed': int((staff_per_train - 1) * trains_needed_peak * shifts_per_day * 1.3)
            },
            'capacity_analysis': {
                'seats_per_hour_peak': int(60 / train_spec.frequency_peak_min * train_spec.capacity_passengers),
                'daily_seat_capacity': int(self._calculate_daily_capacity(train_spec, operating_hours)),
                'annual_seat_capacity': int(self._calculate_daily_capacity(train_spec, operating_hours) * 365)
            }
        }
    
    def _calculate_daily_capacity(self, train_spec: TrainSpecification, operating_hours: int) -> int:
        """Calculate daily seat capacity"""
        peak_hours = 6  # 6 hours peak per day
        offpeak_hours = operating_hours - peak_hours
        
        peak_trains = peak_hours * (60 / train_spec.frequency_peak_min)
        offpeak_trains = offpeak_hours * (60 / train_spec.frequency_offpeak_min)
        
        return int((peak_trains + offpeak_trains) * train_spec.capacity_passengers)
    
    def compare_train_options(self, options: List[TrainSpecification], 
                            route: RouteAnalysis) -> Dict:
        """Compare multiple train type options"""
        
        comparison = {
            'summary': [],
            'cost_comparison': {},
            'performance_comparison': {},
            'recommendation': None
        }
        
        for option in options:
            metrics = self.calculate_operational_metrics(option, route)
            infrastructure = self.validate_infrastructure_requirements(option, route)
            
            option_summary = {
                'train_type': option.category.value,
                'suitability_score': getattr(option, 'suitability_score', 0),
                'annual_cost_eur': metrics['annual_operating_cost_eur'],
                'revenue_potential_eur': metrics['revenue_potential_eur_per_year'],
                'profit_potential_eur': metrics['revenue_potential_eur_per_year'] - metrics['annual_operating_cost_eur'],
                'journey_time_hours': metrics['journey_time_hours'],
                'infrastructure_compliant': infrastructure['compliant'],
                'infrastructure_investment_eur': infrastructure.get('electrification_cost_eur', 0),
                'reasons': option.reasons[:3]  # Top 3 reasons
            }
            
            comparison['summary'].append(option_summary)
        
        # Find best option
        if comparison['summary']:
            # Rank by combination of suitability score and profit potential
            best_option = max(comparison['summary'], 
                            key=lambda x: x['suitability_score'] * 0.6 + 
                                        (x['profit_potential_eur'] / 10_000_000) * 0.4)
            comparison['recommendation'] = best_option
        
        return comparison
    
    def get_train_specifications(self) -> Dict[str, TrainSpecification]:
        """Get all available train specifications"""
        return {cat.value: spec for cat, spec in self.train_specifications.items()}
    
    def get_learned_patterns(self) -> Dict:
        """Get learned classification patterns"""
        return self.learned_patterns
    
    # Utility methods
    
    def copy(self):
        """Create a copy of TrainSpecification"""
        # This method should be added to TrainSpecification dataclass
        import copy
        return copy.deepcopy(self)

# Add copy method to TrainSpecification
def _add_copy_method():
    import copy
    TrainSpecification.copy = lambda self: copy.deepcopy(self)

_add_copy_method()

# Example usage and testing functions

def create_sample_route_analysis() -> RouteAnalysis:
    """Create sample route for testing"""
    return RouteAnalysis(
        total_distance_km=150,
        max_gradient_percent=2.8,
        avg_gradient_percent=1.2,
        min_curve_radius_m=1200,
        station_count=8,
        avg_station_spacing_km=18.75,
        urban_percentage=0.4,
        international_route=False,
        expected_demand_passengers_per_hour=800,
        stops_per_100km=5.3
    )

def example_classification():
    """Example of train classification workflow"""
    classifier = TrainClassifier()
    
    # Sample route: Brussels to Amsterdam (medium distance, moderate demand)
    route = RouteAnalysis(
        total_distance_km=170,
        max_gradient_percent=1.8,
        avg_gradient_percent=0.9,
        min_curve_radius_m=1500,
        station_count=6,
        avg_station_spacing_km=28.3,
        urban_percentage=0.3,
        international_route=True,
        expected_demand_passengers_per_hour=600,
        stops_per_100km=3.5
    )
    
    # Classify optimal train types
    options = classifier.classify_optimal_train_type(route)
    
    print(f"Found {len(options)} suitable train types:")
    for i, option in enumerate(options):
        print(f"\n{i+1}. {option.category.value}")
        print(f"   Max Speed: {option.max_speed_kmh} km/h")
        print(f"   Capacity: {option.capacity_passengers} passengers")
        print(f"   Suitability: {getattr(option, 'suitability_score', 0):.2f}")
        print(f"   Reasons: {', '.join(option.reasons[:2])}")
    
    if options:
        # Generate detailed service plan for best option
        best_option = options[0]
        service_plan = classifier.generate_service_plan(best_option, route)
        metrics = classifier.calculate_operational_metrics(best_option, route)
        
        print(f"\n=== Service Plan for {best_option.category.value} ===")
        print(f"Journey Time: {service_plan['timetable']['journey_time_min']} minutes")
        print(f"Fleet Required: {service_plan['fleet_requirements']['fleet_size_total']} trains")
        print(f"Annual Operating Cost: €{metrics['annual_operating_cost_eur']:,.0f}")
        print(f"Revenue Potential: €{metrics['revenue_potential_eur_per_year']:,.0f}")
        print(f"Projected Profit: €{metrics['revenue_potential_eur_per_year'] - metrics['annual_operating_cost_eur']:,.0f}")

if __name__ == "__main__":
    example_classification()