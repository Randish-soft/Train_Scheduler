# File: Model/grade.py
"""
Plan Grading and Evaluation Module
Comprehensive evaluation and scoring of generated railway route plans.
"""
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .config import RailwayConfig
from .utils.geo import haversine_distance, GeoUtils
from .utils.ml import MLPipeline

class GradeCategory(Enum):
    COST_EFFICIENCY = "cost_efficiency"
    TECHNICAL_FEASIBILITY = "technical_feasibility"
    ENVIRONMENTAL_IMPACT = "environmental_impact"
    RIDERSHIP_POTENTIAL = "ridership_potential"
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    SAFETY_COMPLIANCE = "safety_compliance"
    CONSTRUCTION_VIABILITY = "construction_viability"
    NETWORK_INTEGRATION = "network_integration"

@dataclass
class GradeDetail:
    """Detailed grade information for a specific aspect"""
    score: float  # 0-100
    weight: float  # Importance weight
    max_score: float = 100.0
    issues: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    sub_scores: Dict[str, float] = field(default_factory=dict)

@dataclass
class GradeReport:
    """Comprehensive grading report"""
    plan_name: str
    overall_score: float
    letter_grade: str
    
    # Category scores
    cost_score: float
    feasibility_score: float
    environmental_score: float
    ridership_score: float
    operational_score: float
    safety_score: float
    construction_score: float
    integration_score: float
    
    # Detailed breakdowns
    grade_details: Dict[str, GradeDetail]
    
    # Benchmarking
    percentile_rank: Optional[float] = None
    comparison_baseline: Optional[str] = None
    
    # Metadata
    evaluation_time: datetime = field(default_factory=datetime.now)
    grader_version: str = "1.0.0"
    metrics_used: List[str] = field(default_factory=list)
    
    # Summary
    critical_issues: List[str] = field(default_factory=list)
    key_strengths: List[str] = field(default_factory=list)
    priority_recommendations: List[str] = field(default_factory=list)

class PlanGrader:
    """Main plan evaluation and grading engine"""
    
    def __init__(self, config: RailwayConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Grade weights (can be customized)
        self.grade_weights = {
            GradeCategory.COST_EFFICIENCY: 0.20,
            GradeCategory.TECHNICAL_FEASIBILITY: 0.18,
            GradeCategory.ENVIRONMENTAL_IMPACT: 0.15,
            GradeCategory.RIDERSHIP_POTENTIAL: 0.15,
            GradeCategory.OPERATIONAL_EFFICIENCY: 0.12,
            GradeCategory.SAFETY_COMPLIANCE: 0.10,
            GradeCategory.CONSTRUCTION_VIABILITY: 0.07,
            GradeCategory.NETWORK_INTEGRATION: 0.03
        }
        
        # Benchmark data (would be loaded from historical projects)
        self.benchmarks = self._load_benchmarks()
        
        # ML pipeline for predictive grading
        self.ml_pipeline = MLPipeline()
        self._load_grading_models()
    
    def _load_benchmarks(self) -> Dict[str, Any]:
        """Load benchmark data for comparison"""
        # In production, this would load from a database of historical projects
        return {
            'cost_per_km': {
                'excellent': 3_000_000,    # €3M/km
                'good': 5_000_000,         # €5M/km  
                'average': 8_000_000,      # €8M/km
                'poor': 15_000_000,        # €15M/km
                'unacceptable': 25_000_000 # €25M/km
            },
            'construction_time_months_per_km': {
                'excellent': 3,
                'good': 6,
                'average': 12,
                'poor': 18,
                'unacceptable': 30
            },
            'ridership_per_km': {
                'excellent': 50_000,   # passengers/year/km
                'good': 25_000,
                'average': 15_000,
                'poor': 8_000,
                'unacceptable': 3_000
            },
            'max_gradients': {
                'ICE': 2.5, 'IC': 3.5, 'RE': 4.0, 'S': 4.0, 'FREIGHT': 2.0
            },
            'min_curve_radii': {
                'ICE': 3500, 'IC': 1000, 'RE': 600, 'S': 300, 'FREIGHT': 400
            }
        }
    
    def _load_grading_models(self):
        """Load ML models for predictive grading"""
        try:
            models_dir = self.config.paths.models_dir
            if models_dir.exists():
                self.ml_pipeline.load_models(str(models_dir))
                self.logger.info("📚 Loaded grading models")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load grading models: {e}")
    
    def evaluate_plan(self, 
                     plan_file: Union[str, Dict],
                     metrics: List[str] = None,
                     reference_file: Optional[str] = None) -> GradeReport:
        """Evaluate a route plan and generate comprehensive grade report"""
        
        self.logger.info(f"📊 Starting plan evaluation...")
        
        # Load plan data
        if isinstance(plan_file, str):
            with open(plan_file, 'r', encoding='utf-8') as f:
                plan_data = json.load(f)
            plan_name = Path(plan_file).stem
        else:
            plan_data = plan_file
            plan_name = plan_data.get('name', 'Unknown_Plan')
        
        # Load reference data if provided
        reference_data = None
        if reference_file:
            with open(reference_file, 'r', encoding='utf-8') as f:
                reference_data = json.load(f)
        
        # Set default metrics if not specified
        if metrics is None or 'all' in metrics:
            metrics = [cat.value for cat in GradeCategory]
        
        # Validate plan data
        validation_issues = self._validate_plan_data(plan_data)
        if validation_issues:
            self.logger.warning(f"⚠️ Plan validation issues: {len(validation_issues)}")
        
        # Grade each category
        grade_details = {}
        category_scores = {}
        
        for metric in metrics:
            try:
                category = GradeCategory(metric)
                detail = self._grade_category(category, plan_data, reference_data)
                grade_details[metric] = detail
                category_scores[category] = detail.score
                self.logger.debug(f"✓ Graded {metric}: {detail.score:.1f}/100")
            except ValueError:
                self.logger.warning(f"⚠️ Unknown metric: {metric}")
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(category_scores)
        letter_grade = self._score_to_letter_grade(overall_score)
        
        # Generate summary insights
        critical_issues, key_strengths, recommendations = self._generate_summary_insights(
            grade_details, plan_data
        )
        
        # Create grade report
        report = GradeReport(
            plan_name=plan_name,
            overall_score=overall_score,
            letter_grade=letter_grade,
            cost_score=category_scores.get(GradeCategory.COST_EFFICIENCY, 0),
            feasibility_score=category_scores.get(GradeCategory.TECHNICAL_FEASIBILITY, 0),
            environmental_score=category_scores.get(GradeCategory.ENVIRONMENTAL_IMPACT, 0),
            ridership_score=category_scores.get(GradeCategory.RIDERSHIP_POTENTIAL, 0),
            operational_score=category_scores.get(GradeCategory.OPERATIONAL_EFFICIENCY, 0),
            safety_score=category_scores.get(GradeCategory.SAFETY_COMPLIANCE, 0),
            construction_score=category_scores.get(GradeCategory.CONSTRUCTION_VIABILITY, 0),
            integration_score=category_scores.get(GradeCategory.NETWORK_INTEGRATION, 0),
            grade_details=grade_details,
            metrics_used=metrics,
            critical_issues=critical_issues,
            key_strengths=key_strengths,
            priority_recommendations=recommendations
        )
        
        # Add percentile ranking if we have benchmark data
        report.percentile_rank = self._calculate_percentile_rank(overall_score)
        
        self.logger.info(f"✅ Evaluation completed: {overall_score:.1f}/100 ({letter_grade})")
        return report
    
    def _validate_plan_data(self, plan_data: Dict) -> List[str]:
        """Validate plan data structure and completeness"""
        issues = []
        
        required_fields = ['name', 'stations', 'total_length_km', 'total_cost']
        for field in required_fields:
            if field not in plan_data:
                issues.append(f"Missing required field: {field}")
        
        # Validate stations
        if 'stations' in plan_data:
            stations = plan_data['stations']
            if not isinstance(stations, list) or len(stations) < 2:
                issues.append("Need at least 2 stations")
            
            for i, station in enumerate(stations):
                if not isinstance(station, dict):
                    issues.append(f"Station {i} is not a dictionary")
                    continue
                
                required_station_fields = ['name', 'lat', 'lon']
                for field in required_station_fields:
                    if field not in station:
                        issues.append(f"Station {i} missing field: {field}")
        
        # Validate numeric fields
        numeric_fields = ['total_length_km', 'total_cost']
        for field in numeric_fields:
            if field in plan_data:
                try:
                    float(plan_data[field])
                except (ValueError, TypeError):
                    issues.append(f"Field {field} is not numeric")
        
        return issues
    
    def _grade_category(self, category: GradeCategory, plan_data: Dict, reference_data: Optional[Dict]) -> GradeDetail:
        """Grade a specific category"""
        
        if category == GradeCategory.COST_EFFICIENCY:
            return self._grade_cost_efficiency(plan_data, reference_data)
        elif category == GradeCategory.TECHNICAL_FEASIBILITY:
            return self._grade_technical_feasibility(plan_data, reference_data)
        elif category == GradeCategory.ENVIRONMENTAL_IMPACT:
            return self._grade_environmental_impact(plan_data, reference_data)
        elif category == GradeCategory.RIDERSHIP_POTENTIAL:
            return self._grade_ridership_potential(plan_data, reference_data)
        elif category == GradeCategory.OPERATIONAL_EFFICIENCY:
            return self._grade_operational_efficiency(plan_data, reference_data)
        elif category == GradeCategory.SAFETY_COMPLIANCE:
            return self._grade_safety_compliance(plan_data, reference_data)
        elif category == GradeCategory.CONSTRUCTION_VIABILITY:
            return self._grade_construction_viability(plan_data, reference_data)
        elif category == GradeCategory.NETWORK_INTEGRATION:
            return self._grade_network_integration(plan_data, reference_data)
        else:
            return GradeDetail(score=50.0, weight=0.0, issues=["Unknown category"])
    
    def _grade_cost_efficiency(self, plan_data: Dict, reference_data: Optional[Dict]) -> GradeDetail:
        """Grade cost efficiency of the plan"""
        detail = GradeDetail(score=0, weight=self.grade_weights[GradeCategory.COST_EFFICIENCY])
        
        total_cost = plan_data.get('total_cost', 0)
        total_length = plan_data.get('total_length_km', 1)
        cost_per_km = total_cost / total_length
        
        # Score based on cost per km benchmarks
        benchmarks = self.benchmarks['cost_per_km']
        
        if cost_per_km <= benchmarks['excellent']:
            detail.score = 95
            detail.strengths.append(f"Excellent cost efficiency: €{cost_per_km/1_000_000:.1f}M/km")
        elif cost_per_km <= benchmarks['good']:
            detail.score = 85
            detail.strengths.append(f"Good cost efficiency: €{cost_per_km/1_000_000:.1f}M/km")
        elif cost_per_km <= benchmarks['average']:
            detail.score = 70
        elif cost_per_km <= benchmarks['poor']:
            detail.score = 50
            detail.issues.append(f"High cost per km: €{cost_per_km/1_000_000:.1f}M/km")
        else:
            detail.score = 25
            detail.issues.append(f"Very high cost per km: €{cost_per_km/1_000_000:.1f}M/km")
            detail.recommendations.append("Consider alternative routing to reduce tunneling/bridging")
        
        # Sub-scores
        detail.sub_scores = {
            'construction_cost': min(100, (benchmarks['average'] / cost_per_km) * 70),
            'station_efficiency': self._evaluate_station_cost_efficiency(plan_data),
            'track_efficiency': self._evaluate_track_cost_efficiency(plan_data)
        }
        
        # Compare to reference if available
        if reference_data and 'total_cost' in reference_data:
            ref_cost_per_km = reference_data['total_cost'] / reference_data.get('total_length_km', 1)
            if cost_per_km < ref_cost_per_km * 0.9:
                detail.strengths.append(f"10%+ cheaper than reference solution")
            elif cost_per_km > ref_cost_per_km * 1.1:
                detail.issues.append(f"10%+ more expensive than reference solution")
        
        return detail
    
    def _grade_technical_feasibility(self, plan_data: Dict, reference_data: Optional[Dict]) -> GradeDetail:
        """Grade technical feasibility of the plan"""
        detail = GradeDetail(score=80, weight=self.grade_weights[GradeCategory.TECHNICAL_FEASIBILITY])
        
        # Check gradient compliance
        gradient_score = self._check_gradient_compliance(plan_data)
        
        # Check curve radius compliance
        curve_score = self._check_curve_compliance(plan_data)
        
        # Check engineering complexity
        complexity_score = self._evaluate_engineering_complexity(plan_data)
        
        # Combined score
        detail.score = (gradient_score + curve_score + complexity_score) / 3
        
        detail.sub_scores = {
            'gradient_compliance': gradient_score,
            'curve_compliance': curve_score,
            'engineering_complexity': complexity_score
        }
        
        # Add specific recommendations
        if gradient_score < 60:
            detail.recommendations.append("Review gradient limits for selected train types")
        if curve_score < 60:
            detail.recommendations.append("Increase curve radii or reduce operating speeds")
        if complexity_score < 60:
            detail.recommendations.append("Simplify engineering solutions where possible")
        
        return detail
    
    def _grade_environmental_impact(self, plan_data: Dict, reference_data: Optional[Dict]) -> GradeDetail:
        """Grade environmental impact of the plan"""
        detail = GradeDetail(score=70, weight=self.grade_weights[GradeCategory.ENVIRONMENTAL_IMPACT])
        
        # Base score from plan data
        env_score = plan_data.get('environmental_impact_score', 0.7) * 100
        
        # Evaluate specific environmental factors
        habitat_score = self._evaluate_habitat_impact(plan_data)
        noise_score = self._evaluate_noise_impact(plan_data)
        emissions_score = self._evaluate_emissions_reduction(plan_data)
        
        # Weighted combination
        detail.score = (env_score * 0.4 + habitat_score * 0.3 + noise_score * 0.2 + emissions_score * 0.1)
        
        detail.sub_scores = {
            'overall_impact': env_score,
            'habitat_protection': habitat_score,
            'noise_mitigation': noise_score,
            'emissions_reduction': emissions_score
        }
        
        # Add environmental recommendations
        if detail.score < 60:
            detail.recommendations.extend([
                "Consider additional environmental mitigation measures",
                "Explore elevated tracks to reduce habitat fragmentation",
                "Implement comprehensive noise barriers in urban areas"
            ])
        elif detail.score > 85:
            detail.strengths.append("Strong environmental protection measures")
        
        return detail
    
    def _grade_ridership_potential(self, plan_data: Dict, reference_data: Optional[Dict]) -> GradeDetail:
        """Grade ridership potential of the plan"""
        detail = GradeDetail(score=60, weight=self.grade_weights[GradeCategory.RIDERSHIP_POTENTIAL])
        
        total_length = plan_data.get('total_length_km', 1)
        ridership = plan_data.get('ridership_potential', 0)
        ridership_per_km = ridership / total_length
        
        # Score based on ridership per km benchmarks
        benchmarks = self.benchmarks['ridership_per_km']
        
        if ridership_per_km >= benchmarks['excellent']:
            detail.score = 95
            detail.strengths.append(f"Excellent ridership potential: {ridership_per_km:,.0f} pax/year/km")
        elif ridership_per_km >= benchmarks['good']:
            detail.score = 85
            detail.strengths.append(f"Good ridership potential: {ridership_per_km:,.0f} pax/year/km")
        elif ridership_per_km >= benchmarks['average']:
            detail.score = 70
        elif ridership_per_km >= benchmarks['poor']:
            detail.score = 50
            detail.issues.append(f"Low ridership potential: {ridership_per_km:,.0f} pax/year/km")
        else:
            detail.score = 30
            detail.issues.append(f"Very low ridership potential: {ridership_per_km:,.0f} pax/year/km")
            detail.recommendations.append("Consider serving higher-density population centers")
        
        # Evaluate station catchment areas
        catchment_score = self._evaluate_station_catchments(plan_data)
        
        # Evaluate connectivity to existing networks
        connectivity_score = self._evaluate_network_connectivity(plan_data)
        
        # Adjust score based on additional factors
        detail.score = (detail.score * 0.6 + catchment_score * 0.25 + connectivity_score * 0.15)
        
        detail.sub_scores = {
            'ridership_density': min(100, (ridership_per_km / benchmarks['average']) * 70),
            'station_catchments': catchment_score,
            'network_connectivity': connectivity_score
        }
        
        return detail
    
    def _grade_operational_efficiency(self, plan_data: Dict, reference_data: Optional[Dict]) -> GradeDetail:
        """Grade operational efficiency of the plan"""
        detail = GradeDetail(score=75, weight=self.grade_weights[GradeCategory.OPERATIONAL_EFFICIENCY])
        
        # Evaluate journey times
        journey_efficiency = self._evaluate_journey_times(plan_data)
        
        # Evaluate service frequency
        frequency_score = self._evaluate_service_frequency(plan_data)
        
        # Evaluate capacity utilization
        capacity_score = self._evaluate_capacity_utilization(plan_data)
        
        # Evaluate maintenance efficiency
        maintenance_score = self._evaluate_maintenance_efficiency(plan_data)
        
        # Combined score
        detail.score = (journey_efficiency * 0.3 + frequency_score * 0.25 + 
                       capacity_score * 0.25 + maintenance_score * 0.2)
        
        detail.sub_scores = {
            'journey_efficiency': journey_efficiency,
            'service_frequency': frequency_score,
            'capacity_utilization': capacity_score,
            'maintenance_efficiency': maintenance_score
        }
        
        if detail.score > 85:
            detail.strengths.append("Highly efficient operational design")
        elif detail.score < 60:
            detail.recommendations.append("Optimize service patterns and frequency")
        
        return detail
    
    def _grade_safety_compliance(self, plan_data: Dict, reference_data: Optional[Dict]) -> GradeDetail:
        """Grade safety compliance and standards adherence"""
        detail = GradeDetail(score=85, weight=self.grade_weights[GradeCategory.SAFETY_COMPLIANCE])
        
        # Check signaling requirements
        signaling_score = self._check_signaling_requirements(plan_data)
        
        # Check emergency access
        emergency_score = self._check_emergency_access(plan_data)
        
        # Check platform safety
        platform_score = self._check_platform_safety(plan_data)
        
        # Check grade separation
        separation_score = self._check_grade_separation(plan_data)
        
        # Combined safety score
        detail.score = (signaling_score + emergency_score + platform_score + separation_score) / 4
        
        detail.sub_scores = {
            'signaling_compliance': signaling_score,
            'emergency_access': emergency_score,
            'platform_safety': platform_score,
            'grade_separation': separation_score
        }
        
        if detail.score < 80:
            detail.critical_issues = ["Safety compliance below acceptable threshold"]
            detail.recommendations.append("Conduct comprehensive safety review")
        
        return detail
    
    def _grade_construction_viability(self, plan_data: Dict, reference_data: Optional[Dict]) -> GradeDetail:
        """Grade construction viability and timeline"""
        detail = GradeDetail(score=70, weight=self.grade_weights[GradeCategory.CONSTRUCTION_VIABILITY])
        
        construction_time = plan_data.get('construction_time_months', 120)
        total_length = plan_data.get('total_length_km', 1)
        time_per_km = construction_time / total_length
        
        # Score based on construction time benchmarks
        benchmarks = self.benchmarks['construction_time_months_per_km']
        
        if time_per_km <= benchmarks['excellent']:
            detail.score = 95
            detail.strengths.append(f"Fast construction timeline: {time_per_km:.1f} months/km")
        elif time_per_km <= benchmarks['good']:
            detail.score = 85
        elif time_per_km <= benchmarks['average']:
            detail.score = 70
        elif time_per_km <= benchmarks['poor']:
            detail.score = 50
            detail.issues.append(f"Long construction timeline: {time_per_km:.1f} months/km")
        else:
            detail.score = 30
            detail.issues.append(f"Very long construction timeline: {time_per_km:.1f} months/km")
            detail.recommendations.append("Consider phased construction approach")
        
        # Evaluate construction complexity
        complexity_score = self._evaluate_construction_complexity(plan_data)
        
        # Evaluate resource requirements
        resource_score = self._evaluate_resource_requirements(plan_data)
        
        # Adjust score
        detail.score = (detail.score * 0.5 + complexity_score * 0.3 + resource_score * 0.2)
        
        detail.sub_scores = {
            'timeline_efficiency': min(100, (benchmarks['average'] / time_per_km) * 70),
            'construction_complexity': complexity_score,
            'resource_availability': resource_score
        }
        
        return detail
    
    def _grade_network_integration(self, plan_data: Dict, reference_data: Optional[Dict]) -> GradeDetail:
        """Grade integration with existing transport networks"""
        detail = GradeDetail(score=65, weight=self.grade_weights[GradeCategory.NETWORK_INTEGRATION])
        
        # Evaluate connections to existing rail networks
        rail_integration = self._evaluate_rail_integration(plan_data)
        
        # Evaluate connections to other transport modes
        multimodal_integration = self._evaluate_multimodal_integration(plan_data)
        
        # Evaluate regional connectivity
        regional_connectivity = self._evaluate_regional_connectivity(plan_data)
        
        # Combined score
        detail.score = (rail_integration * 0.5 + multimodal_integration * 0.3 + regional_connectivity * 0.2)
        
        detail.sub_scores = {
            'rail_network_integration': rail_integration,
            'multimodal_connections': multimodal_integration,
            'regional_connectivity': regional_connectivity
        }
        
        if detail.score > 80:
            detail.strengths.append("Well integrated with existing networks")
        elif detail.score < 50:
            detail.recommendations.append("Improve connections to existing transport infrastructure")
        
        return detail
    
    # Helper methods for specific evaluations
    
    def _check_gradient_compliance(self, plan_data: Dict) -> float:
        """Check if gradients comply with train type requirements"""
        score = 85  # Default good score
        
        track_segments = plan_data.get('track_segments', [])
        train_specs = plan_data.get('train_specifications', [])
        
        if not track_segments or not train_specs:
            return 70  # Neutral score if data missing
        
        # Get strictest gradient requirement
        min_gradient_limit = 4.0  # Default
        for train_spec in train_specs:
            train_type = train_spec.get('category', {}).get('value', 'RE')
            limit = self.benchmarks['max_gradients'].get(train_type, 4.0)
            min_gradient_limit = min(min_gradient_limit, limit)
        
        # Check all track segments
        violations = 0
        max_violation = 0
        
        for segment in track_segments:
            gradient = abs(segment.get('gradient_percent', 0))
            if gradient > min_gradient_limit:
                violations += 1
                max_violation = max(max_violation, gradient - min_gradient_limit)
        
        if violations == 0:
            return 95
        elif max_violation < 0.5:
            return 80
        elif max_violation < 1.0:
            return 65
        else:
            return 40
    
    def _check_curve_compliance(self, plan_data: Dict) -> float:
        """Check if curve radii comply with train type requirements"""
        # Simplified - would analyze actual track geometry in practice
        return 80  # Default good score
    
    def _evaluate_engineering_complexity(self, plan_data: Dict) -> float:
        """Evaluate overall engineering complexity"""
        track_segments = plan_data.get('track_segments', [])
        
        if not track_segments:
            return 70
        
        complexity_sum = 0
        for segment in track_segments:
            complexity = segment.get('engineering_complexity', 0.5)
            complexity_sum += complexity
        
        avg_complexity = complexity_sum / len(track_segments)
        
        # Convert to score (lower complexity = higher score)
        return max(20, 100 - (avg_complexity * 80))
    
    def _evaluate_station_cost_efficiency(self, plan_data: Dict) -> float:
        """Evaluate cost efficiency of station designs"""
        stations = plan_data.get('stations', [])
        
        if not stations:
            return 50
        
        total_station_cost = sum(s.get('construction_cost', 0) for s in stations)
        avg_cost_per_station = total_station_cost / len(stations)
        
        # Compare to benchmark (€5M average per station)
        benchmark_cost = 5_000_000
        
        if avg_cost_per_station <= benchmark_cost * 0.8:
            return 90
        elif avg_cost_per_station <= benchmark_cost:
            return 80
        elif avg_cost_per_station <= benchmark_cost * 1.5:
            return 60
        else:
            return 40
    
    def _evaluate_track_cost_efficiency(self, plan_data: Dict) -> float:
        """Evaluate cost efficiency of track design"""
        track_segments = plan_data.get('track_segments', [])
        
        if not track_segments:
            return 50
        
        # Analyze track type distribution
        surface_ratio = 0
        tunnel_ratio = 0
        
        for segment in track_segments:
            track_type = segment.get('track_type', {}).get('value', 'surface')
            if track_type == 'surface':
                surface_ratio += 1
            elif track_type == 'tunnel':
                tunnel_ratio += 1
        
        total_segments = len(track_segments)
        surface_ratio /= total_segments
        tunnel_ratio /= total_segments
        
        # Higher surface ratio = better cost efficiency
        return min(100, 60 + (surface_ratio * 40) - (tunnel_ratio * 20))
    
    def _evaluate_habitat_impact(self, plan_data: Dict) -> float:
        """Evaluate impact on natural habitats"""
        # Simplified evaluation - would use GIS analysis in practice
        env_score = plan_data.get('environmental_impact_score', 0.7)
        return env_score * 100
    
    def _evaluate_noise_impact(self, plan_data: Dict) -> float:
        """Evaluate noise impact on communities"""
        # Check for elevated tracks (less noise) vs surface tracks
        track_segments = plan_data.get('track_segments', [])
        
        if not track_segments:
            return 70
        
        elevated_count = sum(1 for s in track_segments 
                           if s.get('track_type', {}).get('value') == 'elevated')
        
        elevated_ratio = elevated_count / len(track_segments)
        
        # More elevated tracks = less noise impact
        return min(100, 60 + (elevated_ratio * 40))
    
    def _evaluate_emissions_reduction(self, plan_data: Dict) -> float:
        """Evaluate CO2 emissions reduction potential"""
        ridership = plan_data.get('ridership_potential', 0)
        
        # Estimate emissions reduction from modal shift (car to rail)
        # Assume 50% of passengers would otherwise drive
        car_passengers = ridership * 0.5
        total_length = plan_data.get('total_length_km', 1)
        
        # CO2 savings (grams per passenger-km)
        car_emissions = 120  # g CO2/passenger-km
        rail_emissions = 20  # g CO2/passenger-km (average)
        
        annual_co2_savings = car_passengers * total_length * (car_emissions - rail_emissions) / 1000  # kg
        
        # Score based on CO2 savings potential
        if annual_co2_savings > 10_000_000:  # >10,000 tons/year
            return 95
        elif annual_co2_savings > 5_000_000:
            return 85
        elif annual_co2_savings > 1_000_000:
            return 75
        else:
            return 60
    
    def _evaluate_station_catchments(self, plan_data: Dict) -> float:
        """Evaluate population catchment areas of stations"""
        stations = plan_data.get('stations', [])
        
        if not stations:
            return 50
        
        total_catchment = sum(s.get('estimated_daily_passengers', 0) for s in stations)
        avg_catchment = total_catchment / len(stations)
        
        # Score based on average daily passengers per station
        if avg_catchment > 20_000:
            return 95
        elif avg_catchment > 10_000:
            return 85
        elif avg_catchment > 5_000:
            return 70
        elif avg_catchment > 2_000:
            return 55
        else:
            return 40
    
    def _evaluate_network_connectivity(self, plan_data: Dict) -> float:
        """Evaluate connectivity to existing rail networks"""
        stations = plan_data.get('stations', [])
        
        # Count stations with transfer connections
        transfer_stations = sum(1 for s in stations 
                              if s.get('transfer_connections', []))
        
        if not stations:
            return 50
        
        transfer_ratio = transfer_stations / len(stations)
        
        # Higher transfer ratio = better connectivity
        return min(100, 40 + (transfer_ratio * 60))
    
    def _evaluate_journey_times(self, plan_data: Dict) -> float:
        """Evaluate competitiveness of journey times"""
        operational_metrics = plan_data.get('operational_metrics', {})
        
        journey_time_hours = operational_metrics.get('journey_time_hours', 2.0)
        total_length = plan_data.get('total_length_km', 1)
        
        # Calculate average speed
        avg_speed = total_length / journey_time_hours if journey_time_hours > 0 else 50
        
        # Score based on average speed
        if avg_speed > 200:  # High-speed rail
            return 95
        elif avg_speed > 150:  # Fast intercity
            return 85
        elif avg_speed > 100:  # Good intercity
            return 75
        elif avg_speed > 80:   # Average
            return 65
        else:
            return 45
    
    def _evaluate_service_frequency(self, plan_data: Dict) -> float:
        """Evaluate service frequency adequacy"""
        train_specs = plan_data.get('train_specifications', [])
        
        if not train_specs:
            return 60
        
        # Find highest frequency service
        min_frequency = float('inf')
        for train_spec in train_specs:
            freq = train_spec.get('frequency_peak_min', 60)
            min_frequency = min(min_frequency, freq)
        
        # Score based on peak frequency
        if min_frequency <= 10:    # Every 10 min
            return 95
        elif min_frequency <= 20:  # Every 20 min
            return 85
        elif min_frequency <= 30:  # Every 30 min
            return 75
        elif min_frequency <= 60:  # Every hour
            return 65
        else:
            return 45
    
    def _evaluate_capacity_utilization(self, plan_data: Dict) -> float:
        """Evaluate capacity vs demand matching"""
        operational_metrics = plan_data.get('operational_metrics', {})
        
        load_factor = operational_metrics.get('peak_load_factor', 0.7)
        
        # Optimal load factor is 70-80%
        if 0.7 <= load_factor <= 0.8:
            return 95
        elif 0.6 <= load_factor <= 0.85:
            return 85
        elif 0.5 <= load_factor <= 0.9:
            return 75
        elif load_factor < 0.4:
            return 50  # Under-utilized
        else:
            return 45  # Over-crowded
    
    def _evaluate_maintenance_efficiency(self, plan_data: Dict) -> float:
        """Evaluate maintenance facility adequacy"""
        railyards = plan_data.get('railyards', [])
        stations = plan_data.get('stations', [])
        
        if not railyards or not stations:
            return 60
        
        # Calculate railyard to station ratio
        ratio = len(railyards) / len(stations)
        
        # Optimal ratio is around 0.2-0.3 (1 railyard per 3-5 stations)
        if 0.2 <= ratio <= 0.3:
            return 90
        elif 0.15 <= ratio <= 0.35:
            return 80
        elif 0.1 <= ratio <= 0.4:
            return 70
        else:
            return 55
    
    def _check_signaling_requirements(self, plan_data: Dict) -> float:
        """Check signaling system requirements"""
        train_specs = plan_data.get('train_specifications', [])
        
        max_speed = 0
        for train_spec in train_specs:
            speed = train_spec.get('max_speed_kmh', 120)
            max_speed = max(max_speed, speed)
        
        # Score based on speed requirements
        if max_speed > 250:
            return 90  # Requires ETCS Level 2+
        elif max_speed > 160:
            return 85  # Requires ETCS Level 1
        else:
            return 95  # Standard signaling adequate
    
    def _check_emergency_access(self, plan_data: Dict) -> float:
        """Check emergency access provisions"""
        track_segments = plan_data.get('track_segments', [])
        
        if not track_segments:
            return 80
        
        # Count tunnel segments (higher risk, need better emergency access)
        tunnel_segments = sum(1 for s in track_segments 
                            if s.get('track_type', {}).get('value') == 'tunnel')
        
        tunnel_ratio = tunnel_segments / len(track_segments)
        
        # More tunnels = higher emergency access requirements
        return max(70, 90 - (tunnel_ratio * 30))
    
    def _check_platform_safety(self, plan_data: Dict) -> float:
        """Check platform safety standards"""
        stations = plan_data.get('stations', [])
        
        # Assume all stations meet basic safety standards
        # In practice would check platform heights, gap widths, etc.
        return 85
    
    def _check_grade_separation(self, plan_data: Dict) -> float:
        """Check grade separation from roads/pedestrians"""
        track_segments = plan_data.get('track_segments', [])
        
        if not track_segments:
            return 80
        
        # Count surface segments (potential conflict points)
        surface_segments = sum(1 for s in track_segments 
                             if s.get('track_type', {}).get('value') == 'surface')
        
        surface_ratio = surface_segments / len(track_segments)
        
        # Less surface track = better grade separation
        return max(60, 90 - (surface_ratio * 25))
    
    def _evaluate_construction_complexity(self, plan_data: Dict) -> float:
        """Evaluate construction complexity factors"""
        track_segments = plan_data.get('track_segments', [])
        
        if not track_segments:
            return 70
        
        complexity_sum = sum(s.get('engineering_complexity', 0.5) for s in track_segments)
        avg_complexity = complexity_sum / len(track_segments)
        
        # Lower complexity = higher score
        return max(30, 100 - (avg_complexity * 70))
    
    def _evaluate_resource_requirements(self, plan_data: Dict) -> float:
        """Evaluate resource and material requirements"""
        # Simplified - would analyze steel, concrete, labor requirements
        total_cost = plan_data.get('total_cost', 0)
        
        # Assume higher cost = more resources required
        if total_cost < 500_000_000:    # <€500M
            return 90
        elif total_cost < 1_000_000_000:  # <€1B
            return 80
        elif total_cost < 2_000_000_000:  # <€2B
            return 70
        else:
            return 60
    
    def _evaluate_rail_integration(self, plan_data: Dict) -> float:
        """Evaluate integration with existing rail networks"""
        stations = plan_data.get('stations', [])
        
        # Count stations with rail transfer connections
        rail_transfers = sum(1 for s in stations 
                           if any('rail' in conn.lower() or 'train' in conn.lower() 
                                 for conn in s.get('transfer_connections', [])))
        
        if not stations:
            return 50
        
        integration_ratio = rail_transfers / len(stations)
        return min(100, 40 + (integration_ratio * 60))
    
    def _evaluate_multimodal_integration(self, plan_data: Dict) -> float:
        """Evaluate integration with other transport modes"""
        stations = plan_data.get('stations', [])
        
        # Count stations with multimodal connections
        multimodal_count = 0
        for station in stations:
            services = station.get('services', [])
            transfers = station.get('transfer_connections', [])
            
            if ('parking' in services or 'bus' in transfers or 
                'metro' in transfers or 'airport' in transfers):
                multimodal_count += 1
        
        if not stations:
            return 50
        
        multimodal_ratio = multimodal_count / len(stations)
        return min(100, 30 + (multimodal_ratio * 70))
    
    def _evaluate_regional_connectivity(self, plan_data: Dict) -> float:
        """Evaluate regional connectivity improvements"""
        stations = plan_data.get('stations', [])
        total_length = plan_data.get('total_length_km', 1)
        
        # Calculate station spacing
        if len(stations) > 1:
            avg_spacing = total_length / (len(stations) - 1)
            
            # Optimal spacing for regional connectivity is 15-25km
            if 15 <= avg_spacing <= 25:
                return 90
            elif 10 <= avg_spacing <= 30:
                return 80
            elif 5 <= avg_spacing <= 35:
                return 70
            else:
                return 60
        
        return 70
    
    def _calculate_overall_score(self, category_scores: Dict[GradeCategory, float]) -> float:
        """Calculate weighted overall score"""
        total_score = 0
        total_weight = 0
        
        for category, score in category_scores.items():
            weight = self.grade_weights.get(category, 0)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 50
    
    def _score_to_letter_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "A-"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "B-"
        elif score >= 65:
            return "C+"
        elif score >= 60:
            return "C"
        elif score >= 55:
            return "C-"
        elif score >= 50:
            return "D+"
        elif score >= 45:
            return "D"
        elif score >= 40:
            return "D-"
        else:
            return "F"
    
    def _generate_summary_insights(self, grade_details: Dict[str, GradeDetail], 
                                 plan_data: Dict) -> Tuple[List[str], List[str], List[str]]:
        """Generate summary insights from detailed grades"""
        
        critical_issues = []
        key_strengths = []
        recommendations = []
        
        # Collect issues and strengths from all categories
        for category, detail in grade_details.items():
            if detail.score < 60:
                critical_issues.extend(detail.issues)
            elif detail.score > 85:
                key_strengths.extend(detail.strengths)
            
            recommendations.extend(detail.recommendations)
        
        # Add overall insights
        overall_cost = plan_data.get('total_cost', 0)
        if overall_cost > 2_000_000_000:  # >€2B
            critical_issues.append("Very high total project cost")
        
        ridership = plan_data.get('ridership_potential', 0)
        if ridership > 5_000_000:  # >5M annual passengers
            key_strengths.append("High ridership potential")
        
        # Prioritize recommendations
        priority_recommendations = self._prioritize_recommendations(recommendations, grade_details)
        
        return critical_issues[:5], key_strengths[:5], priority_recommendations[:5]
    
    def _prioritize_recommendations(self, recommendations: List[str], 
                                  grade_details: Dict[str, GradeDetail]) -> List[str]:
        """Prioritize recommendations based on impact and feasibility"""
        
        # Simple prioritization - in practice would use more sophisticated scoring
        priority_keywords = {
            'safety': 10,
            'gradient': 8,
            'cost': 7,
            'environmental': 6,
            'frequency': 5
        }
        
        scored_recs = []
        for rec in recommendations:
            score = 0
            for keyword, points in priority_keywords.items():
                if keyword in rec.lower():
                    score += points
            scored_recs.append((score, rec))
        
        # Sort by priority score and remove duplicates
        scored_recs.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        prioritized = []
        
        for score, rec in scored_recs:
            if rec not in seen:
                prioritized.append(rec)
                seen.add(rec)
        
        return prioritized
    
    def _calculate_percentile_rank(self, overall_score: float) -> float:
        """Calculate percentile rank compared to historical projects"""
        # Simplified percentile calculation
        # In practice would use database of historical project scores
        
        if overall_score >= 90:
            return 95.0
        elif overall_score >= 80:
            return 80.0
        elif overall_score >= 70:
            return 60.0
        elif overall_score >= 60:
            return 40.0
        else:
            return 20.0
    
    def save_report(self, report: GradeReport, file_path: str):
        """Save grade report to file"""
        
        # Convert report to serializable format
        report_dict = self._report_to_dict(report)
        
        # Save to JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"📁 Grade report saved to {file_path}")
    
    def _report_to_dict(self, report: GradeReport) -> Dict[str, Any]:
        """Convert grade report to dictionary for serialization"""
        
        def convert_dataclass(obj):
            if hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    if hasattr(value, '__dict__'):
                        result[key] = convert_dataclass(value)
                    elif isinstance(value, list):
                        result[key] = [convert_dataclass(item) if hasattr(item, '__dict__') else item for item in value]
                    elif isinstance(value, dict):
                        result[key] = {k: convert_dataclass(v) if hasattr(v, '__dict__') else v for k, v in value.items()}
                    else:
                        result[key] = value
                return result
            else:
                return obj
        
        return convert_dataclass(report)
    
    def compare_plans(self, plan_files: List[str]) -> Dict[str, Any]:
        """Compare multiple plans and rank them"""
        
        plans_grades = []
        
        for plan_file in plan_files:
            try:
                grade_report = self.evaluate_plan(plan_file)
                plans_grades.append({
                    'file': plan_file,
                    'name': grade_report.plan_name,
                    'grade': grade_report.overall_score,
                    'letter': grade_report.letter_grade,
                    'report': grade_report
                })
            except Exception as e:
                self.logger.error(f"Failed to grade {plan_file}: {e}")
        
        # Sort by grade
        plans_grades.sort(key=lambda x: x['grade'], reverse=True)
        
        return {
            'ranking': plans_grades,
            'best_plan': plans_grades[0] if plans_grades else None,
            'comparison_date': datetime.now().isoformat()
        }

# Example usage and testing
if __name__ == "__main__":
    from .config import create_development_config
    
    config = create_development_config()
    grader = PlanGrader(config)
    
    # Example plan data for testing
    example_plan = {
        'name': 'Test_Route',
        'total_length_km': 150,
        'total_cost': 1_200_000_000,  # €1.2B
        'construction_time_months': 72,
        'ridership_potential': 2_500_000,
        'environmental_impact_score': 0.75,
        'stations': [
            {
                'name': 'City A Central',
                'lat': 50.8503,
                'lon': 4.3517,
                'station_type': 'intercity',
                'platform_count': 6,
                'estimated_daily_passengers': 15000,
                'construction_cost': 12_000_000,
                'services': ['restaurants', 'shops', 'wifi', 'parking'],
                'transfer_connections': ['metro', 'bus']
            },
            {
                'name': 'City B Regional',
                'lat': 51.2194,
                'lon': 4.4025,
                'station_type': 'regional',
                'platform_count': 4,
                'estimated_daily_passengers': 8000,
                'construction_cost': 8_000_000,
                'services': ['cafe', 'parking'],
                'transfer_connections': ['bus']
            }
        ],
        'track_segments': [
            {
                'track_type': {'value': 'surface'},
                'gradient_percent': 1.8,
                'engineering_complexity': 0.4
            },
            {
                'track_type': {'value': 'tunnel'},
                'gradient_percent': 3.2,
                'engineering_complexity': 0.8
            }
        ],
        'train_specifications': [
            {
                'category': {'value': 'IC'},
                'max_speed_kmh': 200,
                'capacity_passengers': 500,
                'frequency_peak_min': 30,
                'frequency_offpeak_min': 60
            }
        ],
        'railyards': [
            {
                'maintenance_type': 'depot',
                'capacity_estimate': 20,
                'land_cost_factor': 1.2
            }
        ],
        'operational_metrics': {
            'journey_time_hours': 1.2,
            'peak_load_factor': 0.75
        }
    }
    
    print("📊 Testing plan grading system...")
    
    try:
        # Grade the example plan
        report = grader.evaluate_plan(example_plan)
        
        print(f"\n✅ Grade Report for {report.plan_name}")
        print(f"Overall Score: {report.overall_score:.1f}/100 ({report.letter_grade})")
        print(f"Percentile Rank: {report.percentile_rank:.0f}%")
        
        print(f"\n📊 Category Scores:")
        print(f"  💰 Cost Efficiency: {report.cost_score:.1f}/100")
        print(f"  🔧 Technical Feasibility: {report.feasibility_score:.1f}/100")
        print(f"  🌱 Environmental Impact: {report.environmental_score:.1f}/100")
        print(f"  🚄 Ridership Potential: {report.ridership_score:.1f}/100")
        print(f"  ⚡ Operational Efficiency: {report.operational_score:.1f}/100")
        print(f"  🛡️  Safety Compliance: {report.safety_score:.1f}/100")
        print(f"  🏗️  Construction Viability: {report.construction_score:.1f}/100")
        print(f"  🔗 Network Integration: {report.integration_score:.1f}/100")
        
        if report.critical_issues:
            print(f"\n⚠️ Critical Issues:")
            for issue in report.critical_issues:
                print(f"  • {issue}")
        
        if report.key_strengths:
            print(f"\n✅ Key Strengths:")
            for strength in report.key_strengths:
                print(f"  • {strength}")
        
        if report.priority_recommendations:
            print(f"\n🎯 Priority Recommendations:")
            for rec in report.priority_recommendations:
                print(f"  • {rec}")
        
        # Save the report
        grader.save_report(report, "example_grade_report.json")
        print(f"\n📁 Report saved to example_grade_report.json")
        
    except Exception as e:
        print(f"❌ Grading failed: {e}")
        import traceback
        traceback.print_exc()