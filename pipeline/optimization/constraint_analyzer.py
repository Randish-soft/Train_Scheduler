"""
Route Constraint Analyzer
=========================

Analyzes routing constraints including:
- NIMBY factors (residential impacts, community concerns)
- Environmental constraints (protected areas, wetlands)
- Infrastructure conflicts (roads, utilities, airports)
- Economic factors (land costs, construction complexity)
- Regulatory requirements (permits, approvals)

This module works with route data from plotting_route.py

Author: Miguel Ibrahim E
"""

import logging
import math
from typing import Dict, List, Any, Tuple
from enum import Enum
from pathlib import Path
import json


class ConstraintSeverity(Enum):
    """Severity levels for routing constraints."""
    ABSOLUTE = "absolute"  # Cannot cross
    HIGH = "high"          # Very difficult/expensive
    MEDIUM = "medium"      # Moderately difficult
    LOW = "low"           # Minor constraints


class ConstraintAnalyzer:
    """Analyzes constraints for railway routes."""
    
    def __init__(self, terrain_data: Dict[str, Any], output_dir: Path):
        self.terrain_data = terrain_data
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
    
    def analyze_route_constraints(self, route_options: List[Dict]) -> Dict[str, Any]:
        """
        Analyze constraints for all route options.
        
        Args:
            route_options: List of route dictionaries from plotting_route
            
        Returns:
            Dictionary containing constraint analysis for each route
        """
        self.logger.info("🚧 Analyzing route constraints...")
        
        constraint_analysis = {}
        
        for route in route_options:
            route_id = route['route_id']
            
            # Analyze each constraint type
            constraints = {
                'nimby_factors': self._analyze_nimby_constraints(route),
                'environmental_constraints': self._analyze_environmental_constraints(route),
                'terrain_constraints': self._analyze_terrain_constraints(route),
                'infrastructure_conflicts': self._analyze_infrastructure_conflicts(route),
                'economic_constraints': self._analyze_economic_constraints(route),
                'regulatory_constraints': self._analyze_regulatory_constraints(route)
            }
            
            # Calculate overall constraint score
            overall_score = self._calculate_overall_constraint_score(constraints)
            
            constraint_analysis[route_id] = {
                'route_name': route['route_name'],
                'constraints': constraints,
                'overall_constraint_score': overall_score,
                'constraint_severity': self._classify_constraint_severity(overall_score),
                'mitigation_strategies': self._suggest_mitigation_strategies(constraints),
                'implementation_complexity': self._assess_implementation_complexity(constraints)
            }
        
        # Save constraint analysis
        self._save_constraint_analysis(constraint_analysis)
        
        self.logger.info("✅ Constraint analysis completed")
        return constraint_analysis
    
    def _analyze_nimby_constraints(self, route: Dict) -> Dict[str, Any]:
        """Analyze NIMBY (Not In My Backyard) factors."""
        nimby_factors = {
            'residential_impacts': [],
            'noise_concerns': [],
            'property_value_impacts': [],
            'community_severance': [],
            'visual_impacts': []
        }
        
        origin = route['origin']
        destination = route['destination']
        
        # Analyze proximity to cities (proxy for residential areas)
        for city in [origin, destination]:
            # Check if route passes close to city center
            city_distance = 0  # Would calculate actual distance to route
            
            if city['population'] > 500000:  # Major city
                impact_severity = "high"
                affected_population = min(city['population'] * 0.3, 200000)  # Estimate affected
            elif city['population'] > 100000:  # Medium city
                impact_severity = "medium"
                affected_population = min(city['population'] * 0.2, 50000)
            else:  # Smaller city
                impact_severity = "low"
                affected_population = min(city['population'] * 0.1, 10000)
            
            nimby_factors['residential_impacts'].append({
                'location': f"Near {city['city_name']}",
                'severity': impact_severity,
                'affected_population': int(affected_population),
                'distance_to_city_center_km': 5  # Placeholder
            })
            
            # Add noise concerns for populated areas
            if city['population'] > 100000:
                nimby_factors['noise_concerns'].append({
                    'location': f"Residential areas near {city['city_name']}",
                    'severity': impact_severity,
                    'mitigation_needed': True,
                    'estimated_complaints': max(10, int(affected_population / 1000))
                })
        
        # Property value impacts
        high_impact_zones = len([impact for impact in nimby_factors['residential_impacts'] 
                               if impact['severity'] == 'high'])
        
        if high_impact_zones > 0:
            nimby_factors['property_value_impacts'] = [{
                'zones_affected': high_impact_zones,
                'estimated_impact_percent': "5-15% decrease within 500m",
                'compensation_required': True,
                'estimated_compensation_cost': high_impact_zones * 10000000  # $10M per zone
            }]
        
        # Community severance analysis
        if len(nimby_factors['residential_impacts']) > 1:
            nimby_factors['community_severance'].append({
                'concern': "Railway may divide communities",
                'affected_connections': "School-residential, commercial-residential",
                'mitigation': "Grade separation or community bridges required",
                'estimated_bridges_needed': max(1, len(nimby_factors['residential_impacts']) - 1)
            })
        
        return nimby_factors
    
    def _analyze_environmental_constraints(self, route: Dict) -> Dict[str, Any]:
        """Analyze environmental constraints and protected areas."""
        env_constraints = {
            'protected_areas': [],
            'water_body_crossings': [],
            'forest_impact': [],
            'wildlife_corridors': [],
            'wetland_impacts': [],
            'carbon_footprint': {}
        }
        
        distance = route['total_distance_km']
        route_type = route['route_type']
        
        # Estimate environmental impacts based on route characteristics
        
        # Forest impact estimation (simplified)
        if distance > 100:  # Long routes more likely to cross forests
            forest_risk_factor = min(distance / 500, 1.0)  # Normalize
            estimated_forest_area = distance * 0.05 * forest_risk_factor * 100  # hectares
            
            if estimated_forest_area > 50:  # Significant forest impact
                env_constraints['forest_impact'].append({
                    'estimated_area_affected_hectares': int(estimated_forest_area),
                    'forest_type': 'Mixed tropical/temperate',
                    'biodiversity_risk': 'Medium to High',
                    'mitigation_required': "Reforestation at 2:1 ratio, wildlife corridors",
                    'environmental_impact_assessment': "Required",
                    'estimated_mitigation_cost': estimated_forest_area * 5000  # $5k per hectare
                })
        
        # Water body crossing analysis
        major_crossings = max(1, int(distance / 80))  # Assume 1 major crossing per 80km
        if major_crossings > 0:
            env_constraints['water_body_crossings'] = [{
                'estimated_major_crossings': major_crossings,
                'crossing_types': ['Rivers', 'Streams', 'Possible wetlands'],
                'bridge_requirements': f"{major_crossings} major bridges needed",
                'environmental_permits': "Waterway crossing permits required",
                'estimated_bridge_cost': major_crossings * 15000000,  # $15M per major bridge
                'construction_impact': "Temporary habitat disruption during construction"
            }]
        
        # Wildlife corridor considerations
        if distance > 150:  # Long routes more likely to cross wildlife areas
            env_constraints['wildlife_corridors'] = [{
                'potential_wildlife_crossings': max(2, int(distance / 100)),
                'species_at_risk': 'Local fauna migration patterns',
                'mitigation_measures': 'Wildlife overpasses, underpasses, or bridges',
                'monitoring_required': 'Pre and post-construction wildlife studies'
            }]
        
        # Carbon footprint calculation
        construction_emissions = distance * 800  # Tons CO2 per km (estimated)
        operational_savings = distance * 200 * 50  # Tons CO2 saved per year (vs cars)
        
        env_constraints['carbon_footprint'] = {
            'construction_co2_tons': construction_emissions,
            'annual_operational_savings_tons': operational_savings,
            'break_even_years': construction_emissions / max(operational_savings, 1),
            'net_benefit_20_years': (operational_savings * 20) - construction_emissions,
            'environmental_benefit': "Significant long-term CO2 reduction vs road transport"
        }
        
        return env_constraints
    
    def _analyze_terrain_constraints(self, route: Dict) -> Dict[str, Any]:
        """Analyze terrain-related routing constraints."""
        terrain_constraints = {
            'elevation_challenges': [],
            'grade_analysis': {},
            'geological_risks': [],
            'construction_access': [],
            'drainage_requirements': []
        }
        
        origin = route['origin']
        destination = route['destination']
        distance = route['total_distance_km']
        
        # Use basic metrics from route if available
        basic_metrics = route.get('basic_metrics', {})
        elevation_change = basic_metrics.get('elevation_change', 0)
        avg_grade_estimate = basic_metrics.get('average_grade_estimate', 0)
        
        # Elevation analysis
        if elevation_change > 300:  # Significant elevation change
            terrain_constraints['elevation_challenges'].append({
                'elevation_change_m': elevation_change,
                'challenge_level': "high" if elevation_change > 800 else "medium",
                'engineering_solutions': self._suggest_elevation_solutions(elevation_change),
                'estimated_additional_cost_percent': min(50, elevation_change / 20)  # Up to 50% extra
            })
        
        # Grade analysis using route's basic metrics
        max_grade_estimate = avg_grade_estimate * 1.5  # Estimate max grade
        
        terrain_constraints['grade_analysis'] = {
            'average_grade_percent': avg_grade_estimate,
            'estimated_max_grade_percent': max_grade_estimate,
            'railway_suitability': self._assess_grade_suitability(max_grade_estimate),
            'grade_mitigation': self._suggest_grade_mitigation(max_grade_estimate),
            'switchback_sections_needed': max(0, int((max_grade_estimate - 4) / 2)) if max_grade_estimate > 4 else 0
        }
        
        # Geological risks based on terrain data
        terrain_difficulty = self.terrain_data.get('terrain_difficulty', 'Medium')
        if terrain_difficulty in ['High', 'Very High']:
            terrain_constraints['geological_risks'] = [
                {
                    'risk_type': 'unstable_slopes',
                    'likelihood': 'medium',
                    'mitigation': 'detailed_geological_survey',
                    'estimated_survey_cost': distance * 50000  # $50k per km
                },
                {
                    'risk_type': 'rock_excavation',
                    'likelihood': 'high',
                    'mitigation': 'blasting_and_specialized_equipment',
                    'estimated_additional_cost': distance * 2000000  # $2M per km extra
                }
            ]
        
        # Construction access analysis
        if distance > 100:  # Long routes need multiple access points
            access_points_needed = max(3, int(distance / 75))
            terrain_constraints['construction_access'] = [{
                'access_points_needed': access_points_needed,
                'access_road_construction': f'{access_points_needed * 10} km of access roads estimated',
                'logistics_complexity': 'High for remote sections',
                'estimated_access_cost': access_points_needed * 1000000  # $1M per access point
            }]
        
        # Drainage requirements
        terrain_constraints['drainage_requirements'] = [{
            'culverts_needed': max(10, int(distance * 2)),  # 2 per km minimum
            'drainage_systems': 'Required for slope stability',
            'maintenance_access': 'Regular drainage maintenance points needed',
            'estimated_drainage_cost': distance * 200000  # $200k per km
        }]
        
        return terrain_constraints
    
    def _suggest_elevation_solutions(self, elevation_change: float) -> List[str]:
        """Suggest engineering solutions for elevation challenges."""
        if elevation_change < 300:
            return ["Standard grading techniques", "Minor cut and fill"]
        elif elevation_change < 600:
            return ["Extended alignment", "Moderate tunneling", "Viaducts"]
        elif elevation_change < 1000:
            return ["Spiral tunnels", "Switchback sections", "Major viaducts"]
        else:
            return ["Extensive tunneling", "Rack railway sections", "Cable-assisted sections"]
    
    def _assess_grade_suitability(self, max_grade: float) -> str:
        """Assess railway suitability based on maximum grade."""
        if max_grade <= 2.5:
            return "Excellent - Standard railway operation"
        elif max_grade <= 4.0:
            return "Good - Minor grade compensation needed"
        elif max_grade <= 6.0:
            return "Moderate - Significant engineering required"
        elif max_grade <= 8.0:
            return "Difficult - Specialized solutions needed"
        else:
            return "Very Difficult - Rack railway or major rerouting required"
    
    def _suggest_grade_mitigation(self, max_grade: float) -> List[str]:
        """Suggest mitigation strategies for steep grades."""
        if max_grade <= 2.5:
            return ["Standard construction techniques"]
        elif max_grade <= 4.0:
            return ["Longer curves", "Minor grade compensation"]
        elif max_grade <= 6.0:
            return ["Switchbacks", "Longer route alignment", "Partial tunneling"]
        elif max_grade <= 8.0:
            return ["Extensive tunneling", "Rack railway sections", "Spiral tunnels"]
        else:
            return ["Complete rerouting", "Rack railway system", "Funicular sections"]
    
    def _analyze_infrastructure_conflicts(self, route: Dict) -> Dict[str, Any]:
        """Analyze conflicts with existing infrastructure."""
        infrastructure_conflicts = {
            'road_crossings': [],
            'utility_conflicts': [],
            'airport_restrictions': [],
            'existing_rail_conflicts': [],
            'port_interactions': []
        }
        
        distance = route['total_distance_km']
        origin = route['origin']
        destination = route['destination']
        
        # Road crossing analysis
        major_road_crossings = max(2, int(distance / 25))  # 1 major road per 25km
        minor_road_crossings = max(5, int(distance / 8))   # 1 minor road per 8km
        
        infrastructure_conflicts['road_crossings'] = [{
            'major_highways': major_road_crossings,
            'minor_roads': minor_road_crossings,
            'grade_separation_required': major_road_crossings,
            'at_grade_crossings_permitted': minor_road_crossings,
            'estimated_crossing_cost': major_road_crossings * 8000000 + minor_road_crossings * 500000,  # $8M major, $500k minor
            'traffic_disruption_during_construction': "Significant - alternative routes needed"
        }]
        
        # Utility conflicts
        infrastructure_conflicts['utility_conflicts'] = [{
            'power_line_crossings': max(3, int(distance / 40)),
            'telecommunications': 'Multiple fiber optic and cell tower relocations',
            'water_gas_pipelines': 'Potential conflicts requiring detailed surveys',
            'utility_relocation_cost': distance * 300000,  # $300k per km average
            'coordination_timeline_months': max(6, distance / 10)  # Longer for longer routes
        }]
        
        # Airport proximity analysis
        for city in [origin, destination]:
            if city['population'] > 300000:  # Cities likely to have airports
                infrastructure_conflicts['airport_restrictions'].append({
                    'city': city['city_name'],
                    'airport_type': 'International' if city['population'] > 1000000 else 'Domestic',
                    'restrictions': 'Height limits, electromagnetic interference controls',
                    'coordination_required': 'Civil aviation authority approval',
                    'potential_rerouting': 'May require route modification near airport'
                })
        
        return infrastructure_conflicts
    
    def _analyze_economic_constraints(self, route: Dict) -> Dict[str, Any]:
        """Analyze economic constraints including land acquisition costs."""
        economic_constraints = {
            'land_acquisition': {},
            'construction_cost_factors': [],
            'financing_challenges': [],
            'economic_viability': {}
        }
        
        distance = route['total_distance_km']
        origin = route['origin']
        destination = route['destination']
        
        # Land acquisition cost estimation
        avg_population = (origin['population'] + destination['population']) / 2
        land_cost_per_hectare = self._estimate_land_cost(avg_population)
        
        # Railway corridor requirements
        corridor_width_m = 30  # Standard railway corridor
        land_area_hectares = distance * (corridor_width_m / 10000)  # Convert to hectares
        total_land_cost = land_area_hectares * land_cost_per_hectare
        
        economic_constraints['land_acquisition'] = {
            'corridor_width_m': corridor_width_m,
            'total_area_hectares': land_area_hectares,
            'estimated_cost_per_hectare': land_cost_per_hectare,
            'total_land_cost_usd': total_land_cost,
            'acquisition_timeline_months': 12 + int(distance / 50),  # Longer for longer routes
            'eminent_domain_risk': 'Medium' if avg_population > 500000 else 'Low'
        }
        
        # Construction cost factors
        base_cost_per_km = 20000000  # $20M base cost per km
        cost_multipliers = []
        
        # Route type impact on cost
        route_type = route['route_type']
        if route_type == 'via_intermediate':
            cost_multipliers.append(("Intermediate city stations", 1.15))
        elif route_type == 'terrain_aware':
            terrain_difficulty = route.get('terrain_difficulty', 0)
            if terrain_difficulty > 0.6:
                cost_multipliers.append(("Complex terrain routing", 1.4))
        
        # Distance impact
        if distance > 200:
            cost_multipliers.append(("Long route logistics premium", 1.1))
        
        # Urban proximity impact
        if origin['population'] > 1000000 or destination['population'] > 1000000:
            cost_multipliers.append(("Major metropolitan area premium", 1.25))
        
        # Calculate final cost
        final_cost_per_km = base_cost_per_km
        for description, multiplier in cost_multipliers:
            final_cost_per_km *= multiplier
            economic_constraints['construction_cost_factors'].append(f"{description}: +{(multiplier-1)*100:.0f}%")
        
        total_construction_cost = distance * final_cost_per_km
        economic_constraints['estimated_construction_cost'] = total_construction_cost
        
        # Economic viability analysis
        demand_data = route['demand_data']
        annual_passengers = demand_data.get('annual_passengers', 0)
        avg_ticket_price = demand_data.get('avg_ticket_price', 30)
        annual_revenue = annual_passengers * avg_ticket_price
        
        # Operating costs (simplified)
        annual_operating_cost = total_construction_cost * 0.08  # 8% of construction cost annually
        annual_profit = annual_revenue - annual_operating_cost
        
        payback_period = total_construction_cost / max(annual_profit, 1) if annual_profit > 0 else float('inf')
        
        economic_constraints['economic_viability'] = {
            'estimated_annual_revenue': annual_revenue,
            'estimated_annual_operating_cost': annual_operating_cost,
            'estimated_annual_profit': annual_profit,
            'construction_cost': total_construction_cost,
            'payback_period_years': payback_period if payback_period != float('inf') else 999,
            'viability_assessment': self._assess_economic_viability(payback_period),
            'break_even_passengers': max(1, int(annual_operating_cost / avg_ticket_price)) if avg_ticket_price > 0 else 0
        }
        
        return economic_constraints
    
    def _estimate_land_cost(self, avg_population: float) -> float:
        """Estimate land cost per hectare based on population (proxy for development)."""
        if avg_population > 2000000:
            return 300000  # $300k per hectare in major metropolitan areas
        elif avg_population > 500000:
            return 150000  # $150k per hectare in major cities
        elif avg_population > 100000:
            return 75000   # $75k per hectare in medium cities
        else:
            return 30000   # $30k per hectare in rural/small town areas
    
    def _assess_economic_viability(self, payback_period: float) -> str:
        """Assess economic viability based on payback period."""
        if payback_period == float('inf') or payback_period > 50:
            return "Poor viability - Revenue insufficient for operations"
        elif payback_period > 30:
            return "Low viability - Very long payback period"
        elif payback_period > 20:
            return "Marginal viability - Long payback period"
        elif payback_period > 10:
            return "Reasonable viability - Moderate payback period"
        else:
            return "Good viability - Acceptable payback period"
    
    def _analyze_regulatory_constraints(self, route: Dict) -> Dict[str, Any]:
        """Analyze regulatory and legal constraints."""
        regulatory_constraints = {
            'environmental_permits': [],
            'land_use_approvals': [],
            'safety_regulations': [],
            'heritage_protections': [],
            'international_considerations': []
        }
        
        distance = route['total_distance_km']
        origin = route['origin']
        destination = route['destination']
        
        # Environmental permits (mandatory for all routes)
        regulatory_constraints['environmental_permits'] = [
            'Environmental Impact Assessment (EIA) - 12-18 months',
            'Wildlife protection permits - 6-12 months',
            'Water crossing permits for bridges - 8-15 months',
            'Air quality impact assessment - 6 months',
            'Noise impact assessment and mitigation plan - 4-8 months'
        ]
        
        # Land use approvals
        regulatory_constraints['land_use_approvals'] = [
            'Railway corridor designation - 6-12 months',
            'Zoning changes for stations and facilities - 3-8 months',
            'Eminent domain proceedings if needed - 12-24 months',
            'Agricultural land conversion permits - 4-10 months',
            'Urban planning approvals - 6-15 months'
        ]
        
        # Safety regulations
        regulatory_constraints['safety_regulations'] = [
            'Railway safety authority approval - 6-12 months',
            'Grade crossing safety standards compliance',
            'Emergency access and evacuation procedures',
            'Fire safety systems for tunnels and bridges',
            'Operational safety management system certification'
        ]
        
        # Heritage and cultural considerations
        major_cities = [city for city in [origin, destination] if city['population'] > 200000]
        if major_cities or distance > 150:
            regulatory_constraints['heritage_protections'] = [
                'Archaeological survey and clearance - 6-18 months',
                'Cultural heritage impact assessment - 4-12 months',
                'Community consultation requirements - 6-24 months',
                'Indigenous/traditional land rights consultation',
                'Historical site preservation measures'
            ]
        
        return regulatory_constraints
    
    def _calculate_overall_constraint_score(self, constraints: Dict) -> float:
        """Calculate overall constraint score (0-1, where 1 is most constrained)."""
        scores = []
        
        # NIMBY score (0.3 weight)
        nimby_impacts = len(constraints['nimby_factors']['residential_impacts'])
        nimby_score = min(nimby_impacts / 4, 1.0)  # Normalize to 0-1
        scores.append(nimby_score * 0.3)
        
        # Environmental score (0.25 weight)
        env_issues = (len(constraints['environmental_constraints']['protected_areas']) +
                     len(constraints['environmental_constraints']['water_body_crossings']) +
                     len(constraints['environmental_constraints']['forest_impact']))
        env_score = min(env_issues / 3, 1.0)
        scores.append(env_score * 0.25)
        
        # Terrain score (0.2 weight)
        grade_analysis = constraints['terrain_constraints']['grade_analysis']
        grade_percent = grade_analysis.get('estimated_max_grade_percent', 0)
        terrain_score = min(grade_percent / 8, 1.0)  # 8% is very difficult
        scores.append(terrain_score * 0.2)
        
        # Economic score (0.25 weight)
        economic_data = constraints['economic_constraints']['economic_viability']
        payback_years = economic_data.get('payback_period_years', 25)
        if payback_years >= 999:  # Infinite payback
            economic_score = 1.0
        else:
            economic_score = min(payback_years / 40, 1.0)  # 40 years is very poor
        scores.append(economic_score * 0.25)
        
        return sum(scores)
    
    def _classify_constraint_severity(self, overall_score: float) -> str:
        """Classify overall constraint severity."""
        if overall_score <= 0.25:
            return "Low - Favorable conditions for development"
        elif overall_score <= 0.45:
            return "Medium - Manageable constraints with mitigation"
        elif overall_score <= 0.65:
            return "High - Significant challenges requiring extensive planning"
        else:
            return "Very High - Major obstacles, consider alternative routes"
    
    def _suggest_mitigation_strategies(self, constraints: Dict) -> List[str]:
        """Suggest mitigation strategies for identified constraints."""
        strategies = []
        
        # NIMBY mitigation
        nimby_factors = constraints['nimby_factors']
        if nimby_factors['residential_impacts']:
            strategies.extend([
                "Comprehensive community engagement starting in planning phase",
                "Noise barriers (3-5m high) along residential sections",
                "Property value guarantee programs for affected homeowners",
                "Grade-separated crossings to maintain community connectivity"
            ])
        
        # Environmental mitigation
        env_constraints = constraints['environmental_constraints']
        if env_constraints['forest_impact']:
            strategies.extend([
                "Wildlife corridors every 5-10km with native vegetation",
                "Reforestation at 2:1 ratio using native species",
                "Seasonal construction restrictions during breeding periods",
                "Continuous environmental monitoring during and after construction"
            ])
        
        # Terrain mitigation
        terrain_constraints = constraints['terrain_constraints']
        grade_analysis = terrain_constraints['grade_analysis']
        if grade_analysis.get('estimated_max_grade_percent', 0) > 5:
            strategies.extend([
                "Spiral tunnels or extended alignments to reduce grades",
                "Rack railway systems for steepest sections (>6% grade)",
                "Retaining walls and slope stabilization for cuts",
                "Advanced drainage systems for slope stability"
            ])
        
        # Economic mitigation
        economic_viability = constraints['economic_constraints']['economic_viability']
        if 'Poor' in economic_viability.get('viability_assessment', ''):
            strategies.extend([
                "Phased construction to spread costs over time",
                "Mixed passenger/freight service to increase revenue",
                "Real estate development around stations for additional revenue",
                "Public-private partnership to share risks and costs"
            ])
        
        return strategies[:12]  # Return top 12 strategies
    
    def _assess_implementation_complexity(self, constraints: Dict) -> Dict[str, Any]:
        """Assess implementation complexity based on constraints."""
        complexity_factors = []
        risk_level = "Low"
        estimated_delays_months = 0
        
        # Check various constraint factors
        nimby_score = len(constraints['nimby_factors']['residential_impacts'])
        if nimby_score > 2:
            complexity_factors.append("High community opposition risk")
            estimated_delays_months += 12
            risk_level = "High"
        
        grade_percent = constraints['terrain_constraints']['grade_analysis'].get('estimated_max_grade_percent', 0)
        if grade_percent > 6:
            complexity_factors.append("Complex engineering requirements")
            estimated_delays_months += 18
            if risk_level != "High":
                risk_level = "Medium"
        
        payback_years = constraints['economic_constraints']['economic_viability'].get('payback_period_years', 20)
        if payback_years > 30:
            complexity_factors.append("Challenging project financing")
            estimated_delays_months += 6
            if risk_level == "Low":
                risk_level = "Medium"
        
        if len(constraints['environmental_constraints']['forest_impact']) > 0:
            complexity_factors.append("Extensive environmental compliance required")
            estimated_delays_months += 9
        
        return {
            'complexity_level': risk_level,
            'key_complexity_factors': complexity_factors,
            'estimated_additional_delays_months': estimated_delays_months,
            'recommended_project_management': "Enhanced" if risk_level in ["Medium", "High"] else "Standard",
            'stakeholder_engagement_intensity': "High" if nimby_score > 1 else "Medium"
        }
    
    def _save_constraint_analysis(self, constraint_analysis: Dict) -> None:
        """Save constraint analysis to file."""
        output_path = self.output_dir / "constraint_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(constraint_analysis, f, indent=2, default=str)
        
        self.logger.info(f"✅ Constraint analysis saved to {output_path}")