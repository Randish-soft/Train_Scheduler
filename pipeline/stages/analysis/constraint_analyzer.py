import logging
from typing import Dict, Any, List

class ConstraintAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze constraints for rail development"""
        self.logger.info("Analyzing development constraints")
        
        country_data = context['country_data']
        budget_constraints = context['budget_constraints']
        terrain_data = context['terrain_data']
        demand_data = context['demand_data']
        
        constraints = {
            'budget_limitations': self._analyze_budget_constraints(budget_constraints, demand_data),
            'terrain_limitations': self._analyze_terrain_constraints(terrain_data),
            'regulatory_limitations': self._analyze_regulatory_constraints(country_data),
            'temporal_limitations': self._analyze_temporal_constraints(context),
            'technical_limitations': self._analyze_technical_constraints(country_data)
        }
        
        # Calculate overall constraint severity
        constraints['overall_severity'] = self._calculate_overall_severity(constraints)
        
        context['constraints'] = constraints
        self.logger.info("Constraint analysis completed")
        return context
    
    def _analyze_budget_constraints(self, budget_constraints: Dict[str, Any], demand_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze budget limitations and their impact"""
        total_budget = budget_constraints['total_budget']
        infrastructure_budget = budget_constraints['infrastructure_allocation']
        
        # Estimate total infrastructure cost based on demand
        total_demand_km = sum(corridor['distance_km'] for corridor in demand_data['demand_corridors'])
        
        # Rough cost estimation (USD per km)
        avg_cost_per_km = 20000000  # $20M per km average
        estimated_total_cost = total_demand_km * avg_cost_per_km
        
        budget_sufficiency = infrastructure_budget / estimated_total_cost if estimated_total_cost > 0 else 1.0
        
        return {
            'total_budget': total_budget,
            'infrastructure_budget': infrastructure_budget,
            'estimated_total_cost': estimated_total_cost,
            'budget_sufficiency': budget_sufficiency,
            'severity': 'critical' if budget_sufficiency < 0.5 else 'high' if budget_sufficiency < 0.8 else 'medium',
            'recommendations': self._generate_budget_recommendations(budget_sufficiency, total_budget)
        }
    
    def _generate_budget_recommendations(self, budget_sufficiency: float, total_budget: float) -> List[str]:
        """Generate budget-related recommendations"""
        recommendations = []
        
        if budget_sufficiency < 0.5:
            recommendations.extend([
                "Consider phased implementation starting with highest priority corridors",
                "Explore public-private partnership funding models",
                "Prioritize cost-effective construction methods",
                "Consider starting with single-track lines instead of double-track"
            ])
        elif budget_sufficiency < 0.8:
            recommendations.extend([
                "Optimize route selection to minimize expensive terrain crossings",
                "Consider modular station designs to reduce costs",
                "Plan for future expansion rather than building everything at once"
            ])
        else:
            recommendations.append("Budget appears sufficient for comprehensive network development")
        
        if total_budget < 100000000:  # Less than $100M
            recommendations.append("Consider starting with light rail or bus rapid transit as lower-cost alternatives")
        
        return recommendations
    
    def _analyze_terrain_constraints(self, terrain_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze terrain-related constraints"""
        terrain_difficulty = terrain_data.get('terrain_difficulty', 0)
        obstacles = terrain_data.get('obstacles', [])
        water_bodies = terrain_data.get('rivers_lakes', [])
        
        # Calculate terrain constraint severity
        if terrain_difficulty > 0.8:
            severity = 'critical'
        elif terrain_difficulty > 0.6:
            severity = 'high'
        elif terrain_difficulty > 0.4:
            severity = 'medium'
        else:
            severity = 'low'
        
        major_obstacles = [obs for obs in obstacles if obs['difficulty'] in ['high', 'critical']]
        major_water_crossings = [wb for wb in water_bodies if wb['crossing_difficulty'] == 'high']
        
        return {
            'terrain_difficulty_score': terrain_difficulty,
            'major_obstacles_count': len(major_obstacles),
            'major_water_crossings_count': len(major_water_crossings),
            'severity': severity,
            'challenges': self._identify_terrain_challenges(terrain_data),
            'mitigation_strategies': self._suggest_terrain_mitigations(terrain_data)
        }
    
    def _identify_terrain_challenges(self, terrain_data: Dict[str, Any]) -> List[str]:
        """Identify specific terrain challenges"""
        challenges = []
        terrain_type = terrain_data.get('type', 'mixed')
        
        if terrain_type == 'mountainous':
            challenges.extend([
                "Steep gradients requiring specialized traction",
                "Extensive tunneling increasing costs",
                "Avalanche/landslide risks in certain areas",
                "Limited access for construction equipment"
            ])
        elif terrain_type == 'flat':
            challenges.extend([
                "Potential flooding risks in low-lying areas",
                "Limited natural barriers for noise reduction",
                "Higher land acquisition costs in developed flat areas"
            ])
        
        if terrain_data.get('rivers_lakes'):
            challenges.append("Multiple major water crossings requiring bridges")
        
        if any(obs['type'] == 'urban_area' for obs in terrain_data.get('obstacles', [])):
            challenges.append("Dense urban areas complicating route alignment")
        
        return challenges
    
    def _suggest_terrain_mitigations(self, terrain_data: Dict[str, Any]) -> List[str]:
        """Suggest mitigation strategies for terrain challenges"""
        mitigations = []
        terrain_type = terrain_data.get('type', 'mixed')
        
        if terrain_type == 'mountainous':
            mitigations.extend([
                "Use rack railways or cogwheels for steep sections",
                "Plan tunnels through mountain passes rather than going over",
                "Implement comprehensive slope stabilization measures"
            ])
        
        if any(wb['crossing_difficulty'] == 'high' for wb in terrain_data.get('rivers_lakes', [])):
            mitigations.extend([
                "Use pre-fabricated bridge components to reduce construction time",
                "Consider submerged tunnels for very wide water crossings",
                "Coordinate with environmental agencies for sensitive areas"
            ])
        
        if any(obs['type'] == 'urban_area' for obs in terrain_data.get('obstacles', [])):
            mitigations.extend([
                "Consider elevated or underground sections through dense urban areas",
                "Use existing transportation corridors (highways) where possible",
                "Implement cut-and-cover construction to minimize surface disruption"
            ])
        
        return mitigations
    
    def _analyze_regulatory_constraints(self, country_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze regulatory and legal constraints"""
        existing_infrastructure = country_data.get('existing_infrastructure', {})
        regulatory_environment = country_data.get('regulatory_environment', {})
        
        constraints = {
            'land_acquisition_difficulty': regulatory_environment.get('land_acquisition_difficulty', 'medium'),
            'environmental_regulations': regulatory_environment.get('environmental_regulations', 'medium'),
            'safety_standards': regulatory_environment.get('safety_standards', 'international'),
            'permitting_timeline_months': regulatory_environment.get('permitting_timeline', 24)
        }
        
        # Calculate regulatory severity
        severity_factors = []
        if constraints['land_acquisition_difficulty'] == 'high':
            severity_factors.append(0.8)
        if constraints['environmental_regulations'] == 'strict':
            severity_factors.append(0.7)
        if constraints['permitting_timeline_months'] > 36:
            severity_factors.append(0.6)
        
        avg_severity = sum(severity_factors) / len(severity_factors) if severity_factors else 0.3
        
        if avg_severity > 0.7:
            severity = 'critical'
        elif avg_severity > 0.5:
            severity = 'high'
        elif avg_severity > 0.3:
            severity = 'medium'
        else:
            severity = 'low'
        
        constraints['severity'] = severity
        constraints['recommendations'] = self._generate_regulatory_recommendations(constraints)
        
        return constraints
    
    def _generate_regulatory_recommendations(self, regulatory_constraints: Dict[str, Any]) -> List[str]:
        """Generate recommendations for regulatory challenges"""
        recommendations = []
        
        if regulatory_constraints['land_acquisition_difficulty'] == 'high':
            recommendations.extend([
                "Start land acquisition process early in project timeline",
                "Consider alternative routes with lower land acquisition complexity",
                "Explore land value capture financing mechanisms"
            ])
        
        if regulatory_constraints['environmental_regulations'] == 'strict':
            recommendations.extend([
                "Conduct comprehensive environmental impact assessments early",
                "Engage with environmental agencies during planning phase",
                "Consider eco-friendly construction methods and materials"
            ])
        
        if regulatory_constraints['permitting_timeline_months'] > 24:
            recommendations.append("Apply for permits in parallel rather than sequentially where possible")
        
        return recommendations
    
    def _analyze_temporal_constraints(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze time-related constraints"""
        user_priorities = context['user_priorities']
        time_frame = user_priorities.get('time_frame', 'medium_term')
        
        # Map time frames to realistic timelines
        time_mapping = {
            'short_term': (12, 36),   # 1-3 years
            'medium_term': (36, 84),  # 3-7 years  
            'long_term': (84, 180)    # 7-15 years
        }
        
        min_months, max_months = time_mapping.get(time_frame, (36, 84))
        
        # Adjust based on terrain difficulty
        terrain_difficulty = context['terrain_data'].get('terrain_difficulty', 0.5)
        time_multiplier = 1.0 + (terrain_difficulty * 0.5)  # 50% increase for difficult terrain
        
        adjusted_min = min_months * time_multiplier
        adjusted_max = max_months * time_multiplier
        
        return {
            'user_time_preference': time_frame,
            'estimated_min_months': adjusted_min,
            'estimated_max_months': adjusted_max,
            'realistic_timeline': f"{int(adjusted_min/12)}-{int(adjusted_max/12)} years",
            'critical_path_items': self._identify_critical_path_items(context),
            'severity': 'high' if time_frame == 'short_term' and terrain_difficulty > 0.6 else 'medium'
        }
    
    def _identify_critical_path_items(self, context: Dict[str, Any]) -> List[str]:
        """Identify items on the critical path for timeline"""
        critical_items = ["Land acquisition", "Environmental permitting", "Detailed design"]
        
        terrain_data = context['terrain_data']
        if terrain_data.get('terrain_difficulty', 0) > 0.7:
            critical_items.append("Major tunneling projects")
        
        if any(wb['crossing_difficulty'] == 'high' for wb in terrain_data.get('rivers_lakes', [])):
            critical_items.append("Major bridge construction")
        
        return critical_items
    
    def _analyze_technical_constraints(self, country_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze technical and engineering constraints"""
        existing_rail = country_data.get('existing_infrastructure', {}).get('rail', {})
        technical_capability = country_data.get('technical_capability', 'medium')
        
        constraints = {
            'local_engineering_expertise': technical_capability,
            'existing_rail_standards': existing_rail.get('track_gauge', 'standard'),
            'electrification_existing': existing_rail.get('electrification', 'none'),
            'signal_systems': existing_rail.get('signaling', 'none')
        }
        
        # Assess technical severity
        if technical_capability == 'low':
            severity = 'high'
        elif technical_capability == 'medium':
            severity = 'medium'
        else:
            severity = 'low'
        
        constraints['severity'] = severity
        constraints['recommendations'] = self._generate_technical_recommendations(constraints)
        
        return constraints
    
    def _generate_technical_recommendations(self, technical_constraints: Dict[str, Any]) -> List[str]:
        """Generate technical recommendations"""
        recommendations = []
        
        if technical_constraints['local_engineering_expertise'] == 'low':
            recommendations.extend([
                "Consider international engineering partnerships",
                "Plan for comprehensive knowledge transfer programs",
                "Start with simpler rail technologies before advancing to complex systems"
            ])
        
        if technical_constraints['electrification_existing'] == 'none':
            recommendations.append("Consider diesel multiple units for initial deployment to simplify electrification")
        
        return recommendations
    
    def _calculate_overall_severity(self, constraints: Dict[str, Any]) -> str:
        """Calculate overall constraint severity"""
        severities = {
            'critical': 4,
            'high': 3, 
            'medium': 2,
            'low': 1
        }
        
        # Get severity from each constraint category
        budget_severity = constraints['budget_limitations']['severity']
        terrain_severity = constraints['terrain_limitations']['severity']
        regulatory_severity = constraints['regulatory_limitations']['severity']
        temporal_severity = constraints['temporal_limitations']['severity']
        technical_severity = constraints['technical_limitations']['severity']
        
        # Calculate weighted average
        scores = [
            severities[budget_severity] * 0.3,      # Budget is most important
            severities[terrain_severity] * 0.25,    # Terrain is second
            severities[regulatory_severity] * 0.2,  # Regulatory third
            severities[temporal_severity] * 0.15,   # Time fourth
            severities[technical_severity] * 0.1    # Technical least
        ]
        
        avg_score = sum(scores)
        
        if avg_score >= 3.5:
            return 'critical'
        elif avg_score >= 2.5:
            return 'high'
        elif avg_score >= 1.5:
            return 'medium'
        else:
            return 'low'