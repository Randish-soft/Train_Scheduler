import logging
from typing import Dict, Any, List

class NIMBYAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze NIMBY (Not In My Backyard) issues for proposed routes"""
        self.logger.info("Analyzing NIMBY factors")
        
        proposed_routes = context.get('proposed_routes', [])
        terrain_data = context['terrain_data']
        country_data = context['country_data']
        
        analyzed_routes = []
        for route in proposed_routes:
            nimby_analysis = self._analyze_route_nimby(route, terrain_data, country_data)
            route['nimby_analysis'] = nimby_analysis
            analyzed_routes.append(route)
        
        context['proposed_routes'] = analyzed_routes
        context['nimby_summary'] = self._generate_nimby_summary(analyzed_routes)
        
        self.logger.info("NIMBY analysis completed")
        return context
    
    def _analyze_route_nimby(self, route: Dict[str, Any], terrain_data: Dict[str, Any], 
                           country_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze NIMBY factors for a specific route"""
        route_alignments = route.get('possible_alignments', [])
        cities_served = route.get('cities_served', [])
        
        nimby_factors = {
            'urban_impact': self._assess_urban_impact(route_alignments, cities_served),
            'environmental_impact': self._assess_environmental_impact(route_alignments, terrain_data),
            'cultural_heritage_impact': self._assess_cultural_impact(route_alignments, country_data),
            'visual_impact': self._assess_visual_impact(route_alignments, terrain_data),
            'noise_impact': self._assess_noise_impact(route_alignments, cities_served)
        }
        
        # Calculate overall NIMBY score (0-1, higher = more problematic)
        overall_score = self._calculate_nimby_score(nimby_factors)
        
        return {
            'factors': nimby_factors,
            'overall_score': overall_score,
            'risk_level': self._assess_risk_level(overall_score),
            'mitigation_strategies': self._suggest_mitigation_strategies(nimby_factors, overall_score),
            'community_engagement_recommendations': self._suggest_community_engagement(nimby_factors)
        }
    
    def _assess_urban_impact(self, route_alignments: List[Dict[str, Any]], cities_served: List[str]) -> Dict[str, Any]:
        """Assess impact on urban areas"""
        high_density_areas = 0
        residential_areas_affected = 0
        business_disruption = 0
        
        for alignment in route_alignments:
            # Simulate urban impact assessment
            alignment_type = alignment.get('type', 'elevated')
            urban_density = alignment.get('urban_density', 'medium')
            
            if urban_density == 'high':
                high_density_areas += 1
                residential_areas_affected += 2
                business_disruption += 1
            elif urban_density == 'medium':
                residential_areas_affected += 1
        
        impact_score = min(1.0, (high_density_areas * 0.3 + residential_areas_affected * 0.2 + business_disruption * 0.1))
        
        return {
            'score': impact_score,
            'high_density_areas_affected': high_density_areas,
            'residential_areas_affected': residential_areas_affected,
            'business_disruption_potential': business_disruption,
            'severity': 'high' if impact_score > 0.7 else 'medium' if impact_score > 0.4 else 'low'
        }
    
    def _assess_environmental_impact(self, route_alignments: List[Dict[str, Any]], terrain_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess environmental impact"""
        protected_areas = 0
        water_crossings = 0
        forest_areas = 0
        
        obstacles = terrain_data.get('obstacles', [])
        water_bodies = terrain_data.get('rivers_lakes', [])
        
        # Check for protected areas in route path
        for alignment in route_alignments:
            # Simulate environmental assessment
            if alignment.get('near_protected_area', False):
                protected_areas += 1
            
            if alignment.get('crosses_water', False):
                water_crossings += 1
            
            if alignment.get('through_forest', False):
                forest_areas += 1
        
        impact_score = min(1.0, (protected_areas * 0.4 + water_crossings * 0.3 + forest_areas * 0.2))
        
        return {
            'score': impact_score,
            'protected_areas_affected': protected_areas,
            'water_body_crossings': water_crossings,
            'forest_areas_affected': forest_areas,
            'severity': 'high' if impact_score > 0.6 else 'medium' if impact_score > 0.3 else 'low'
        }
    
    def _assess_cultural_impact(self, route_alignments: List[Dict[str, Any]], country_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess impact on cultural heritage sites"""
        heritage_sites = country_data.get('cultural_heritage_sites', [])
        sites_affected = 0
        
        for alignment in route_alignments:
            # Check if alignment affects any heritage sites
            if alignment.get('near_heritage_site', False):
                sites_affected += 1
        
        impact_score = min(1.0, sites_affected * 0.5)  # High weight for heritage sites
        
        return {
            'score': impact_score,
            'heritage_sites_affected': sites_affected,
            'severity': 'high' if sites_affected > 0 else 'low',
            'recommendations': ['Conduct archaeological survey'] if sites_affected > 0 else []
        }
    
    def _assess_visual_impact(self, route_alignments: List[Dict[str, Any]], terrain_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess visual impact of the route"""
        scenic_areas_affected = 0
        elevated_sections = 0
        
        for alignment in route_alignments:
            alignment_type = alignment.get('type', 'ground_level')
            
            if alignment_type == 'elevated':
                elevated_sections += 1
            
            if alignment.get('scenic_area', False):
                scenic_areas_affected += 1
        
        impact_score = min(1.0, (scenic_areas_affected * 0.6 + elevated_sections * 0.2))
        
        return {
            'score': impact_score,
            'scenic_areas_affected': scenic_areas_affected,
            'elevated_sections_km': elevated_sections,
            'severity': 'high' if impact_score > 0.5 else 'medium' if impact_score > 0.2 else 'low'
        }
    
    def _assess_noise_impact(self, route_alignments: List[Dict[str, Any]], cities_served: List[str]) -> Dict[str, Any]:
        """Assess noise impact on communities"""
        residential_proximity = 0
        sensitive_areas = 0  # Schools, hospitals, etc.
        
        for alignment in route_alignments:
            urban_density = alignment.get('urban_density', 'low')
            
            if urban_density in ['high', 'medium']:
                residential_proximity += 1
            
            if alignment.get('near_sensitive_area', False):
                sensitive_areas += 1
        
        impact_score = min(1.0, (residential_proximity * 0.4 + sensitive_areas * 0.6))
        
        return {
            'score': impact_score,
            'residential_areas_near_track': residential_proximity,
            'sensitive_areas_affected': sensitive_areas,
            'severity': 'high' if impact_score > 0.6 else 'medium' if impact_score > 0.3 else 'low',
            'mitigation': ['Noise barriers', 'Speed restrictions in sensitive areas'] if impact_score > 0.3 else []
        }
    
    def _calculate_nimby_score(self, nimby_factors: Dict[str, Any]) -> float:
        """Calculate overall NIMBY score from all factors"""
        weights = {
            'urban_impact': 0.25,
            'environmental_impact': 0.20,
            'cultural_heritage_impact': 0.15,
            'visual_impact': 0.20,
            'noise_impact': 0.20
        }
        
        total_score = 0
        for factor, weight in weights.items():
            factor_score = nimby_factors[factor]['score']
            total_score += factor_score * weight
        
        return min(1.0, total_score)
    
    def _assess_risk_level(self, nimby_score: float) -> str:
        """Assess overall NIMBY risk level"""
        if nimby_score > 0.7:
            return 'critical'
        elif nimby_score > 0.5:
            return 'high'
        elif nimby_score > 0.3:
            return 'medium'
        else:
            return 'low'
    
    def _suggest_mitigation_strategies(self, nimby_factors: Dict[str, Any], overall_score: float) -> List[str]:
        """Suggest strategies to mitigate NIMBY concerns"""
        strategies = []
        
        if nimby_factors['urban_impact']['severity'] in ['high', 'critical']:
            strategies.extend([
                "Consider underground sections through dense urban areas",
                "Use existing transportation corridors",
                "Implement property value protection programs"
            ])
        
        if nimby_factors['environmental_impact']['severity'] in ['high', 'critical']:
            strategies.extend([
                "Route around protected natural areas",
                "Implement wildlife crossings and corridors",
                "Use environmentally friendly construction methods"
            ])
        
        if nimby_factors['cultural_heritage_impact']['severity'] == 'high':
            strategies.append("Reroute to avoid heritage sites or use tunneling")
        
        if nimby_factors['visual_impact']['severity'] in ['high', 'critical']:
            strategies.extend([
                "Use landscaping to screen views",
                "Consider cut-and-cover tunnels in scenic areas",
                "Minimize elevated sections in visually sensitive areas"
            ])
        
        if nimby_factors['noise_impact']['severity'] in ['high', 'critical']:
            strategies.extend([
                "Install noise barriers along residential sections",
                "Use noise-reducing track technology",
                "Implement speed restrictions near sensitive areas"
            ])
        
        if overall_score > 0.6:
            strategies.append("Develop comprehensive community benefits package")
        
        return strategies
    
    def _suggest_community_engagement(self, nimby_factors: Dict[str, Any]) -> List[str]:
        """Suggest community engagement strategies"""
        engagement = ["Establish regular community consultation meetings"]
        
        if nimby_factors['urban_impact']['severity'] in ['high', 'critical']:
            engagement.extend([
                "Create neighborhood advisory committees",
                "Develop transparent property acquisition process",
                "Offer relocation assistance where needed"
            ])
        
        if nimby_factors['environmental_impact']['severity'] in ['high', 'critical']:
            engagement.append("Engage with environmental organizations early")
        
        if any(factor['severity'] in ['high', 'critical'] for factor in nimby_factors.values()):
            engagement.extend([
                "Create project information center",
                "Develop regular progress updates for affected communities",
                "Establish grievance resolution mechanism"
            ])
        
        return engagement
    
    def _generate_nimby_summary(self, analyzed_routes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of NIMBY analysis across all routes"""
        if not analyzed_routes:
            return {'total_routes': 0, 'average_nimby_score': 0, 'high_risk_routes': 0}
        
        total_routes = len(analyzed_routes)
        total_score = sum(route['nimby_analysis']['overall_score'] for route in analyzed_routes)
        average_score = total_score / total_routes
        
        high_risk_routes = sum(1 for route in analyzed_routes 
                              if route['nimby_analysis']['risk_level'] in ['high', 'critical'])
        
        return {
            'total_routes': total_routes,
            'average_nimby_score': average_score,
            'high_risk_routes': high_risk_routes,
            'percentage_high_risk': (high_risk_routes / total_routes) * 100 if total_routes > 0 else 0,
            'overall_risk_level': 'high' if average_score > 0.5 else 'medium' if average_score > 0.3 else 'low'
        }