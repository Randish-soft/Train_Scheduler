import logging
from typing import Dict, Any, List, Optional

class CountryReference:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.reference_countries = self._load_reference_countries()
    
    def find_similar_countries(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Find countries with similar characteristics for reference"""
        self.logger.info("Finding similar countries for reference")
        
        country_data = context['country_data']
        country_features = context['country_features']
        
        similar_countries = self._find_similar_countries(country_features)
        best_match = self._select_best_match(similar_countries, country_features) if similar_countries else None
        
        context['reference_data'] = {
            'similar_countries': similar_countries,
            'best_match': best_match,
            'applicable_patterns': self._extract_applicable_patterns(best_match, country_features) if best_match else []
        }
        
        self.logger.info(f"Found {len(similar_countries)} similar countries")
        return context
    
    def _load_reference_countries(self) -> List[Dict[str, Any]]:
        """Load database of reference countries with their rail systems"""
        return [
            {
                'name': 'Switzerland',
                'population_density': 219,
                'terrain_type': 'mountainous',
                'development_level': 'developed',
                'rail_system': {
                    'total_km': 5300,
                    'urban_lines': 1200,
                    'regional_lines': 2800,
                    'high_speed_lines': 1300,
                    'notable_features': ['mountain_tunnels', 'efficient_transfers', 'punctuality'],
                    'cost_per_km': 25000000  # USD
                }
            },
            {
                'name': 'Netherlands',
                'population_density': 508,
                'terrain_type': 'flat',
                'development_level': 'developed', 
                'rail_system': {
                    'total_km': 3200,
                    'urban_lines': 800,
                    'regional_lines': 2000,
                    'high_speed_lines': 400,
                    'notable_features': ['dense_network', 'bike_integration', 'electrification'],
                    'cost_per_km': 18000000
                }
            },
            {
                'name': 'Japan',
                'population_density': 347,
                'terrain_type': 'mixed', 
                'development_level': 'developed',
                'rail_system': {
                    'total_km': 27500,
                    'urban_lines': 8000,
                    'regional_lines': 15000,
                    'high_speed_lines': 4500,
                    'notable_features': ['shinkansen', 'precision_timing', 'private_operators'],
                    'cost_per_km': 35000000
                }
            },
            {
                'name': 'Lebanon',
                'population_density': 667,
                'terrain_type': 'mixed',
                'development_level': 'developing',
                'rail_system': {
                    'total_km': 0,  # Currently no functional rail
                    'urban_lines': 0,
                    'regional_lines': 0,
                    'high_speed_lines': 0,
                    'notable_features': ['coastal_challenges', 'urban_density', 'terrain_issues'],
                    'cost_per_km': 15000000
                }
            }
        ]
    
    def _find_similar_countries(self, country_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find countries with similar characteristics"""
        target_density = country_features.get('population_density', 0)
        target_terrain = country_features.get('terrain_type', 'mixed')
        target_development = country_features.get('development_level', 'developing')
        
        similar_countries = []
        
        for country in self.reference_countries:
            similarity_score = self._calculate_similarity_score(country, target_density, target_terrain, target_development)
            
            if similarity_score > 0.5:  # Threshold for similarity
                country['similarity_score'] = similarity_score
                similar_countries.append(country)
        
        # Sort by similarity score descending
        similar_countries.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar_countries
    
    def _calculate_similarity_score(self, country: Dict[str, Any], target_density: float, 
                                  target_terrain: str, target_development: str) -> float:
        """Calculate similarity score between target and reference country"""
        density_score = 1 - min(1.0, abs(country['population_density'] - target_density) / max(target_density, 1))
        terrain_score = 1.0 if country['terrain_type'] == target_terrain else 0.3
        development_score = 1.0 if country['development_level'] == target_development else 0.5
        
        # Weighted average
        return (density_score * 0.4 + terrain_score * 0.4 + development_score * 0.2)
    
    def _select_best_match(self, similar_countries: List[Dict[str, Any]], country_features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select the best matching country for detailed reference"""
        if not similar_countries:
            return None
        
        # Return the highest scoring match
        best_match = similar_countries[0]
        
        # Enhance with specific recommendations
        best_match['recommendations'] = self._generate_country_recommendations(best_match, country_features)
        
        return best_match
    
    def _generate_country_recommendations(self, reference_country: Dict[str, Any], country_features: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on reference country"""
        recommendations = []
        rail_system = reference_country['rail_system']
        
        # Terrain-specific recommendations
        if reference_country['terrain_type'] == 'mountainous':
            recommendations.extend([
                "Consider extensive tunneling through mountainous regions",
                "Implement cogwheel or rack railway systems for steep gradients",
                "Plan for frequent maintenance due to terrain wear"
            ])
        elif reference_country['terrain_type'] == 'flat':
            recommendations.extend([
                "Optimize for straight, high-speed routes",
                "Consider elevated tracks to minimize land acquisition",
                "Plan for extensive cycling integration at stations"
            ])
        
        # Density-specific recommendations
        if reference_country['population_density'] > 400:
            recommendations.extend([
                "Focus on high-frequency urban and suburban services",
                "Implement integrated ticketing with other transit modes",
                "Consider underground stations in dense urban centers"
            ])
        
        # Development level recommendations
        if country_features['development_level'] == 'undeveloped':
            recommendations.extend([
                "Start with core inter-city connections",
                "Consider phased implementation with bus feeder services",
                "Focus on cost-effective construction methods"
            ])
        
        return recommendations
    
    def _extract_applicable_patterns(self, best_match: Dict[str, Any], country_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract applicable rail patterns from reference country"""
        if not best_match:
            return []
        
        rail_system = best_match['rail_system']
        patterns = []
        
        # Network structure patterns
        if rail_system['urban_lines'] > 0:
            patterns.append({
                'type': 'urban_network',
                'description': 'Dense urban rail network with frequent stops',
                'applicability': 'high' if country_features['population_density'] > 300 else 'medium'
            })
        
        if rail_system['high_speed_lines'] > 0:
            patterns.append({
                'type': 'high_speed_corridors',
                'description': 'Dedicated high-speed lines connecting major cities',
                'applicability': 'medium'
            })
        
        # Terrain-specific patterns
        if best_match['terrain_type'] == 'mountainous' and 'mountain_tunnels' in rail_system['notable_features']:
            patterns.append({
                'type': 'mountain_tunneling',
                'description': 'Extensive tunneling through mountainous terrain',
                'applicability': 'high' if country_features['terrain_type'] == 'mountainous' else 'low'
            })
        
        return patterns