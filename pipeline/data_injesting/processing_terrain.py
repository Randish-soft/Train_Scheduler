"""
Terrain Processing Module
=========================

Analyzes terrain and geographic factors affecting railway construction:
- Elevation analysis and topographic complexity
- Slope calculations and gradient analysis
- Geographic obstacles (rivers, mountains, valleys)
- Construction difficulty assessment
- Cost impact estimation based on terrain

Author: Miguel Ibrahim E
"""

import logging
import math
import requests
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import statistics


@dataclass
class TerrainPoint:
    """Data class for terrain information at a specific point."""
    latitude: float
    longitude: float
    elevation: float
    slope: Optional[float] = None
    terrain_type: Optional[str] = None


@dataclass
class RouteTerrainAnalysis:
    """Analysis results for terrain along a route."""
    route_name: str
    distance_km: float
    max_elevation: float
    min_elevation: float
    elevation_gain: float
    max_slope: float
    avg_slope: float
    difficulty_score: float
    construction_complexity: str
    estimated_cost_multiplier: float
    terrain_obstacles: List[str]


class TerrainProcessor:
    """Main class for processing terrain and geographic data."""
    
    def __init__(self, country: str):
        self.country = country
        self.logger = logging.getLogger(__name__)
        self.elevation_cache = {}
        
    def process(self) -> Dict[str, Any]:
        """
        Main processing method that analyzes terrain for the country.
        
        Returns:
            Dictionary containing terrain analysis results
        """
        self.logger.info(f"🏔️ Processing terrain data for {self.country}...")
        
        # Get country-specific terrain characteristics
        country_terrain = self._get_country_terrain_profile()
        
        # Analyze elevation patterns
        elevation_analysis = self._analyze_elevation_patterns()
        
        # Calculate construction complexity
        construction_analysis = self._analyze_construction_complexity()
        
        # Identify geographic obstacles
        obstacles = self._identify_geographic_obstacles()
        
        # Generate terrain recommendations
        recommendations = self._generate_terrain_recommendations(
            country_terrain, elevation_analysis, construction_analysis
        )
        
        results = {
            'country_terrain_profile': country_terrain,
            'elevation_analysis': elevation_analysis,
            'construction_complexity': construction_analysis,
            'geographic_obstacles': obstacles,
            'terrain_recommendations': recommendations,
            'overall_difficulty': self._calculate_overall_difficulty(country_terrain, elevation_analysis),
            'cost_impact_assessment': self._assess_cost_impact(construction_analysis)
        }
        
        self.logger.info("✅ Terrain processing completed")
        return results
    
    def _get_country_terrain_profile(self) -> Dict[str, Any]:
        """Get general terrain characteristics for the country."""
        # Built-in terrain profiles for known countries
        terrain_profiles = {
            "ghana": {
                "terrain_type": "coastal_plains_highlands",
                "max_elevation": 885,  # Mount Afadja
                "avg_elevation": 190,
                "coastal_length_km": 539,
                "major_rivers": ["Volta", "Ankobra", "Tano"],
                "terrain_zones": {
                    "coastal_plains": {"percentage": 40, "elevation_range": [0, 150], "difficulty": "low"},
                    "forest_hills": {"percentage": 35, "elevation_range": [150, 500], "difficulty": "medium"},
                    "northern_plains": {"percentage": 20, "elevation_range": [100, 300], "difficulty": "low"},
                    "eastern_highlands": {"percentage": 5, "elevation_range": [300, 885], "difficulty": "high"}
                },
                "climate_challenges": ["tropical_rainfall", "humidity", "seasonal_flooding"],
                "soil_types": ["laterite", "sandy", "clay"],
                "seismic_activity": "low"
            },
            "nigeria": {
                "terrain_type": "diverse_coastal_to_plateau",
                "max_elevation": 2419,  # Chappal Waddi
                "avg_elevation": 380,
                "coastal_length_km": 853,
                "major_rivers": ["Niger", "Benue", "Cross"],
                "terrain_zones": {
                    "coastal_plains": {"percentage": 25, "elevation_range": [0, 200], "difficulty": "low"},
                    "middle_belt": {"percentage": 40, "elevation_range": [200, 600], "difficulty": "medium"},
                    "northern_plains": {"percentage": 25, "elevation_range": [200, 500], "difficulty": "low"},
                    "eastern_highlands": {"percentage": 10, "elevation_range": [500, 2419], "difficulty": "high"}
                },
                "climate_challenges": ["monsoons", "sahel_conditions", "flooding"],
                "soil_types": ["alluvial", "laterite", "sandy", "clay"],
                "seismic_activity": "low"
            },
            "kenya": {
                "terrain_type": "rift_valley_highlands",
                "max_elevation": 5199,  # Mount Kenya
                "avg_elevation": 762,
                "coastal_length_km": 536,
                "major_rivers": ["Tana", "Galana", "Ewaso Ng'iro"],
                "terrain_zones": {
                    "coastal_plains": {"percentage": 20, "elevation_range": [0, 500], "difficulty": "low"},
                    "central_highlands": {"percentage": 30, "elevation_range": [1000, 3000], "difficulty": "high"},
                    "rift_valley": {"percentage": 25, "elevation_range": [500, 2000], "difficulty": "medium"},
                    "northern_plains": {"percentage": 25, "elevation_range": [200, 1000], "difficulty": "medium"}
                },
                "climate_challenges": ["altitude_variation", "seasonal_rains", "drought"],
                "soil_types": ["volcanic", "alluvial", "sandy", "clay"],
                "seismic_activity": "moderate"
            },
            "rwanda": {
                "terrain_type": "mountainous_highlands",
                "max_elevation": 4519,  # Mount Karisimbi
                "avg_elevation": 1598,
                "coastal_length_km": 0,  # Landlocked
                "major_rivers": ["Kagera", "Nyabarongo", "Akanyaru"],
                "terrain_zones": {
                    "western_highlands": {"percentage": 40, "elevation_range": [1200, 4519], "difficulty": "very_high"},
                    "central_plateau": {"percentage": 35, "elevation_range": [1200, 2000], "difficulty": "high"},
                    "eastern_lowlands": {"percentage": 25, "elevation_range": [900, 1500], "difficulty": "medium"}
                },
                "climate_challenges": ["high_altitude", "steep_slopes", "erosion"],
                "soil_types": ["volcanic", "laterite", "clay"],
                "seismic_activity": "moderate"
            },
            "uganda": {
                "terrain_type": "plateau_with_mountains",
                "max_elevation": 5109,  # Margherita Peak
                "avg_elevation": 1100,
                "coastal_length_km": 0,  # Landlocked
                "major_rivers": ["Nile", "Kagera", "Katonga"],
                "terrain_zones": {
                    "central_plateau": {"percentage": 50, "elevation_range": [900, 1500], "difficulty": "medium"},
                    "eastern_mountains": {"percentage": 15, "elevation_range": [1000, 4321], "difficulty": "high"},
                    "western_rift": {"percentage": 20, "elevation_range": [600, 5109], "difficulty": "very_high"},
                    "northern_plains": {"percentage": 15, "elevation_range": [500, 1200], "difficulty": "low"}
                },
                "climate_challenges": ["equatorial_climate", "lake_effects", "swamps"],
                "soil_types": ["volcanic", "alluvial", "laterite"],
                "seismic_activity": "moderate"
            }
        }
        
        country_key = self.country.lower()
        if country_key in terrain_profiles:
            self.logger.info(f"✅ Using detailed terrain profile for {self.country}")
            return terrain_profiles[country_key]
        else:
            # Generic terrain profile based on geographic region
            self.logger.info(f"🌍 Using generic terrain profile for {self.country}")
            return self._generate_generic_terrain_profile()
    
    def _generate_generic_terrain_profile(self) -> Dict[str, Any]:
        """Generate a generic terrain profile for unknown countries."""
        # This could be enhanced with external APIs for real data
        return {
            "terrain_type": "varied",
            "max_elevation": 1000,  # Conservative estimate
            "avg_elevation": 400,
            "coastal_length_km": 0,  # Unknown
            "major_rivers": [],
            "terrain_zones": {
                "lowlands": {"percentage": 60, "elevation_range": [0, 500], "difficulty": "low"},
                "hills": {"percentage": 30, "elevation_range": [500, 1000], "difficulty": "medium"},
                "highlands": {"percentage": 10, "elevation_range": [1000, 2000], "difficulty": "high"}
            },
            "climate_challenges": ["seasonal_weather"],
            "soil_types": ["mixed"],
            "seismic_activity": "unknown"
        }
    
    def _analyze_elevation_patterns(self) -> Dict[str, Any]:
        """Analyze elevation patterns and their impact on railway construction."""
        self.logger.info("📊 Analyzing elevation patterns...")
        
        # This would ideally use real elevation data from SRTM or similar
        # For now, we'll use the terrain profile data
        country_profile = self._get_country_terrain_profile()
        
        terrain_zones = country_profile.get('terrain_zones', {})
        max_elevation = country_profile.get('max_elevation', 1000)
        avg_elevation = country_profile.get('avg_elevation', 400)
        
        # Calculate elevation statistics
        elevation_variance = 0
        weighted_difficulty = 0
        
        for zone_name, zone_data in terrain_zones.items():
            percentage = zone_data.get('percentage', 0) / 100
            elevation_range = zone_data.get('elevation_range', [0, 500])
            difficulty = zone_data.get('difficulty', 'medium')
            
            # Calculate variance contribution
            zone_variance = (elevation_range[1] - elevation_range[0]) ** 2
            elevation_variance += zone_variance * percentage
            
            # Calculate difficulty contribution
            difficulty_scores = {'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}
            weighted_difficulty += difficulty_scores.get(difficulty, 2) * percentage
        
        return {
            'max_elevation': max_elevation,
            'avg_elevation': avg_elevation,
            'elevation_variance': elevation_variance,
            'terrain_diversity_score': len(terrain_zones),
            'weighted_difficulty': weighted_difficulty,
            'elevation_classification': self._classify_elevation_difficulty(max_elevation, weighted_difficulty),
            'gradient_challenges': self._assess_gradient_challenges(terrain_zones),
            'elevation_zones': terrain_zones
        }
    
    def _classify_elevation_difficulty(self, max_elevation: float, weighted_difficulty: float) -> str:
        """Classify overall elevation difficulty for railway construction."""
        if max_elevation > 3000 or weighted_difficulty > 3.0:
            return "Very High"
        elif max_elevation > 1500 or weighted_difficulty > 2.5:
            return "High"
        elif max_elevation > 500 or weighted_difficulty > 2.0:
            return "Medium"
        else:
            return "Low"
    
    def _assess_gradient_challenges(self, terrain_zones: Dict) -> Dict[str, Any]:
        """Assess gradient challenges for different terrain zones."""
        gradient_analysis = {}
        
        for zone_name, zone_data in terrain_zones.items():
            elevation_range = zone_data.get('elevation_range', [0, 500])
            difficulty = zone_data.get('difficulty', 'medium')
            
            # Estimate typical gradients for each zone
            if difficulty == 'low':
                typical_gradient = 1.5  # 1.5% grade
                max_gradient = 3.0
            elif difficulty == 'medium':
                typical_gradient = 2.5  # 2.5% grade
                max_gradient = 5.0
            elif difficulty == 'high':
                typical_gradient = 4.0  # 4.0% grade
                max_gradient = 8.0
            else:  # very_high
                typical_gradient = 6.0  # 6.0% grade
                max_gradient = 12.0
            
            gradient_analysis[zone_name] = {
                'typical_gradient_percent': typical_gradient,
                'max_gradient_percent': max_gradient,
                'tunnel_requirement': max_gradient > 6.0,
                'bridge_requirement': elevation_range[1] - elevation_range[0] > 200,
                'engineering_complexity': difficulty
            }
        
        return gradient_analysis
    
    def _analyze_construction_complexity(self) -> Dict[str, Any]:
        """Analyze construction complexity factors."""
        self.logger.info("🏗️ Analyzing construction complexity...")
        
        country_profile = self._get_country_terrain_profile()
        terrain_zones = country_profile.get('terrain_zones', {})
        climate_challenges = country_profile.get('climate_challenges', [])
        seismic_activity = country_profile.get('seismic_activity', 'unknown')
        
        # Calculate complexity scores
        terrain_complexity = self._calculate_terrain_complexity(terrain_zones)
        climate_complexity = self._calculate_climate_complexity(climate_challenges)
        seismic_complexity = self._calculate_seismic_complexity(seismic_activity)
        
        # Overall complexity score (1-10 scale)
        overall_complexity = (terrain_complexity + climate_complexity + seismic_complexity) / 3
        
        return {
            'terrain_complexity_score': terrain_complexity,
            'climate_complexity_score': climate_complexity,
            'seismic_complexity_score': seismic_complexity,
            'overall_complexity_score': overall_complexity,
            'construction_difficulty': self._classify_construction_difficulty(overall_complexity),
            'required_technologies': self._identify_required_technologies(terrain_zones, climate_challenges),
            'construction_timeline_factor': self._calculate_timeline_factor(overall_complexity),
            'specialized_equipment_needed': self._identify_specialized_equipment(terrain_zones)
        }
    
    def _calculate_terrain_complexity(self, terrain_zones: Dict) -> float:
        """Calculate terrain complexity score (1-10)."""
        complexity_score = 0
        difficulty_weights = {'low': 2, 'medium': 5, 'high': 8, 'very_high': 10}
        
        for zone_data in terrain_zones.values():
            percentage = zone_data.get('percentage', 0) / 100
            difficulty = zone_data.get('difficulty', 'medium')
            zone_score = difficulty_weights.get(difficulty, 5)
            complexity_score += zone_score * percentage
        
        return min(complexity_score, 10)
    
    def _calculate_climate_complexity(self, climate_challenges: List[str]) -> float:
        """Calculate climate complexity score (1-10)."""
        climate_weights = {
            'tropical_rainfall': 3,
            'monsoons': 4,
            'seasonal_flooding': 5,
            'humidity': 2,
            'sahel_conditions': 3,
            'altitude_variation': 4,
            'drought': 3,
            'high_altitude': 5,
            'equatorial_climate': 3,
            'lake_effects': 2,
            'swamps': 6,
            'seasonal_weather': 2
        }
        
        total_score = sum(climate_weights.get(challenge, 2) for challenge in climate_challenges)
        return min(total_score, 10)
    
    def _calculate_seismic_complexity(self, seismic_activity: str) -> float:
        """Calculate seismic complexity score (1-10)."""
        seismic_scores = {
            'low': 2,
            'moderate': 5,
            'high': 8,
            'very_high': 10,
            'unknown': 3
        }
        return seismic_scores.get(seismic_activity, 3)
    
    def _classify_construction_difficulty(self, complexity_score: float) -> str:
        """Classify overall construction difficulty."""
        if complexity_score >= 8:
            return "Very High"
        elif complexity_score >= 6:
            return "High" 
        elif complexity_score >= 4:
            return "Medium"
        else:
            return "Low"
    
    def _identify_required_technologies(self, terrain_zones: Dict, climate_challenges: List[str]) -> List[str]:
        """Identify required construction technologies."""
        technologies = set()
        
        # Terrain-based technologies
        for zone_data in terrain_zones.values():
            difficulty = zone_data.get('difficulty', 'medium')
            elevation_range = zone_data.get('elevation_range', [0, 500])
            
            if difficulty in ['high', 'very_high']:
                technologies.add('tunnel_boring_machines')
                technologies.add('heavy_earth_moving')
                
            if elevation_range[1] - elevation_range[0] > 200:
                technologies.add('bridge_construction')
                technologies.add('elevated_track_systems')
                
            if difficulty == 'very_high':
                technologies.add('rock_blasting')
                technologies.add('slope_stabilization')
        
        # Climate-based technologies
        if 'seasonal_flooding' in climate_challenges or 'swamps' in climate_challenges:
            technologies.add('drainage_systems')
            technologies.add('flood_resistant_design')
            
        if 'tropical_rainfall' in climate_challenges or 'monsoons' in climate_challenges:
            technologies.add('weather_resistant_materials')
            technologies.add('erosion_control')
            
        if 'high_altitude' in climate_challenges:
            technologies.add('altitude_construction_methods')
            
        return list(technologies)
    
    def _calculate_timeline_factor(self, complexity_score: float) -> float:
        """Calculate timeline multiplication factor based on complexity."""
        # Base timeline is multiplied by this factor
        if complexity_score >= 8:
            return 2.5  # Very complex: 2.5x longer
        elif complexity_score >= 6:
            return 2.0  # High complexity: 2x longer
        elif complexity_score >= 4:
            return 1.5  # Medium complexity: 1.5x longer
        else:
            return 1.0  # Low complexity: normal timeline
    
    def _identify_specialized_equipment(self, terrain_zones: Dict) -> List[str]:
        """Identify specialized equipment needed for construction."""
        equipment = set()
        
        for zone_data in terrain_zones.values():
            difficulty = zone_data.get('difficulty', 'medium')
            
            if difficulty == 'low':
                equipment.update(['standard_excavators', 'graders', 'compactors'])
            elif difficulty == 'medium':
                equipment.update(['heavy_excavators', 'rock_hammers', 'pile_drivers'])
            elif difficulty == 'high':
                equipment.update(['tunnel_boring_machines', 'rock_drills', 'crane_systems'])
            else:  # very_high
                equipment.update(['specialized_tunnel_equipment', 'mountain_construction_gear', 'helicopter_transport'])
        
        return list(equipment)
    
    def _identify_geographic_obstacles(self) -> Dict[str, Any]:
        """Identify major geographic obstacles affecting railway routes."""
        self.logger.info("🌊 Identifying geographic obstacles...")
        
        country_profile = self._get_country_terrain_profile()
        major_rivers = country_profile.get('major_rivers', [])
        coastal_length = country_profile.get('coastal_length_km', 0)
        terrain_zones = country_profile.get('terrain_zones', {})
        
        obstacles = {
            'water_bodies': {
                'major_rivers': major_rivers,
                'coastal_challenges': coastal_length > 0,
                'bridge_requirements': len(major_rivers),
                'marine_engineering_needed': coastal_length > 200
            },
            'elevation_obstacles': [],
            'climate_obstacles': country_profile.get('climate_challenges', []),
            'geological_challenges': self._assess_geological_challenges(country_profile)
        }
        
        # Identify elevation-based obstacles
        for zone_name, zone_data in terrain_zones.items():
            if zone_data.get('difficulty') in ['high', 'very_high']:
                obstacles['elevation_obstacles'].append({
                    'zone': zone_name,
                    'elevation_range': zone_data.get('elevation_range', [0, 500]),
                    'challenge_type': 'steep_terrain' if zone_data.get('difficulty') == 'high' else 'extreme_terrain'
                })
        
        return obstacles
    
    def _assess_geological_challenges(self, country_profile: Dict) -> Dict[str, Any]:
        """Assess geological challenges for construction."""
        soil_types = country_profile.get('soil_types', ['mixed'])
        seismic_activity = country_profile.get('seismic_activity', 'unknown')
        
        challenges = {
            'soil_stability': self._assess_soil_stability(soil_types),
            'seismic_considerations': seismic_activity,
            'foundation_requirements': self._determine_foundation_requirements(soil_types, seismic_activity),
            'ground_treatment_needed': 'clay' in soil_types or 'sandy' in soil_types
        }
        
        return challenges
    
    def _assess_soil_stability(self, soil_types: List[str]) -> str:
        """Assess soil stability for railway construction."""
        stability_scores = {
            'rock': 5,
            'volcanic': 4,
            'laterite': 3,
            'clay': 2,
            'sandy': 2,
            'alluvial': 3,
            'mixed': 3
        }
        
        avg_stability = sum(stability_scores.get(soil, 3) for soil in soil_types) / len(soil_types)
        
        if avg_stability >= 4:
            return "High"
        elif avg_stability >= 3:
            return "Medium"
        else:
            return "Low"
    
    def _determine_foundation_requirements(self, soil_types: List[str], seismic_activity: str) -> List[str]:
        """Determine foundation requirements based on soil and seismic conditions."""
        requirements = []
        
        if 'sandy' in soil_types or 'clay' in soil_types:
            requirements.append('deep_foundations')
            requirements.append('soil_stabilization')
        
        if seismic_activity in ['moderate', 'high', 'very_high']:
            requirements.append('seismic_isolation')
            requirements.append('flexible_connections')
        
        if 'alluvial' in soil_types:
            requirements.append('drainage_systems')
        
        return requirements
    
    def _generate_terrain_recommendations(self, country_terrain: Dict, elevation_analysis: Dict, 
                                        construction_analysis: Dict) -> Dict[str, Any]:
        """Generate terrain-based recommendations for railway construction."""
        self.logger.info("💡 Generating terrain recommendations...")
        
        recommendations = {
            'route_planning': [],
            'construction_methods': [],
            'technology_requirements': [],
            'timeline_considerations': [],
            'cost_optimization': []
        }
        
        # Route planning recommendations
        terrain_zones = country_terrain.get('terrain_zones', {})
        for zone_name, zone_data in terrain_zones.items():
            if zone_data.get('difficulty') == 'low':
                recommendations['route_planning'].append(
                    f"Prioritize routes through {zone_name} for easier construction"
                )
            elif zone_data.get('difficulty') in ['high', 'very_high']:
                recommendations['route_planning'].append(
                    f"Consider alternative routing to avoid {zone_name} or plan for extensive engineering"
                )
        
        # Construction method recommendations
        complexity_score = construction_analysis.get('overall_complexity_score', 5)
        if complexity_score >= 7:
            recommendations['construction_methods'].extend([
                'Employ phased construction approach',
                'Use specialized mountain railway techniques',
                'Consider rack railway systems for steep sections'
            ])
        elif complexity_score >= 5:
            recommendations['construction_methods'].extend([
                'Use standard heavy rail construction with modifications',
                'Plan for moderate tunneling and bridging'
            ])
        else:
            recommendations['construction_methods'].append('Standard railway construction methods suitable')
        
        # Technology requirements
        required_tech = construction_analysis.get('required_technologies', [])
        if required_tech:
            recommendations['technology_requirements'] = [
                f"Required: {', '.join(required_tech)}"
            ]
        
        # Timeline considerations
        timeline_factor = construction_analysis.get('construction_timeline_factor', 1.0)
        if timeline_factor > 2:
            recommendations['timeline_considerations'].append(
                f"Expect {timeline_factor:.1f}x longer construction time due to terrain complexity"
            )
        elif timeline_factor > 1.5:
            recommendations['timeline_considerations'].append(
                "Plan for extended construction timeline due to moderate terrain challenges"
            )
        
        # Cost optimization
        if complexity_score >= 6:
            recommendations['cost_optimization'].extend([
                'Consider public-private partnerships for high-cost sections',
                'Phase development to spread costs over time',
                'Investigate alternative technologies to reduce costs'
            ])
        
        return recommendations
    
    def _calculate_overall_difficulty(self, country_terrain: Dict, elevation_analysis: Dict) -> str:
        """Calculate overall terrain difficulty rating."""
        elevation_difficulty = elevation_analysis.get('elevation_classification', 'Medium')
        terrain_zones = country_terrain.get('terrain_zones', {})
        
        # Count difficult zones
        difficult_zones = sum(1 for zone in terrain_zones.values() 
                            if zone.get('difficulty') in ['high', 'very_high'])
        total_zones = len(terrain_zones)
        
        difficulty_ratio = difficult_zones / max(total_zones, 1)
        
        # Combine elevation and zone difficulty
        if elevation_difficulty == 'Very High' or difficulty_ratio > 0.5:
            return 'Very High'
        elif elevation_difficulty == 'High' or difficulty_ratio > 0.3:
            return 'High'
        elif elevation_difficulty == 'Medium' or difficulty_ratio > 0.1:
            return 'Medium'
        else:
            return 'Low'
    
    def _assess_cost_impact(self, construction_analysis: Dict) -> Dict[str, Any]:
        """Assess cost impact of terrain conditions."""
        complexity_score = construction_analysis.get('overall_complexity_score', 5)
        timeline_factor = construction_analysis.get('construction_timeline_factor', 1.0)
        
        # Base cost multipliers
        if complexity_score >= 8:
            cost_multiplier = 3.0  # 3x base cost
            risk_factor = 'Very High'
        elif complexity_score >= 6:
            cost_multiplier = 2.2  # 2.2x base cost
            risk_factor = 'High'
        elif complexity_score >= 4:
            cost_multiplier = 1.6  # 1.6x base cost
            risk_factor = 'Medium'
        else:
            cost_multiplier = 1.1  # 1.1x base cost
            risk_factor = 'Low'
        
        return {
            'terrain_cost_multiplier': cost_multiplier,
            'timeline_cost_impact': timeline_factor,
            'total_cost_multiplier': cost_multiplier * timeline_factor,
            'cost_risk_factor': risk_factor,
            'contingency_recommended': max(20, complexity_score * 5),  # % contingency
            'major_cost_drivers': self._identify_major_cost_drivers(construction_analysis)
        }
    
    def _identify_major_cost_drivers(self, construction_analysis: Dict) -> List[str]:
        """Identify the major cost drivers from terrain analysis."""
        cost_drivers = []
        
        required_tech = construction_analysis.get('required_technologies', [])
        if 'tunnel_boring_machines' in required_tech:
            cost_drivers.append('Extensive tunneling requirements')
        if 'bridge_construction' in required_tech:
            cost_drivers.append('Major bridge construction')
        if 'rock_blasting' in required_tech:
            cost_drivers.append('Rock excavation and blasting')
        if 'slope_stabilization' in required_tech:
            cost_drivers.append('Slope stabilization and retaining walls')
        
        complexity_score = construction_analysis.get('overall_complexity_score', 5)
        if complexity_score >= 7:
            cost_drivers.append('Specialized construction equipment')
            cost_drivers.append('Extended construction timeline')
        
        return cost_drivers