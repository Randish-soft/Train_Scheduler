"""
Similar Country Referencing Module
==================================

Finds countries with existing railway systems that have similar characteristics
to the target country for benchmarking:
- Geographic and terrain similarities
- Economic development level
- Population density patterns
- Climate and construction challenges
- Successful railway projects for cost/timeline reference

Author: Miguel Ibrahim E
"""

import logging
import math
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class RailwayQuality(Enum):
    """Railway system quality levels."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    MODERATE = "Moderate"
    DEVELOPING = "Developing"
    POOR = "Poor"


@dataclass
class CountryProfile:
    """Comprehensive country profile for comparison."""
    name: str
    area_km2: float
    population: int
    gdp_per_capita: float
    terrain_type: str
    climate_type: str
    avg_elevation: float
    max_elevation: float
    railway_network_km: float
    railway_quality: RailwayQuality
    electrification_percent: float
    major_railway_projects: List[str]
    construction_costs_per_km: float  # USD per km
    construction_timeline_years: float
    similar_characteristics: List[str]


@dataclass
class SimilarityScore:
    """Similarity scoring between countries."""
    country_name: str
    overall_score: float
    geographic_score: float
    economic_score: float
    terrain_score: float
    climate_score: float
    development_score: float
    recommendation_confidence: str


class CountryReferencer:
    """Main class for finding similar countries with railway systems."""
    
    def __init__(self, target_country: str):
        self.target_country = target_country
        self.logger = logging.getLogger(__name__)
        self.reference_countries = self._initialize_reference_database()
        
    def find_similar(self) -> Dict[str, Any]:
        """
        Find countries with railway systems similar to target country.
        
        Returns:
            Dictionary containing similarity analysis and recommendations
        """
        self.logger.info(f"🌍 Finding similar countries with railways for {self.target_country}...")
        
        # Get target country profile
        target_profile = self._get_target_country_profile()
        
        # Calculate similarity scores
        similarity_scores = self._calculate_similarity_scores(target_profile)
        
        # Get top similar countries
        top_similar = self._get_top_similar_countries(similarity_scores, limit=5)
        
        # Generate benchmarking data
        benchmarks = self._generate_benchmarking_data(top_similar)
        
        # Create recommendations
        recommendations = self._generate_recommendations(target_profile, top_similar)
        
        results = {
            'target_country_profile': self._profile_to_dict(target_profile),
            'similar_countries': [self._score_to_dict(score) for score in top_similar],
            'benchmarking_data': benchmarks,
            'recommendations': recommendations,
            'analysis_summary': self._generate_analysis_summary(target_profile, top_similar)
        }
        
        self.logger.info("✅ Similar country analysis completed")
        return results
    
    def _initialize_reference_database(self) -> Dict[str, CountryProfile]:
        """Initialize database of countries with established railway systems."""
        return {
            "india": CountryProfile(
                name="India",
                area_km2=3287263,
                population=1380004385,
                gdp_per_capita=2100,
                terrain_type="diverse_plains_mountains",
                climate_type="tropical_subtropical",
                avg_elevation=160,
                max_elevation=8586,
                railway_network_km=67956,
                railway_quality=RailwayQuality.GOOD,
                electrification_percent=74,
                major_railway_projects=[
                    "Dedicated Freight Corridors",
                    "High Speed Rail (Mumbai-Ahmedabad)",
                    "Konkan Railway",
                    "Northeast Frontier Railway"
                ],
                construction_costs_per_km=3500000,  # $3.5M per km
                construction_timeline_years=4,
                similar_characteristics=["large_population", "diverse_terrain", "tropical_climate", "developing_economy"]
            ),
            
            "china": CountryProfile(
                name="China",
                area_km2=9596960,
                population=1439323776,
                gdp_per_capita=10500,
                terrain_type="diverse_plains_mountains_desert",
                climate_type="varied_continental",
                avg_elevation=1840,
                max_elevation=8849,
                railway_network_km=146300,
                railway_quality=RailwayQuality.EXCELLENT,
                electrification_percent=72,
                major_railway_projects=[
                    "Beijing-Shanghai High-Speed Railway",
                    "Qinghai-Tibet Railway",
                    "China-Laos Railway",
                    "Jakarta-Bandung HSR (Indonesian project)"
                ],
                construction_costs_per_km=25000000,  # $25M per km (HSR)
                construction_timeline_years=5,
                similar_characteristics=["large_scale", "mountainous_terrain", "rapid_development"]
            ),
            
            "brazil": CountryProfile(
                name="Brazil",
                area_km2=8515767,
                population=212559417,
                gdp_per_capita=8700,
                terrain_type="coastal_plains_highlands_rainforest",
                climate_type="tropical_subtropical",
                avg_elevation=320,
                max_elevation=2994,
                railway_network_km=29817,
                railway_quality=RailwayQuality.MODERATE,
                electrification_percent=15,
                major_railway_projects=[
                    "Carajás Railway",
                    "Vitória-Minas Railway",
                    "North-South Railway",
                    "São Paulo Metro expansion"
                ],
                construction_costs_per_km=8000000,  # $8M per km
                construction_timeline_years=6,
                similar_characteristics=["tropical_climate", "large_country", "developing_economy", "diverse_terrain"]
            ),
            
            "south_africa": CountryProfile(
                name="South Africa",
                area_km2=1221037,
                population=59308690,
                gdp_per_capita=6000,
                terrain_type="plateau_coastal_plains",
                climate_type="subtropical_arid",
                avg_elevation=1034,
                max_elevation=3408,
                railway_network_km=20192,
                railway_quality=RailwayQuality.GOOD,
                electrification_percent=45,
                major_railway_projects=[
                    "Gautrain Rapid Rail",
                    "Blue Train luxury service",
                    "Coal Line upgrades",
                    "Port connectivity projects"
                ],
                construction_costs_per_km=12000000,  # $12M per km
                construction_timeline_years=5,
                similar_characteristics=["african_continent", "mining_economy", "diverse_population", "plateau_terrain"]
            ),
            
            "vietnam": CountryProfile(
                name="Vietnam",
                area_km2=331210,
                population=97338579,
                gdp_per_capita=3500,
                terrain_type="coastal_plains_mountains",
                climate_type="tropical_monsoon",
                avg_elevation=398,
                max_elevation=3144,
                railway_network_km=2600,
                railway_quality=RailwayQuality.DEVELOPING,
                electrification_percent=25,
                major_railway_projects=[
                    "North-South Railway upgrade",
                    "Hanoi Metro",
                    "Ho Chi Minh City Metro",
                    "Hanoi-Lao Cai Railway"
                ],
                construction_costs_per_km=15000000,  # $15M per km
                construction_timeline_years=7,
                similar_characteristics=["tropical_climate", "monsoons", "mountainous", "developing_economy"]
            ),
            
            "thailand": CountryProfile(
                name="Thailand",
                area_km2=513120,
                population=69799978,
                gdp_per_capita=7800,
                terrain_type="central_plains_mountains",
                climate_type="tropical_monsoon",
                avg_elevation=287,
                max_elevation=2565,
                railway_network_km=4507,
                railway_quality=RailwayQuality.MODERATE,
                electrification_percent=8,
                major_railway_projects=[
                    "Bangkok Mass Transit System",
                    "Thailand-China High-Speed Rail",
                    "Southern Line Double Track",
                    "Airport Rail Link"
                ],
                construction_costs_per_km=18000000,  # $18M per km
                construction_timeline_years=5,
                similar_characteristics=["tropical_climate", "monsoons", "tourism_economy", "moderate_development"]
            ),
            
            "morocco": CountryProfile(
                name="Morocco",
                area_km2=446550,
                population=36910560,
                gdp_per_capita=3200,
                terrain_type="coastal_plains_mountains_desert",
                climate_type="mediterranean_arid",
                avg_elevation=909,
                max_elevation=4167,
                railway_network_km=2067,
                railway_quality=RailwayQuality.GOOD,
                electrification_percent=60,
                major_railway_projects=[
                    "LGV (High-Speed Rail Tangier-Casablanca)",
                    "Casablanca Tramway",
                    "Rabat-Salé Tramway",
                    "Atlantic Rail expansion"
                ],
                construction_costs_per_km=20000000,  # $20M per km
                construction_timeline_years=4,
                similar_characteristics=["african_continent", "developing_economy", "varied_terrain", "coastal"]
            ),
            
            "indonesia": CountryProfile(
                name="Indonesia",
                area_km2=1904569,
                population=273523615,
                gdp_per_capita=4100,
                terrain_type="islands_mountains_coastal",
                climate_type="tropical_equatorial",
                avg_elevation=367,
                max_elevation=4884,
                railway_network_km=8159,
                railway_quality=RailwayQuality.DEVELOPING,
                electrification_percent=20,
                major_railway_projects=[
                    "Jakarta-Bandung High-Speed Railway",
                    "Trans-Java Railway",
                    "Jakarta MRT",
                    "Sumatra Railway rehabilitation"
                ],
                construction_costs_per_km=22000000,  # $22M per km
                construction_timeline_years=6,
                similar_characteristics=["tropical_climate", "islands", "developing_economy", "high_population_density"]
            ),
            
            "egypt": CountryProfile(
                name="Egypt",
                area_km2=1001450,
                population=102334404,
                gdp_per_capita=3000,
                terrain_type="desert_nile_valley",
                climate_type="arid_desert",
                avg_elevation=321,
                max_elevation=2629,
                railway_network_km=5085,
                railway_quality=RailwayQuality.MODERATE,
                electrification_percent=10,
                major_railway_projects=[
                    "New Administrative Capital Monorail",
                    "Alexandria-Cairo High-Speed Rail",
                    "Upper Egypt Railway upgrades",
                    "Cairo Metro expansion"
                ],
                construction_costs_per_km=16000000,  # $16M per km
                construction_timeline_years=5,
                similar_characteristics=["african_continent", "arid_climate", "developing_economy", "linear_development"]
            ),
            
            "malaysia": CountryProfile(
                name="Malaysia",
                area_km2=329847,
                population=32365999,
                gdp_per_capita=11200,
                terrain_type="coastal_plains_mountains",
                climate_type="tropical_rainforest",
                avg_elevation=538,
                max_elevation=4095,
                railway_network_km=1851,
                railway_quality=RailwayQuality.GOOD,
                electrification_percent=35,
                major_railway_projects=[
                    "East Coast Rail Link (ECRL)",
                    "Kuala Lumpur MRT",
                    "KL-Singapore HSR (cancelled)",
                    "Sabah State Railway upgrade"
                ],
                construction_costs_per_km=30000000,  # $30M per km
                construction_timeline_years=7,
                similar_characteristics=["tropical_rainforest", "monsoons", "emerging_economy", "palm_oil"]
            )
        }
    
    def _get_target_country_profile(self) -> CountryProfile:
        """Get or create profile for target country."""
        # Check if target country is in our reference database
        target_key = self.target_country.lower()
        if target_key in self.reference_countries:
            profile = self.reference_countries[target_key]
            # Mark as target (no existing railway)
            profile.railway_network_km = 0
            profile.railway_quality = RailwayQuality.POOR
            return profile
        
        # Create basic profile for unknown countries
        basic_profiles = {
            "ghana": CountryProfile(
                name="Ghana",
                area_km2=238533,
                population=31072940,
                gdp_per_capita=2200,
                terrain_type="coastal_plains_hills",
                climate_type="tropical",
                avg_elevation=190,
                max_elevation=885,
                railway_network_km=0,  # Target country
                railway_quality=RailwayQuality.POOR,
                electrification_percent=0,
                major_railway_projects=[],
                construction_costs_per_km=0,
                construction_timeline_years=0,
                similar_characteristics=["west_africa", "tropical_climate", "developing_economy", "coastal"]
            ),
            "nigeria": CountryProfile(
                name="Nigeria",
                area_km2=923768,
                population=206139589,
                gdp_per_capita=2100,
                terrain_type="coastal_plains_plateau",
                climate_type="tropical_arid",
                avg_elevation=380,
                max_elevation=2419,
                railway_network_km=3505,  # Limited existing
                railway_quality=RailwayQuality.POOR,
                electrification_percent=5,
                major_railway_projects=["Lagos-Kano Railway", "Abuja Light Rail"],
                construction_costs_per_km=0,
                construction_timeline_years=0,
                similar_characteristics=["west_africa", "large_population", "oil_economy", "diverse_terrain"]
            ),
            "kenya": CountryProfile(
                name="Kenya",
                area_km2=580367,
                population=53771296,
                gdp_per_capita=1800,
                terrain_type="rift_valley_highlands",
                climate_type="tropical_arid",
                avg_elevation=762,
                max_elevation=5199,
                railway_network_km=3334,  # Some existing
                railway_quality=RailwayQuality.DEVELOPING,
                electrification_percent=0,
                major_railway_projects=["Standard Gauge Railway (Nairobi-Mombasa)"],
                construction_costs_per_km=0,
                construction_timeline_years=0,
                similar_characteristics=["east_africa", "highlands", "tourism", "agricultural"]
            ),
            "rwanda": CountryProfile(
                name="Rwanda",
                area_km2=26338,
                population=12952218,
                gdp_per_capita=800,
                terrain_type="mountainous_highlands",
                climate_type="temperate_tropical",
                avg_elevation=1598,
                max_elevation=4519,
                railway_network_km=0,
                railway_quality=RailwayQuality.POOR,
                electrification_percent=0,
                major_railway_projects=[],
                construction_costs_per_km=0,
                construction_timeline_years=0,
                similar_characteristics=["east_africa", "mountainous", "small_country", "high_density"]
            ),
            "uganda": CountryProfile(
                name="Uganda",
                area_km2=241038,
                population=45741007,
                gdp_per_capita=800,
                terrain_type="plateau_mountains",
                climate_type="tropical",
                avg_elevation=1100,
                max_elevation=5109,
                railway_network_km=1244,  # Limited existing
                railway_quality=RailwayQuality.POOR,
                electrification_percent=0,
                major_railway_projects=["Standard Gauge Railway extension"],
                construction_costs_per_km=0,
                construction_timeline_years=0,
                similar_characteristics=["east_africa", "landlocked", "agricultural", "plateau"]
            )
        }
        
        if target_key in basic_profiles:
            return basic_profiles[target_key]
        else:
            # Generic profile for unknown countries
            return CountryProfile(
                name=self.target_country,
                area_km2=500000,  # Estimate
                population=50000000,  # Estimate
                gdp_per_capita=2000,  # Estimate
                terrain_type="varied",
                climate_type="tropical",
                avg_elevation=500,
                max_elevation=2000,
                railway_network_km=0,
                railway_quality=RailwayQuality.POOR,
                electrification_percent=0,
                major_railway_projects=[],
                construction_costs_per_km=0,
                construction_timeline_years=0,
                similar_characteristics=["developing_economy"]
            )
    
    def _calculate_similarity_scores(self, target: CountryProfile) -> List[SimilarityScore]:
        """Calculate similarity scores between target and reference countries."""
        scores = []
        
        for ref_name, ref_country in self.reference_countries.items():
            if ref_country.railway_quality in [RailwayQuality.POOR, RailwayQuality.DEVELOPING]:
                continue  # Skip countries without good railway systems
            
            # Calculate individual similarity scores
            geographic_score = self._calculate_geographic_similarity(target, ref_country)
            economic_score = self._calculate_economic_similarity(target, ref_country)
            terrain_score = self._calculate_terrain_similarity(target, ref_country)
            climate_score = self._calculate_climate_similarity(target, ref_country)
            development_score = self._calculate_development_similarity(target, ref_country)
            
            # Calculate weighted overall score
            overall_score = (
                geographic_score * 0.2 +
                economic_score * 0.25 +
                terrain_score * 0.25 +
                climate_score * 0.15 +
                development_score * 0.15
            )
            
            # Determine confidence level
            confidence = self._determine_confidence_level(overall_score)
            
            score = SimilarityScore(
                country_name=ref_country.name,
                overall_score=overall_score,
                geographic_score=geographic_score,
                economic_score=economic_score,
                terrain_score=terrain_score,
                climate_score=climate_score,
                development_score=development_score,
                recommendation_confidence=confidence
            )
            
            scores.append(score)
        
        return sorted(scores, key=lambda x: x.overall_score, reverse=True)
    
    def _calculate_geographic_similarity(self, target: CountryProfile, ref: CountryProfile) -> float:
        """Calculate geographic similarity score (0-1)."""
        # Area similarity (logarithmic scale due to wide variation)
        area_ratio = min(target.area_km2, ref.area_km2) / max(target.area_km2, ref.area_km2)
        area_score = math.log10(area_ratio * 9 + 1)  # Normalize to 0-1
        
        # Population density similarity
        target_density = target.population / target.area_km2
        ref_density = ref.population / ref.area_km2
        density_ratio = min(target_density, ref_density) / max(target_density, ref_density)
        density_score = density_ratio
        
        return (area_score * 0.4 + density_score * 0.6)
    
    def _calculate_economic_similarity(self, target: CountryProfile, ref: CountryProfile) -> float:
        """Calculate economic similarity score (0-1)."""
        # GDP per capita similarity
        gdp_ratio = min(target.gdp_per_capita, ref.gdp_per_capita) / max(target.gdp_per_capita, ref.gdp_per_capita)
        
        # Economic development level similarity
        target_level = self._classify_economic_level(target.gdp_per_capita)
        ref_level = self._classify_economic_level(ref.gdp_per_capita)
        
        level_similarity = 1.0 if target_level == ref_level else 0.7 if abs(target_level - ref_level) == 1 else 0.4
        
        return (gdp_ratio * 0.6 + level_similarity * 0.4)
    
    def _classify_economic_level(self, gdp_per_capita: float) -> int:
        """Classify economic development level."""
        if gdp_per_capita < 1500:
            return 1  # Low income
        elif gdp_per_capita < 5000:
            return 2  # Lower middle income
        elif gdp_per_capita < 15000:
            return 3  # Upper middle income
        else:
            return 4  # High income
    
    def _calculate_terrain_similarity(self, target: CountryProfile, ref: CountryProfile) -> float:
        """Calculate terrain similarity score (0-1)."""
        # Elevation similarity
        elevation_diff = abs(target.avg_elevation - ref.avg_elevation)
        elevation_score = max(0, 1 - elevation_diff / 1000)  # Penalty per 1000m difference
        
        # Max elevation similarity
        max_elev_diff = abs(target.max_elevation - ref.max_elevation)
        max_elev_score = max(0, 1 - max_elev_diff / 3000)  # Penalty per 3000m difference
        
        # Terrain type similarity
        terrain_score = self._calculate_terrain_type_similarity(target.terrain_type, ref.terrain_type)
        
        return (elevation_score * 0.3 + max_elev_score * 0.3 + terrain_score * 0.4)
    
    def _calculate_terrain_type_similarity(self, target_terrain: str, ref_terrain: str) -> float:
        """Calculate terrain type similarity."""
        terrain_keywords = {
            "coastal": {"coastal", "plains"},
            "plains": {"plains", "coastal", "plateau"},
            "mountains": {"mountains", "highlands", "mountainous"},
            "highlands": {"highlands", "mountains", "plateau"},
            "plateau": {"plateau", "highlands", "plains"},
            "desert": {"desert", "arid"},
            "rainforest": {"rainforest", "forest"},
            "islands": {"islands", "coastal"}
        }
        
        target_words = set(target_terrain.lower().split('_'))
        ref_words = set(ref_terrain.lower().split('_'))
        
        # Find common terrain features
        common_features = 0
        total_features = len(target_words | ref_words)
        
        for target_word in target_words:
            for ref_word in ref_words:
                if target_word == ref_word:
                    common_features += 1
                else:
                    # Check for related terrain types
                    target_related = terrain_keywords.get(target_word, set())
                    if ref_word in target_related:
                        common_features += 0.7
        
        return min(1.0, common_features / max(1, total_features)) if total_features > 0 else 0.5
    
    def _calculate_climate_similarity(self, target: CountryProfile, ref: CountryProfile) -> float:
        """Calculate climate similarity score (0-1)."""
        climate_categories = {
            "tropical": ["tropical", "equatorial", "monsoon"],
            "subtropical": ["subtropical", "temperate"],
            "arid": ["arid", "desert", "semi_arid"],
            "temperate": ["temperate", "continental", "mediterranean"],
            "continental": ["continental", "temperate"]
        }
        
        target_climate = target.climate_type.lower()
        ref_climate = ref.climate_type.lower()
        
        if target_climate == ref_climate:
            return 1.0
        
        # Check for related climate types
        for climate_group in climate_categories.values():
            if target_climate in climate_group and ref_climate in climate_group:
                return 0.8
        
        # Check for partial matches
        target_words = set(target_climate.split('_'))
        ref_words = set(ref_climate.split('_'))
        common_words = target_words & ref_words
        
        if common_words:
            return 0.6
        
        return 0.3  # Different climate types
    
    def _calculate_development_similarity(self, target: CountryProfile, ref: CountryProfile) -> float:
        """Calculate development level similarity."""
        # Population similarity (logarithmic scale)
        pop_ratio = min(target.population, ref.population) / max(target.population, ref.population)
        pop_score = math.log10(pop_ratio * 9 + 1)
        
        # Characteristic similarity
        target_chars = set(target.similar_characteristics)
        ref_chars = set(ref.similar_characteristics)
        
        if target_chars and ref_chars:
            common_chars = target_chars & ref_chars
            char_score = len(common_chars) / len(target_chars | ref_chars)
        else:
            char_score = 0.5
        
        return (pop_score * 0.4 + char_score * 0.6)
    
    def _determine_confidence_level(self, overall_score: float) -> str:
        """Determine confidence level for recommendations."""
        if overall_score >= 0.8:
            return "Very High"
        elif overall_score >= 0.65:
            return "High"
        elif overall_score >= 0.5:
            return "Medium"
        elif overall_score >= 0.35:
            return "Low"
        else:
            return "Very Low"
    
    def _get_top_similar_countries(self, scores: List[SimilarityScore], limit: int = 5) -> List[SimilarityScore]:
        """Get top similar countries with good railway systems."""
        return scores[:limit]
    
    def _generate_benchmarking_data(self, similar_countries: List[SimilarityScore]) -> Dict[str, Any]:
        """Generate benchmarking data from similar countries."""
        if not similar_countries:
            return {}
        
        benchmarks = {
            'construction_costs': [],
            'timeline_estimates': [],
            'electrification_rates': [],
            'successful_projects': [],
            'lessons_learned': []
        }
        
        for similarity_score in similar_countries:
            country_name = similarity_score.country_name
            ref_profile = None
            
            # Find the reference profile
            for profile in self.reference_countries.values():
                if profile.name == country_name:
                    ref_profile = profile
                    break
            
            if ref_profile:
                benchmarks['construction_costs'].append({
                    'country': country_name,
                    'cost_per_km_usd': ref_profile.construction_costs_per_km,
                    'similarity_score': similarity_score.overall_score
                })
                
                benchmarks['timeline_estimates'].append({
                    'country': country_name,
                    'typical_timeline_years': ref_profile.construction_timeline_years,
                    'similarity_score': similarity_score.overall_score
                })
                
                benchmarks['electrification_rates'].append({
                    'country': country_name,
                    'electrification_percent': ref_profile.electrification_percent,
                    'network_km': ref_profile.railway_network_km
                })
                
                benchmarks['successful_projects'].extend([
                    {
                        'country': country_name,
                        'project': project,
                        'relevance_score': similarity_score.overall_score
                    }
                    for project in ref_profile.major_railway_projects
                ])
        
        # Calculate weighted averages
        if benchmarks['construction_costs']:
            total_weight = sum(item['similarity_score'] for item in benchmarks['construction_costs'])
            weighted_cost = sum(
                item['cost_per_km_usd'] * item['similarity_score'] 
                for item in benchmarks['construction_costs']
            ) / total_weight
            
            benchmarks['estimated_cost_per_km'] = weighted_cost
        
        if benchmarks['timeline_estimates']:
            total_weight = sum(item['similarity_score'] for item in benchmarks['timeline_estimates'])
            weighted_timeline = sum(
                item['typical_timeline_years'] * item['similarity_score']
                for item in benchmarks['timeline_estimates']
            ) / total_weight
            
            benchmarks['estimated_timeline_years'] = weighted_timeline
        
        return benchmarks
    
    def _generate_recommendations(self, target: CountryProfile, similar_countries: List[SimilarityScore]) -> Dict[str, Any]:
        """Generate recommendations based on similar countries."""
        recommendations = {
            'primary_reference_country': None,
            'construction_approach': [],
            'technology_recommendations': [],
            'financing_strategies': [],
            'implementation_timeline': [],
            'risk_mitigation': []
        }
        
        if not similar_countries:
            return recommendations
        
        # Primary reference country (most similar)
        primary_ref = similar_countries[0]
        recommendations['primary_reference_country'] = {
            'name': primary_ref.country_name,
            'similarity_score': primary_ref.overall_score,
            'confidence': primary_ref.recommendation_confidence,
            'why_similar': self._explain_similarity(target, primary_ref)
        }
        
        # Get reference profile for detailed recommendations
        primary_profile = None
        for profile in self.reference_countries.values():
            if profile.name == primary_ref.country_name:
                primary_profile = profile
                break
        
        if primary_profile:
            # Construction approach recommendations
            recommendations['construction_approach'] = [
                f"Follow {primary_profile.name}'s phased development approach",
                f"Target {primary_profile.electrification_percent}% electrification rate",
                f"Budget approximately ${primary_profile.construction_costs_per_km:,.0f} per km",
                f"Plan for {primary_profile.construction_timeline_years}-year construction timeline"
            ]
            
            # Technology recommendations
            if primary_profile.electrification_percent > 50:
                recommendations['technology_recommendations'].append("Prioritize electrification from the start")
            else:
                recommendations['technology_recommendations'].append("Consider diesel-electric hybrid approach initially")
            
            if primary_profile.railway_quality == RailwayQuality.EXCELLENT:
                recommendations['technology_recommendations'].extend([
                    "Invest in modern signaling systems",
                    "Consider high-speed rail for major corridors",
                    "Implement advanced maintenance systems"
                ])
            
            # Financing strategies based on similar countries
            recommendations['financing_strategies'] = [
                f"Study {primary_profile.name}'s railway financing model",
                "Consider public-private partnerships for major projects",
                "Explore development bank funding (World Bank, ADB, etc.)",
                "Phase investment to spread costs over time"
            ]
            
            # Implementation timeline
            recommendations['implementation_timeline'] = [
                "Phase 1 (Years 1-2): Major corridor construction",
                "Phase 2 (Years 3-4): Regional network expansion", 
                "Phase 3 (Years 5+): Urban integration and optimization"
            ]
            
            # Risk mitigation based on similar countries' experiences
            recommendations['risk_mitigation'] = [
                "Plan for cost overruns (typical 20-40% in similar countries)",
                "Account for terrain-specific challenges",
                "Establish local technical capacity building",
                "Ensure robust environmental impact assessment"
            ]
        
        return recommendations
    
    def _explain_similarity(self, target: CountryProfile, similar: SimilarityScore) -> List[str]:
        """Explain why countries are similar."""
        explanations = []
        
        if similar.economic_score > 0.7:
            explanations.append("Similar economic development level")
        
        if similar.terrain_score > 0.7:
            explanations.append("Comparable terrain and elevation")
        
        if similar.climate_score > 0.7:
            explanations.append("Similar climate conditions")
        
        if similar.geographic_score > 0.7:
            explanations.append("Similar size and population density")
        
        if similar.development_score > 0.7:
            explanations.append("Comparable development characteristics")
        
        return explanations
    
    def _generate_analysis_summary(self, target: CountryProfile, similar_countries: List[SimilarityScore]) -> Dict[str, Any]:
        """Generate summary of the similarity analysis."""
        if not similar_countries:
            return {'message': 'No similar countries with good railway systems found'}
        
        best_match = similar_countries[0]
        avg_similarity = sum(score.overall_score for score in similar_countries) / len(similar_countries)
        
        # Get cost and timeline estimates
        cost_estimates = []
        timeline_estimates = []
        
        for similarity_score in similar_countries:
            for profile in self.reference_countries.values():
                if profile.name == similarity_score.country_name:
                    cost_estimates.append(profile.construction_costs_per_km)
                    timeline_estimates.append(profile.construction_timeline_years)
                    break
        
        avg_cost = sum(cost_estimates) / len(cost_estimates) if cost_estimates else 0
        avg_timeline = sum(timeline_estimates) / len(timeline_estimates) if timeline_estimates else 0
        
        return {
            'best_match_country': best_match.country_name,
            'best_match_similarity': best_match.overall_score,
            'best_match_confidence': best_match.recommendation_confidence,
            'average_similarity_score': avg_similarity,
            'total_reference_countries': len(similar_countries),
            'estimated_cost_per_km': avg_cost,
            'estimated_timeline_years': avg_timeline,
            'recommendation_quality': self._assess_recommendation_quality(avg_similarity, len(similar_countries))
        }
    
    def _assess_recommendation_quality(self, avg_similarity: float, num_countries: int) -> str:
        """Assess quality of recommendations based on similarity scores."""
        if avg_similarity > 0.75 and num_countries >= 3:
            return "Excellent - High confidence recommendations"
        elif avg_similarity > 0.6 and num_countries >= 2:
            return "Good - Reliable recommendations with some adaptation needed"
        elif avg_similarity > 0.45:
            return "Moderate - General guidance available"
        else:
            return "Limited - Few similar countries found, custom approach recommended"
    
    def _profile_to_dict(self, profile: CountryProfile) -> Dict[str, Any]:
        """Convert CountryProfile to dictionary."""
        return {
            'name': profile.name,
            'area_km2': profile.area_km2,
            'population': profile.population,
            'gdp_per_capita': profile.gdp_per_capita,
            'terrain_type': profile.terrain_type,
            'climate_type': profile.climate_type,
            'avg_elevation': profile.avg_elevation,
            'max_elevation': profile.max_elevation,
            'railway_network_km': profile.railway_network_km,
            'railway_quality': profile.railway_quality.value,
            'electrification_percent': profile.electrification_percent,
            'major_railway_projects': profile.major_railway_projects,
            'construction_costs_per_km': profile.construction_costs_per_km,
            'construction_timeline_years': profile.construction_timeline_years,
            'similar_characteristics': profile.similar_characteristics
        }
    
    def _score_to_dict(self, score: SimilarityScore) -> Dict[str, Any]:
        """Convert SimilarityScore to dictionary."""
        # Get the reference country profile for additional details
        ref_profile = None
        for profile in self.reference_countries.values():
            if profile.name == score.country_name:
                ref_profile = profile
                break
        
        result = {
            'country_name': score.country_name,
            'overall_similarity': score.overall_score,
            'geographic_similarity': score.geographic_score,
            'economic_similarity': score.economic_score,
            'terrain_similarity': score.terrain_score,
            'climate_similarity': score.climate_score,
            'development_similarity': score.development_score,
            'recommendation_confidence': score.recommendation_confidence
        }
        
        if ref_profile:
            result.update({
                'railway_network_km': ref_profile.railway_network_km,
                'railway_quality': ref_profile.railway_quality.value,
                'electrification_percent': ref_profile.electrification_percent,
                'construction_cost_per_km': ref_profile.construction_costs_per_km,
                'typical_timeline_years': ref_profile.construction_timeline_years,
                'major_projects': ref_profile.major_railway_projects,
                'key_learnings': self._extract_key_learnings(ref_profile)
            })
        
        return result
    
    def _extract_key_learnings(self, profile: CountryProfile) -> List[str]:
        """Extract key learnings from a reference country's railway development."""
        learnings = []
        
        if profile.electrification_percent > 70:
            learnings.append("High electrification rate achieved - prioritize electric from start")
        
        if profile.railway_quality == RailwayQuality.EXCELLENT:
            learnings.append("World-class system - focus on quality and modern technology")
        
        if profile.construction_costs_per_km > 20000000:  # $20M+
            learnings.append("High construction costs - expect significant investment")
        
        if profile.construction_timeline_years > 6:
            learnings.append("Extended timelines - plan for long-term development")
        
        if "High-Speed" in str(profile.major_railway_projects):
            learnings.append("High-speed rail experience - consider for major corridors")
        
        if any("Metro" in project or "Light Rail" in project for project in profile.major_railway_projects):
            learnings.append("Urban rail experience - valuable for city integration")
        
        # Terrain-specific learnings
        if "mountains" in profile.terrain_type.lower():
            learnings.append("Mountain railway expertise - tunneling and bridge experience")
        
        if "coastal" in profile.terrain_type.lower():
            learnings.append("Coastal construction experience - marine engineering skills")
        
        # Climate-specific learnings
        if "tropical" in profile.climate_type.lower():
            learnings.append("Tropical climate experience - weather-resistant design")
        
        if "monsoon" in profile.climate_type.lower():
            learnings.append("Monsoon adaptation - flood-resistant infrastructure")
        
        return learnings[:5]  # Return top 5 learnings
    
    def get_detailed_country_comparison(self, target_country: str, reference_country: str) -> Dict[str, Any]:
        """Get detailed comparison between target and specific reference country."""
        target_profile = self._get_target_country_profile()
        
        ref_profile = None
        for profile in self.reference_countries.values():
            if profile.name.lower() == reference_country.lower():
                ref_profile = profile
                break
        
        if not ref_profile:
            return {'error': f'Reference country {reference_country} not found in database'}
        
        # Calculate detailed similarities
        similarities = {
            'geographic': self._calculate_geographic_similarity(target_profile, ref_profile),
            'economic': self._calculate_economic_similarity(target_profile, ref_profile),
            'terrain': self._calculate_terrain_similarity(target_profile, ref_profile),
            'climate': self._calculate_climate_similarity(target_profile, ref_profile),
            'development': self._calculate_development_similarity(target_profile, ref_profile)
        }
        
        # Overall similarity
        overall_similarity = sum(similarities.values()) / len(similarities)
        
        return {
            'target_country': target_profile.name,
            'reference_country': ref_profile.name,
            'overall_similarity': overall_similarity,
            'similarity_breakdown': similarities,
            'target_profile': self._profile_to_dict(target_profile),
            'reference_profile': self._profile_to_dict(ref_profile),
            'applicability_assessment': self._assess_applicability(target_profile, ref_profile, overall_similarity),
            'specific_recommendations': self._generate_specific_recommendations(target_profile, ref_profile)
        }
    
    def _assess_applicability(self, target: CountryProfile, ref: CountryProfile, similarity: float) -> Dict[str, Any]:
        """Assess how applicable the reference country's approach is."""
        if similarity > 0.8:
            applicability = "Very High"
            adaptation_needed = "Minimal"
        elif similarity > 0.65:
            applicability = "High" 
            adaptation_needed = "Minor"
        elif similarity > 0.5:
            applicability = "Medium"
            adaptation_needed = "Moderate"
        elif similarity > 0.35:
            applicability = "Low"
            adaptation_needed = "Significant"
        else:
            applicability = "Very Low"
            adaptation_needed = "Extensive"
        
        return {
            'applicability_level': applicability,
            'adaptation_needed': adaptation_needed,
            'cost_scaling_factor': self._calculate_cost_scaling_factor(target, ref),
            'timeline_adjustment': self._calculate_timeline_adjustment(target, ref),
            'key_differences': self._identify_key_differences(target, ref)
        }
    
    def _calculate_cost_scaling_factor(self, target: CountryProfile, ref: CountryProfile) -> float:
        """Calculate cost scaling factor based on economic differences."""
        # GDP per capita ratio affects costs
        gdp_ratio = target.gdp_per_capita / ref.gdp_per_capita
        
        # Terrain complexity affects costs
        terrain_factor = 1.0
        if "mountains" in target.terrain_type and "plains" in ref.terrain_type:
            terrain_factor = 1.5
        elif "plains" in target.terrain_type and "mountains" in ref.terrain_type:
            terrain_factor = 0.7
        
        return gdp_ratio * terrain_factor
    
    def _calculate_timeline_adjustment(self, target: CountryProfile, ref: CountryProfile) -> str:
        """Calculate timeline adjustment recommendation."""
        # Economic development affects timeline
        target_level = self._classify_economic_level(target.gdp_per_capita)
        ref_level = self._classify_economic_level(ref.gdp_per_capita)
        
        if target_level < ref_level:
            return "Add 1-2 years for capacity building"
        elif target_level > ref_level:
            return "Potential for faster implementation"
        else:
            return "Similar timeline expected"
    
    def _identify_key_differences(self, target: CountryProfile, ref: CountryProfile) -> List[str]:
        """Identify key differences requiring adaptation."""
        differences = []
        
        # Economic differences
        if abs(target.gdp_per_capita - ref.gdp_per_capita) > 2000:
            differences.append(f"Significant economic difference: ${target.gdp_per_capita} vs ${ref.gdp_per_capita} GDP per capita")
        
        # Terrain differences
        if target.terrain_type != ref.terrain_type:
            differences.append(f"Terrain difference: {target.terrain_type} vs {ref.terrain_type}")
        
        # Climate differences
        if target.climate_type != ref.climate_type:
            differences.append(f"Climate difference: {target.climate_type} vs {ref.climate_type}")
        
        # Size differences
        area_ratio = target.area_km2 / ref.area_km2
        if area_ratio > 2 or area_ratio < 0.5:
            differences.append(f"Significant size difference: {target.area_km2:,.0f} vs {ref.area_km2:,.0f} km²")
        
        return differences
    
    def _generate_specific_recommendations(self, target: CountryProfile, ref: CountryProfile) -> Dict[str, List[str]]:
        """Generate specific recommendations based on reference country."""
        recommendations = {
            'technology_transfer': [],
            'financing_approach': [],
            'implementation_strategy': [],
            'capacity_building': [],
            'risk_management': []
        }
        
        # Technology transfer recommendations
        recommendations['technology_transfer'].extend([
            f"Study {ref.name}'s railway technology standards",
            f"Consider technology partnerships with {ref.name}",
            f"Adopt {ref.name}'s proven rail gauge and systems"
        ])
        
        if ref.electrification_percent > 50:
            recommendations['technology_transfer'].append(f"Learn from {ref.name}'s electrification approach")
        
        # Financing approach
        recommendations['financing_approach'].extend([
            f"Study {ref.name}'s railway financing model",
            f"Engage with development banks that funded {ref.name}'s projects",
            "Consider similar public-private partnership structures"
        ])
        
        # Implementation strategy
        recommendations['implementation_strategy'].extend([
            f"Follow {ref.name}'s phased development approach",
            f"Prioritize corridors similar to {ref.name}'s initial network",
            "Establish similar regulatory and institutional frameworks"
        ])
        
        # Capacity building
        recommendations['capacity_building'].extend([
            f"Train technical staff in {ref.name}",
            f"Establish educational exchanges with {ref.name}'s railway institutions",
            "Develop local technical expertise using proven methods"
        ])
        
        # Risk management
        recommendations['risk_management'].extend([
            f"Learn from {ref.name}'s project management experiences",
            "Implement similar quality control and safety standards",
            "Adopt proven maintenance and operational practices"
        ])
        
        return recommendations