import logging
from typing import Dict, Any, List

class InputProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user input and country data"""
        self.logger.info("Processing input data")
        
        country_data = context['country_data']
        user_input = context['user_input']
        
        # Validate input data
        self._validate_input(country_data, user_input)
        
        # Process budget constraints
        budget = user_input.get('budget', 0)
        context['budget_constraints'] = self._calculate_budget_constraints(budget)
        
        # Process country specifics
        context['country_features'] = self._extract_country_features(country_data)
        
        # Process user priorities
        context['user_priorities'] = self._process_user_priorities(user_input)
        
        self.logger.info("Input processing completed")
        return context
    
    def _validate_input(self, country_data: Dict[str, Any], user_input: Dict[str, Any]):
        """Validate input data"""
        required_country_fields = ['name', 'population', 'area', 'existing_infrastructure']
        required_user_fields = ['budget', 'priority_areas']
        
        for field in required_country_fields:
            if field not in country_data:
                raise ValueError(f"Missing required country field: {field}")
        
        for field in required_user_fields:
            if field not in user_input:
                raise ValueError(f"Missing required user field: {field}")
    
    def _calculate_budget_constraints(self, budget: float) -> Dict[str, Any]:
        """Calculate budget constraints for different project aspects"""
        return {
            'total_budget': budget,
            'infrastructure_allocation': budget * 0.6,
            'station_allocation': budget * 0.25,
            'rolling_stock_allocation': budget * 0.15,
            'contingency': budget * 0.1
        }
    
    def _extract_country_features(self, country_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant country features for planning"""
        population = country_data['population']
        area = country_data['area']
        
        return {
            'population_density': population / area if area > 0 else 0,
            'urban_centers': country_data.get('cities', []),
            'existing_rail': country_data.get('existing_infrastructure', {}).get('rail', {}),
            'terrain_type': country_data.get('terrain', 'mixed'),
            'development_level': self._assess_development_level(country_data)
        }
    
    def _process_user_priorities(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process and normalize user priorities"""
        priorities = user_input.get('priority_areas', [])
        time_frame = user_input.get('time_frame', 'medium_term')
        
        return {
            'priority_areas': priorities,
            'time_frame': time_frame,
            'speed_focus': user_input.get('speed_focus', False),
            'cost_focus': user_input.get('cost_focus', True),
            'environmental_focus': user_input.get('environmental_focus', False)
        }
    
    def _assess_development_level(self, country_data: Dict[str, Any]) -> str:
        """Assess country development level based on infrastructure"""
        existing_rail = country_data.get('existing_infrastructure', {}).get('rail', {})
        rail_density = existing_rail.get('total_km', 0) / country_data['area'] if country_data['area'] > 0 else 0
        
        if rail_density > 0.1:
            return 'developed'
        elif rail_density > 0.01:
            return 'developing'
        else:
            return 'undeveloped'