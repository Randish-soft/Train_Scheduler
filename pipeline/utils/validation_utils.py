import logging
import re
from typing import Dict, Any, List, Tuple

class ValidationUtils:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_country_data(self, country_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate country data structure and values"""
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ['name', 'population', 'area', 'existing_infrastructure']
        for field in required_fields:
            if field not in country_data:
                errors.append(f"Missing required field: {field}")
        
        # Population validation
        population = country_data.get('population', 0)
        if population <= 0:
            errors.append("Population must be positive")
        elif population > 2000000000:  # 2 billion
            warnings.append("Population value seems unusually high")
        
        # Area validation
        area = country_data.get('area', 0)
        if area <= 0:
            errors.append("Area must be positive")
        elif area > 20000000:  # 20 million sq km
            warnings.append("Area value seems unusually large")
        
        # Population density check
        if population > 0 and area > 0:
            density = population / area
            if density > 10000:  # 10,000 people per sq km
                warnings.append("Extremely high population density detected")
            elif density < 1:
                warnings.append("Very low population density detected")
        
        # Existing infrastructure validation
        infrastructure = country_data.get('existing_infrastructure', {})
        if not isinstance(infrastructure, dict):
            errors.append("Existing infrastructure must be a dictionary")
        else:
            rail_infrastructure = infrastructure.get('rail', {})
            if not isinstance(rail_infrastructure, dict):
                errors.append("Rail infrastructure must be a dictionary")
        
        return len(errors) == 0, errors + warnings
    
    def validate_user_input(self, user_input: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate user input parameters"""
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ['budget', 'priority_areas']
        for field in required_fields:
            if field not in user_input:
                errors.append(f"Missing required user input field: {field}")
        
        # Budget validation
        budget = user_input.get('budget', 0)
        if budget <= 0:
            errors.append("Budget must be positive")
        elif budget < 1000000:  # $1M
            warnings.append("Budget seems very low for rail project")
        elif budget > 100000000000:  # $100B
            warnings.append("Budget seems unusually high")
        
        # Priority areas validation
        priority_areas = user_input.get('priority_areas', [])
        if not isinstance(priority_areas, list):
            errors.append("Priority areas must be a list")
        elif len(priority_areas) == 0:
            warnings.append("No priority areas specified")
        
        # Time frame validation
        time_frame = user_input.get('time_frame', 'medium_term')
        valid_time_frames = ['short_term', 'medium_term', 'long_term']
        if time_frame not in valid_time_frames:
            warnings.append(f"Time frame should be one of: {valid_time_frames}")
        
        return len(errors) == 0, errors + warnings
    
    def validate_route_data(self, route_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate route data structure"""
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ['name', 'alignment', 'details']
        for field in required_fields:
            if field not in route_data:
                errors.append(f"Missing required route field: {field}")
        
        # Alignment validation
        alignment = route_data.get('alignment', {})
        if not isinstance(alignment, dict):
            errors.append("Alignment must be a dictionary")
        else:
            distance = alignment.get('total_distance_km', 0)
            if distance <= 0:
                errors.append("Route distance must be positive")
            elif distance > 5000:  # 5000 km
                warnings.append("Route distance seems unusually long")
        
        # Details validation
        details = route_data.get('details', {})
        if not isinstance(details, dict):
            errors.append("Route details must be a dictionary")
        else:
            max_speed = details.get('max_design_speed_kmh', 0)
            if max_speed <= 0:
                errors.append("Maximum design speed must be positive")
            elif max_speed > 500:  # 500 km/h
                warnings.append("Maximum design speed seems unusually high")
        
        # Stations validation
        stations = route_data.get('stations', [])
        if not isinstance(stations, list):
            errors.append("Stations must be a list")
        elif len(stations) < 2:
            errors.append("Route must have at least 2 stations")
        
        return len(errors) == 0, errors + warnings
    
    def validate_cost_data(self, cost_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate cost data structure and values"""
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ['total_estimated_cost', 'breakdown']
        for field in required_fields:
            if field not in cost_data:
                errors.append(f"Missing required cost field: {field}")
        
        # Total cost validation
        total_cost = cost_data.get('total_estimated_cost', 0)
        if total_cost <= 0:
            errors.append("Total cost must be positive")
        elif total_cost > 1000000000000:  # $1T
            warnings.append("Total cost seems unusually high")
        
        # Breakdown validation
        breakdown = cost_data.get('breakdown', {})
        if not isinstance(breakdown, dict):
            errors.append("Cost breakdown must be a dictionary")
        else:
            breakdown_total = sum(breakdown.values())
            if abs(breakdown_total - total_cost) > total_cost * 0.01:  # 1% tolerance
                warnings.append(f"Cost breakdown doesn't match total cost (difference: {breakdown_total - total_cost})")
        
        return len(errors) == 0, errors + warnings
    
    def validate_timetable_data(self, timetable_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate timetable data structure"""
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ['route_name', 'stations', 'daily_schedule']
        for field in required_fields:
            if field not in timetable_data:
                errors.append(f"Missing required timetable field: {field}")
        
        # Stations validation
        stations = timetable_data.get('stations', [])
        if not isinstance(stations, list):
            errors.append("Stations must be a list")
        elif len(stations) < 2:
            errors.append("Timetable must have at least 2 stations")
        
        # Daily schedule validation
        daily_schedule = timetable_data.get('daily_schedule', {})
        if not isinstance(daily_schedule, dict):
            errors.append("Daily schedule must be a dictionary")
        else:
            departures = daily_schedule.get('train_departures', [])
            if not isinstance(departures, list):
                errors.append("Train departures must be a list")
            elif len(departures) == 0:
                warnings.append("No train departures scheduled")
        
        # Station times validation
        station_times = timetable_data.get('station_times', [])
        if not isinstance(station_times, list):
            errors.append("Station times must be a list")
        elif len(station_times) != len(stations):
            warnings.append("Station times count doesn't match stations count")
        
        return len(errors) == 0, errors + warnings
    
    def validate_coordinates(self, coordinates: List[float]) -> Tuple[bool, List[str]]:
        """Validate geographic coordinates"""
        errors = []
        
        if not isinstance(coordinates, list) or len(coordinates) != 2:
            errors.append("Coordinates must be a list of two numbers")
            return False, errors
        
        latitude, longitude = coordinates
        
        # Latitude validation (-90 to 90)
        if not (-90 <= latitude <= 90):
            errors.append(f"Latitude {latitude} is out of valid range (-90 to 90)")
        
        # Longitude validation (-180 to 180)
        if not (-180 <= longitude <= 180):
            errors.append(f"Longitude {longitude} is out of valid range (-180 to 180)")
        
        return len(errors) == 0, errors
    
    def validate_email_format(self, email: str) -> bool:
        """Validate email address format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def validate_percentage(self, value: float, field_name: str = "value") -> Tuple[bool, List[str]]:
        """Validate percentage values (0-100)"""
        errors = []
        
        if not isinstance(value, (int, float)):
            errors.append(f"{field_name} must be a number")
        elif value < 0:
            errors.append(f"{field_name} cannot be negative")
        elif value > 100:
            errors.append(f"{field_name} cannot exceed 100%")
        
        return len(errors) == 0, errors
    
    def validate_positive_number(self, value: float, field_name: str = "value") -> Tuple[bool, List[str]]:
        """Validate positive numbers"""
        errors = []
        
        if not isinstance(value, (int, float)):
            errors.append(f"{field_name} must be a number")
        elif value <= 0:
            errors.append(f"{field_name} must be positive")
        
        return len(errors) == 0, errors
    
    def validate_string_length(self, text: str, min_length: int = 1, 
                             max_length: int = 255, field_name: str = "text") -> Tuple[bool, List[str]]:
        """Validate string length constraints"""
        errors = []
        
        if not isinstance(text, str):
            errors.append(f"{field_name} must be a string")
        elif len(text) < min_length:
            errors.append(f"{field_name} must be at least {min_length} characters long")
        elif len(text) > max_length:
            errors.append(f"{field_name} cannot exceed {max_length} characters")
        
        return len(errors) == 0, errors
    
    def validate_list_contents(self, items: List, expected_type: type, 
                             field_name: str = "list") -> Tuple[bool, List[str]]:
        """Validate that all items in a list are of expected type"""
        errors = []
        
        if not isinstance(items, list):
            errors.append(f"{field_name} must be a list")
            return False, errors
        
        for i, item in enumerate(items):
            if not isinstance(item, expected_type):
                errors.append(f"Item {i} in {field_name} must be of type {expected_type.__name__}")
        
        return len(errors) == 0, errors
    
    def generate_validation_report(self, validation_results: Dict[str, Tuple[bool, List[str]]]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_errors = 0
        total_warnings = 0
        component_reports = {}
        
        for component, (is_valid, messages) in validation_results.items():
            errors = [msg for msg in messages if 'error' in msg.lower() or 'must' in msg.lower() or 'cannot' in msg.lower()]
            warnings = [msg for msg in messages if msg not in errors]
            
            component_reports[component] = {
                'is_valid': is_valid,
                'errors': errors,
                'warnings': warnings,
                'error_count': len(errors),
                'warning_count': len(warnings)
            }
            
            total_errors += len(errors)
            total_warnings += len(warnings)
        
        overall_valid = total_errors == 0
        
        return {
            'overall_valid': overall_valid,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'component_reports': component_reports,
            'recommendations': self._generate_validation_recommendations(overall_valid, total_errors, total_warnings)
        }
    
    def _generate_validation_recommendations(self, overall_valid: bool, 
                                           total_errors: int, 
                                           total_warnings: int) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if not overall_valid:
            recommendations.append("Address all validation errors before proceeding")
        
        if total_errors > 0:
            recommendations.append(f"Fix {total_errors} validation error(s)")
        
        if total_warnings > 0:
            recommendations.append(f"Review {total_warnings} validation warning(s)")
        
        if overall_valid and total_warnings == 0:
            recommendations.append("All data validation checks passed successfully")
        
        return recommendations