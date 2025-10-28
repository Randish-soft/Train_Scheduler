import logging
import numpy as np
from typing import Dict, Any, List

class CostUtils:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_npv(self, cash_flows: List[float], discount_rate: float = 0.08) -> float:
        """Calculate Net Present Value of cash flows"""
        if not cash_flows:
            return 0.0
        
        npv = 0.0
        for t, cash_flow in enumerate(cash_flows):
            npv += cash_flow / ((1 + discount_rate) ** t)
        
        return npv
    
    def calculate_irr(self, cash_flows: List[float], max_iterations: int = 1000) -> float:
        """Calculate Internal Rate of Return using iterative method"""
        if not cash_flows:
            return 0.0
        
        # Simple IRR calculation (in production, use numpy's IRR function)
        guess = 0.1
        tolerance = 0.0001
        
        for _ in range(max_iterations):
            npv = self.calculate_npv(cash_flows, guess)
            
            if abs(npv) < tolerance:
                return guess
            
            # Simple adjustment (Newton-Raphson would be better)
            if npv > 0:
                guess += 0.01
            else:
                guess -= 0.01
        
        return guess
    
    def calculate_benefit_cost_ratio(self, benefits: List[float], 
                                   costs: List[float], 
                                   discount_rate: float = 0.08) -> float:
        """Calculate Benefit-Cost Ratio"""
        if not benefits or not costs:
            return 0.0
        
        # Ensure both lists have same length
        min_length = min(len(benefits), len(costs))
        benefits = benefits[:min_length]
        costs = costs[:min_length]
        
        npv_benefits = self.calculate_npv(benefits, discount_rate)
        npv_costs = self.calculate_npv(costs, discount_rate)
        
        if npv_costs == 0:
            return float('inf')
        
        return npv_benefits / npv_costs
    
    def estimate_construction_cost(self, route_length: float, 
                                 terrain_difficulty: float,
                                 infrastructure_requirements: Dict[str, Any]) -> Dict[str, float]:
        """Estimate construction cost for a route"""
        # Base cost per km (USD)
        base_cost_per_km = 20000000
        
        # Terrain difficulty multiplier
        terrain_multiplier = 1.0 + (terrain_difficulty * 0.8)
        
        # Infrastructure requirements adjustments
        bridge_multiplier = 1.0 + (infrastructure_requirements.get('bridges_count', 0) * 0.1)
        tunnel_multiplier = 1.0 + (infrastructure_requirements.get('tunnels_km', 0) * 0.5)
        station_multiplier = 1.0 + (infrastructure_requirements.get('stations_count', 0) * 0.05)
        
        # Total base cost
        base_cost = route_length * base_cost_per_km * terrain_multiplier
        
        # Apply infrastructure multipliers
        total_cost = base_cost * bridge_multiplier * tunnel_multiplier * station_multiplier
        
        # Cost breakdown
        breakdown = {
            'earthworks': total_cost * 0.25,
            'track_work': total_cost * 0.20,
            'bridges_viaducts': total_cost * 0.15,
            'tunnels': total_cost * 0.10,
            'stations': total_cost * 0.10,
            'electrification': total_cost * 0.08,
            'signaling_communications': total_cost * 0.07,
            'utilities_drainage': total_cost * 0.05
        }
        
        return {
            'total_estimated_cost': total_cost,
            'cost_per_km': total_cost / route_length if route_length > 0 else 0,
            'cost_breakdown': breakdown,
            'contingency': total_cost * 0.15,
            'total_with_contingency': total_cost * 1.15
        }
    
    def calculate_operational_cost(self, route_length: float,
                                 daily_trips: int,
                                 train_specifications: Dict[str, Any],
                                 staffing_level: str = 'standard') -> Dict[str, float]:
        """Calculate annual operational cost for a route"""
        # Energy costs
        energy_consumption_kwh_km = train_specifications.get('energy_consumption_kwh_km', 15.0)
        energy_cost_per_kwh = 0.12  # USD
        
        annual_energy_km = route_length * daily_trips * 365
        annual_energy_cost = annual_energy_km * energy_consumption_kwh_km * energy_cost_per_kwh
        
        # Maintenance costs
        maintenance_cost_per_km = train_specifications.get('maintenance_cost_per_km', 2.5)
        annual_maintenance_cost = annual_energy_km * maintenance_cost_per_km
        
        # Staff costs
        staff_costs = self._calculate_staff_costs(daily_trips, staffing_level)
        
        # Infrastructure maintenance
        infrastructure_maintenance = route_length * 10000  # $10,000 per km annually
        
        total_annual_cost = (
            annual_energy_cost + 
            annual_maintenance_cost + 
            staff_costs + 
            infrastructure_maintenance
        )
        
        return {
            'total_annual_operational_cost': total_annual_cost,
            'energy_costs': annual_energy_cost,
            'maintenance_costs': annual_maintenance_cost,
            'staff_costs': staff_costs,
            'infrastructure_maintenance': infrastructure_maintenance,
            'cost_per_passenger_km': self._calculate_cost_per_passenger_km(total_annual_cost, annual_energy_km)
        }
    
    def _calculate_staff_costs(self, daily_trips: int, staffing_level: str) -> float:
        """Calculate staff costs based on operations"""
        staff_levels = {
            'premium': 1.2,   # 20% more staff
            'standard': 1.0,   # Standard staffing
            'basic': 0.8      # 20% less staff
        }
        
        multiplier = staff_levels.get(staffing_level, 1.0)
        
        # Base staff cost calculation
        base_staff_per_train = 3  # Driver, conductor, maintenance
        annual_salary_per_staff = 50000  # USD
        
        required_staff = base_staff_per_train * (daily_trips / 10) * multiplier
        return required_staff * annual_salary_per_staff
    
    def _calculate_cost_per_passenger_km(self, total_annual_cost: float, 
                                       annual_passenger_km: float) -> float:
        """Calculate cost per passenger-kilometer"""
        if annual_passenger_km <= 0:
            return 0.0
        
        # Estimate passenger-km based on typical load factors
        estimated_passenger_km = annual_passenger_km * 100  # Assume 100 passengers per train
        
        return total_annual_cost / estimated_passenger_km
    
    def calculate_lifecycle_cost(self, capital_cost: float,
                               operational_cost_annual: float,
                               lifespan_years: int = 30,
                               discount_rate: float = 0.08) -> Dict[str, float]:
        """Calculate lifecycle cost over project lifespan"""
        # Capital cost (year 0)
        lifecycle_costs = [capital_cost]
        
        # Operational costs (years 1 through lifespan)
        for year in range(1, lifespan_years + 1):
            lifecycle_costs.append(-operational_cost_annual)  # Negative as cost
        
        # Calculate NPV
        npv = self.calculate_npv(lifecycle_costs, discount_rate)
        
        # Calculate equivalent annual cost
        annuity_factor = (1 - (1 + discount_rate) ** -lifespan_years) / discount_rate
        equivalent_annual_cost = npv / annuity_factor if annuity_factor > 0 else 0
        
        return {
            'net_present_value': npv,
            'equivalent_annual_cost': equivalent_annual_cost,
            'total_lifecycle_cost': capital_cost + (operational_cost_annual * lifespan_years),
            'discounted_lifecycle_cost': abs(npv),
            'break_even_year': self._calculate_break_even_year(lifecycle_costs)
        }
    
    def _calculate_break_even_year(self, cash_flows: List[float]) -> int:
        """Calculate break-even year for investment"""
        cumulative = 0.0
        for year, cash_flow in enumerate(cash_flows):
            cumulative += cash_flow
            if cumulative >= 0:
                return year
        return len(cash_flows)  # Never breaks even
    
    def optimize_cost_allocation(self, total_budget: float,
                               cost_components: Dict[str, float],
                               priorities: Dict[str, float]) -> Dict[str, float]:
        """Optimize cost allocation across components based on priorities"""
        total_required = sum(cost_components.values())
        
        if total_required <= total_budget:
            # Budget sufficient, return full allocation
            return cost_components
        
        # Calculate priority weights
        total_priority = sum(priorities.values())
        if total_priority == 0:
            # Equal allocation if no priorities specified
            equal_share = total_budget / len(cost_components)
            return {component: equal_share for component in cost_components.keys()}
        
        # Allocate based on priorities
        allocation = {}
        for component, cost in cost_components.items():
            priority = priorities.get(component, 1.0)
            allocation[component] = (priority / total_priority) * total_budget
        
        return allocation
    
    def calculate_sensitivity_analysis(self, base_case: Dict[str, float],
                                     variables: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Perform sensitivity analysis on cost variables"""
        results = {}
        
        for var_name, var_range in variables.items():
            base_value = base_case.get(var_name, 0)
            low_value = var_range.get('low', base_value * 0.8)
            high_value = var_range.get('high', base_value * 1.2)
            
            # Calculate impact on total cost
            base_total = sum(base_case.values())
            
            low_total = base_total - base_value + low_value
            high_total = base_total - base_value + high_value
            
            results[var_name] = {
                'base_value': base_value,
                'low_value': low_value,
                'high_value': high_value,
                'impact_low': low_total - base