"""
cost_analysis.py - Cost analysis module for BCPC Pipeline

This module implements the cost breakdown functionality described in the BCPC paper,
calculating infrastructure costs, rolling stock costs, and operational expenses
for proposed rail corridors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
import json
from datetime import datetime
import math

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CostParameters:
    """Parameters for cost calculations based on EU 2019 averages"""
    # Track construction costs (per km)
    conventional_track_cost: float = 15_000_000  # €15M per km
    high_speed_track_cost: float = 25_000_000   # €25M per km
    
    # Infrastructure costs
    tunnel_cost_per_km: float = 50_000_000      # €50M per km
    bridge_cost_per_km: float = 30_000_000      # €30M per km
    station_cost: float = 10_000_000            # €10M per station
    depot_cost: float = 50_000_000              # €50M per depot
    
    # Electrification and signaling
    electrification_cost_per_km: float = 2_000_000  # €2M per km
    signaling_cost_per_km: float = 1_000_000        # €1M per km
    
    # Rolling stock costs
    conventional_trainset_cost: float = 5_000_000    # €5M per trainset
    high_speed_trainset_cost: float = 30_000_000     # €30M per trainset
    
    # Land acquisition (percentage of track cost)
    land_acquisition_factor: float = 0.15  # 15% of track cost
    
    # Engineering and planning (percentage of total)
    engineering_factor: float = 0.10  # 10% of infrastructure cost
    contingency_factor: float = 0.20  # 20% contingency
    
    # Operational costs (annual)
    maintenance_cost_per_km: float = 50_000      # €50k per km per year
    energy_cost_per_train_km: float = 15         # €15 per train-km
    staff_cost_per_train: float = 2_000_000      # €2M per train per year
    
    # Financial parameters
    discount_rate: float = 0.04  # 4% discount rate
    project_lifetime: int = 30   # 30 years
    construction_period: int = 5  # 5 years construction
    
    # Currency conversion
    eur_to_usd: float = 1.1  # EUR to USD conversion rate


@dataclass
class InfrastructureCost:
    """Detailed infrastructure cost breakdown"""
    track_construction: float = 0.0
    tunnels: float = 0.0
    bridges: float = 0.0
    stations: float = 0.0
    depots: float = 0.0
    electrification: float = 0.0
    signaling: float = 0.0
    land_acquisition: float = 0.0
    engineering: float = 0.0
    contingency: float = 0.0
    
    @property
    def total(self) -> float:
        """Calculate total infrastructure cost"""
        base_cost = (self.track_construction + self.tunnels + self.bridges + 
                    self.stations + self.depots + self.electrification + 
                    self.signaling + self.land_acquisition)
        return base_cost + self.engineering + self.contingency
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization"""
        return {
            'track_construction': self.track_construction,
            'tunnels': self.tunnels,
            'bridges': self.bridges,
            'stations': self.stations,
            'depots': self.depots,
            'electrification': self.electrification,
            'signaling': self.signaling,
            'land_acquisition': self.land_acquisition,
            'engineering': self.engineering,
            'contingency': self.contingency,
            'total': self.total
        }


@dataclass
class RollingStockCost:
    """Rolling stock cost breakdown"""
    trainsets_required: int = 0
    trainset_unit_cost: float = 0.0
    total_cost: float = 0.0
    spare_ratio: float = 0.15  # 15% spare trainsets
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'trainsets_required': self.trainsets_required,
            'trainset_unit_cost': self.trainset_unit_cost,
            'spare_ratio': self.spare_ratio,
            'total_cost': self.total_cost
        }


@dataclass
class OperationalCost:
    """Annual operational cost breakdown"""
    maintenance: float = 0.0
    energy: float = 0.0
    staff: float = 0.0
    administration: float = 0.0
    insurance: float = 0.0
    
    @property
    def total_annual(self) -> float:
        """Calculate total annual operational cost"""
        return (self.maintenance + self.energy + self.staff + 
                self.administration + self.insurance)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization"""
        return {
            'maintenance': self.maintenance,
            'energy': self.energy,
            'staff': self.staff,
            'administration': self.administration,
            'insurance': self.insurance,
            'total_annual': self.total_annual
        }


@dataclass
class CostAnalysisResult:
    """Complete cost analysis result"""
    route_id: str
    infrastructure: InfrastructureCost
    rolling_stock: RollingStockCost
    operational: OperationalCost
    total_capex: float = 0.0
    total_opex_annual: float = 0.0
    npv: float = 0.0  # Net Present Value
    cost_per_passenger_km: float = 0.0
    payback_period: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'route_id': self.route_id,
            'infrastructure': self.infrastructure.to_dict(),
            'rolling_stock': self.rolling_stock.to_dict(),
            'operational': self.operational.to_dict(),
            'total_capex': self.total_capex,
            'total_opex_annual': self.total_opex_annual,
            'npv': self.npv,
            'cost_per_passenger_km': self.cost_per_passenger_km,
            'payback_period': self.payback_period
        }


class CostAnalyzer:
    """
    Main cost analysis class for BCPC pipeline.
    Calculates comprehensive costs for rail infrastructure projects.
    """
    
    def __init__(self, parameters: Optional[CostParameters] = None):
        """
        Initialize cost analyzer with parameters.
        
        Args:
            parameters: Cost parameters (uses defaults if None)
        """
        self.params = parameters or CostParameters()
        
    def analyze_route_cost(self, 
                          route_data: Dict[str, Any],
                          demand_data: Dict[str, Any],
                          train_type: str = 'conventional',
                          station_spacing: float = 50.0) -> CostAnalysisResult:
        """
        Perform comprehensive cost analysis for a route.
        
        Args:
            route_data: Route information from route_mapping
            demand_data: Demand information from demand_analysis
            train_type: 'conventional' or 'high_speed'
            station_spacing: Average distance between stations (km)
            
        Returns:
            CostAnalysisResult with detailed breakdown
        """
        # Extract route information
        distance_km = route_data.get('distance_km', 0)
        tunnels_required = route_data.get('tunnels_required', 0)
        bridges_required = route_data.get('bridges_required', 0)
        existing_rail_percent = route_data.get('existing_rail_percent', 0)
        
        # Extract demand information
        daily_passengers = demand_data.get('total_daily_demand', 0)
        peak_hour_demand = demand_data.get('peak_hour_demand', 0)
        
        # Calculate infrastructure costs
        infrastructure = self._calculate_infrastructure_cost(
            distance_km=distance_km,
            tunnels_required=tunnels_required,
            bridges_required=bridges_required,
            existing_rail_percent=existing_rail_percent,
            train_type=train_type,
            station_spacing=station_spacing
        )
        
        # Calculate rolling stock requirements and costs
        rolling_stock = self._calculate_rolling_stock_cost(
            distance_km=distance_km,
            peak_hour_demand=peak_hour_demand,
            train_type=train_type
        )
        
        # Calculate operational costs
        operational = self._calculate_operational_cost(
            distance_km=distance_km,
            rolling_stock=rolling_stock,
            daily_passengers=daily_passengers,
            train_type=train_type
        )
        
        # Calculate financial metrics
        total_capex = infrastructure.total + rolling_stock.total_cost
        total_opex_annual = operational.total_annual
        
        # Calculate NPV
        npv = self._calculate_npv(
            capex=total_capex,
            annual_opex=total_opex_annual,
            annual_revenue=self._estimate_annual_revenue(daily_passengers, distance_km)
        )
        
        # Calculate cost per passenger-km
        annual_passenger_km = daily_passengers * distance_km * 365
        if annual_passenger_km > 0:
            cost_per_passenger_km = total_opex_annual / annual_passenger_km
        else:
            cost_per_passenger_km = 0
            
        # Create result
        result = CostAnalysisResult(
            route_id=f"{route_data.get('origin_city', 'A')}_{route_data.get('destination_city', 'B')}",
            infrastructure=infrastructure,
            rolling_stock=rolling_stock,
            operational=operational,
            total_capex=total_capex,
            total_opex_annual=total_opex_annual,
            npv=npv,
            cost_per_passenger_km=cost_per_passenger_km
        )
        
        # Calculate payback period if NPV is positive
        if npv > 0:
            result.payback_period = self._calculate_payback_period(
                capex=total_capex,
                annual_cashflow=self._estimate_annual_revenue(daily_passengers, distance_km) - total_opex_annual
            )
            
        return result
        
    def _calculate_infrastructure_cost(self,
                                     distance_km: float,
                                     tunnels_required: int,
                                     bridges_required: int,
                                     existing_rail_percent: float,
                                     train_type: str,
                                     station_spacing: float) -> InfrastructureCost:
        """Calculate detailed infrastructure costs."""
        cost = InfrastructureCost()
        
        # Track construction cost
        new_track_km = distance_km * (1 - existing_rail_percent / 100)
        if train_type == 'high_speed':
            cost.track_construction = new_track_km * self.params.high_speed_track_cost
        else:
            cost.track_construction = new_track_km * self.params.conventional_track_cost
            
        # Tunnel costs (assume average 2km per tunnel)
        cost.tunnels = tunnels_required * 2 * self.params.tunnel_cost_per_km
        
        # Bridge costs (assume average 0.5km per bridge)
        cost.bridges = bridges_required * 0.5 * self.params.bridge_cost_per_km
        
        # Station costs
        num_stations = max(2, int(distance_km / station_spacing) + 1)
        cost.stations = num_stations * self.params.station_cost
        
        # Depot costs (1 depot for routes under 200km, 2 for longer)
        num_depots = 1 if distance_km < 200 else 2
        cost.depots = num_depots * self.params.depot_cost
        
        # Electrification and signaling
        cost.electrification = distance_km * self.params.electrification_cost_per_km
        cost.signaling = distance_km * self.params.signaling_cost_per_km
        
        # Land acquisition
        cost.land_acquisition = cost.track_construction * self.params.land_acquisition_factor
        
        # Engineering and contingency
        base_cost = (cost.track_construction + cost.tunnels + cost.bridges + 
                    cost.stations + cost.depots + cost.electrification + 
                    cost.signaling + cost.land_acquisition)
        cost.engineering = base_cost * self.params.engineering_factor
        cost.contingency = base_cost * self.params.contingency_factor
        
        return cost
        
    def _calculate_rolling_stock_cost(self,
                                    distance_km: float,
                                    peak_hour_demand: int,
                                    train_type: str) -> RollingStockCost:
        """Calculate rolling stock requirements and costs."""
        cost = RollingStockCost()
        
        # Determine train capacity
        if train_type == 'high_speed':
            train_capacity = 500  # passengers per train
            avg_speed = 200  # km/h
            unit_cost = self.params.high_speed_trainset_cost
        else:
            train_capacity = 300  # passengers per train
            avg_speed = 100  # km/h
            unit_cost = self.params.conventional_trainset_cost
            
        # Calculate trains needed for peak hour
        trains_for_peak = math.ceil(peak_hour_demand / train_capacity)
        
        # Calculate round trip time (hours)
        round_trip_time = (2 * distance_km / avg_speed) + 0.5  # 30 min turnaround
        
        # Calculate total trainsets needed
        trainsets_needed = math.ceil(trains_for_peak * round_trip_time)
        
        # Add spare trainsets
        total_trainsets = math.ceil(trainsets_needed * (1 + cost.spare_ratio))
        
        cost.trainsets_required = total_trainsets
        cost.trainset_unit_cost = unit_cost
        cost.total_cost = total_trainsets * unit_cost
        
        return cost
        
    def _calculate_operational_cost(self,
                                  distance_km: float,
                                  rolling_stock: RollingStockCost,
                                  daily_passengers: int,
                                  train_type: str) -> OperationalCost:
        """Calculate annual operational costs."""
        cost = OperationalCost()
        
        # Maintenance cost
        cost.maintenance = distance_km * self.params.maintenance_cost_per_km
        
        # Energy cost (based on train-km)
        # Estimate daily train-km based on service frequency
        if train_type == 'high_speed':
            train_capacity = 500
            daily_frequency = 20  # trains per day each direction
        else:
            train_capacity = 300
            daily_frequency = 30  # trains per day each direction
            
        daily_train_km = daily_frequency * 2 * distance_km  # both directions
        annual_train_km = daily_train_km * 365
        cost.energy = annual_train_km * self.params.energy_cost_per_train_km
        
        # Staff cost
        cost.staff = rolling_stock.trainsets_required * self.params.staff_cost_per_train
        
        # Administration (10% of other costs)
        cost.administration = (cost.maintenance + cost.energy + cost.staff) * 0.10
        
        # Insurance (1% of rolling stock value)
        cost.insurance = rolling_stock.total_cost * 0.01
        
        return cost
        
    def _estimate_annual_revenue(self, daily_passengers: int, distance_km: float) -> float:
        """Estimate annual revenue from ticket sales."""
        # Average fare per km (varies by distance)
        if distance_km < 100:
            fare_per_km = 0.15  # €0.15 per km for short distances
        elif distance_km < 300:
            fare_per_km = 0.12  # €0.12 per km for medium distances
        else:
            fare_per_km = 0.10  # €0.10 per km for long distances
            
        # Calculate annual revenue
        annual_revenue = daily_passengers * distance_km * fare_per_km * 365
        
        # Add ancillary revenue (10% of ticket revenue)
        annual_revenue *= 1.10
        
        return annual_revenue
        
    def _calculate_npv(self, capex: float, annual_opex: float, annual_revenue: float) -> float:
        """Calculate Net Present Value of the project."""
        # Cash flows during construction (negative)
        construction_cashflows = [-capex / self.params.construction_period] * self.params.construction_period
        
        # Cash flows during operation
        annual_cashflow = annual_revenue - annual_opex
        operation_cashflows = [annual_cashflow] * (self.params.project_lifetime - self.params.construction_period)
        
        # Combine all cash flows
        all_cashflows = construction_cashflows + operation_cashflows
        
        # Calculate NPV
        npv = 0
        for t, cashflow in enumerate(all_cashflows):
            npv += cashflow / ((1 + self.params.discount_rate) ** t)
            
        return npv
        
    def _calculate_payback_period(self, capex: float, annual_cashflow: float) -> Optional[float]:
        """Calculate simple payback period in years."""
        if annual_cashflow <= 0:
            return None
        return capex / annual_cashflow
        
    def analyze_network_cost(self, 
                           routes_df: pd.DataFrame,
                           demand_df: pd.DataFrame,
                           train_type: str = 'conventional') -> pd.DataFrame:
        """
        Analyze costs for entire network of routes.
        
        Args:
            routes_df: DataFrame with route information
            demand_df: DataFrame with demand information
            train_type: Type of train service
            
        Returns:
            DataFrame with cost analysis for all routes
        """
        results = []
        
        for idx, route in routes_df.iterrows():
            # Get corresponding demand data
            demand_mask = ((demand_df['origin_city'] == route['origin_city']) & 
                          (demand_df['destination_city'] == route['destination_city']))
            
            if demand_mask.any():
                demand_data = demand_df[demand_mask].iloc[0].to_dict()
            else:
                # Use default demand if not found
                demand_data = {
                    'total_daily_demand': 1000,
                    'peak_hour_demand': 200
                }
                
            # Analyze route cost
            cost_result = self.analyze_route_cost(
                route_data=route.to_dict(),
                demand_data=demand_data,
                train_type=train_type
            )
            
            # Convert to flat dictionary for DataFrame
            result_dict = {
                'route_id': cost_result.route_id,
                'origin_city': route['origin_city'],
                'destination_city': route['destination_city'],
                'distance_km': route['distance_km'],
                'infrastructure_cost': cost_result.infrastructure.total,
                'rolling_stock_cost': cost_result.rolling_stock.total_cost,
                'total_capex': cost_result.total_capex,
                'annual_opex': cost_result.total_opex_annual,
                'npv': cost_result.npv,
                'cost_per_passenger_km': cost_result.cost_per_passenger_km,
                'payback_period': cost_result.payback_period
            }
            results.append(result_dict)
            
        return pd.DataFrame(results)
        
    def export_cost_report(self, 
                         cost_result: CostAnalysisResult,
                         output_path: str,
                         format: str = 'json') -> None:
        """
        Export detailed cost report.
        
        Args:
            cost_result: Cost analysis result
            output_path: Path to save report
            format: 'json' or 'csv'
        """
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(cost_result.to_dict(), f, indent=2)
        elif format == 'csv':
            # Flatten the nested structure for CSV
            flat_data = self._flatten_cost_result(cost_result)
            df = pd.DataFrame([flat_data])
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Cost report exported to {output_path}")
        
    def _flatten_cost_result(self, result: CostAnalysisResult) -> Dict[str, Any]:
        """Flatten nested cost result for CSV export."""
        flat = {
            'route_id': result.route_id,
            'total_capex': result.total_capex,
            'total_opex_annual': result.total_opex_annual,
            'npv': result.npv,
            'cost_per_passenger_km': result.cost_per_passenger_km,
            'payback_period': result.payback_period
        }
        
        # Add infrastructure costs
        for key, value in result.infrastructure.to_dict().items():
            flat[f'infrastructure_{key}'] = value
            
        # Add rolling stock costs
        for key, value in result.rolling_stock.to_dict().items():
            flat[f'rolling_stock_{key}'] = value
            
        # Add operational costs
        for key, value in result.operational.to_dict().items():
            flat[f'operational_{key}'] = value
            
        return flat
        
    def sensitivity_analysis(self,
                           base_result: CostAnalysisResult,
                           parameters: List[str],
                           variation_range: float = 0.2) -> pd.DataFrame:
        """
        Perform sensitivity analysis on cost parameters.
        
        Args:
            base_result: Base case cost result
            parameters: List of parameters to vary
            variation_range: +/- percentage to vary (0.2 = 20%)
            
        Returns:
            DataFrame with sensitivity analysis results
        """
        results = []
        
        # Store original parameters
        original_params = {}
        for param in parameters:
            if hasattr(self.params, param):
                original_params[param] = getattr(self.params, param)
                
        # Vary each parameter
        for param in parameters:
            if param not in original_params:
                logger.warning(f"Parameter {param} not found")
                continue
                
            original_value = original_params[param]
            
            # Test different values
            for multiplier in [1 - variation_range, 1, 1 + variation_range]:
                # Set parameter value
                setattr(self.params, param, original_value * multiplier)
                
                # Recalculate cost
                # Note: This is simplified - would need route and demand data
                new_npv = base_result.npv * multiplier  # Simplified
                
                results.append({
                    'parameter': param,
                    'multiplier': multiplier,
                    'parameter_value': original_value * multiplier,
                    'npv': new_npv,
                    'npv_change_percent': ((new_npv - base_result.npv) / base_result.npv) * 100
                })
                
        # Restore original parameters
        for param, value in original_params.items():
            setattr(self.params, param, value)
            
        return pd.DataFrame(results)


# Utility functions for pipeline integration

def create_cost_analyzer(config: Optional[Dict[str, Any]] = None) -> CostAnalyzer:
    """
    Factory function to create CostAnalyzer with custom parameters.
    
    Args:
        config: Configuration dictionary with cost parameters
        
    Returns:
        Configured CostAnalyzer instance
    """
    if config:
        params = CostParameters(**config)
    else:
        params = CostParameters()
        
    return CostAnalyzer(params)


def analyze_project_costs(routes_csv: str,
                        demand_csv: str,
                        output_path: str,
                        train_type: str = 'conventional',
                        currency: str = 'USD') -> pd.DataFrame:
    """
    Main pipeline function to analyze costs for all routes.
    
    Args:
        routes_csv: Path to routes CSV from route_mapping
        demand_csv: Path to demand CSV from demand_analysis
        output_path: Path to save cost analysis results
        train_type: Type of train service
        currency: Output currency (USD or EUR)
        
    Returns:
        DataFrame with cost analysis results
    """
    logger.info("Starting cost analysis...")
    
    # Load data
    routes_df = pd.read_csv(routes_csv)
    demand_df = pd.read_csv(demand_csv)
    
    # Create analyzer
    analyzer = create_cost_analyzer()
    
    # Analyze network costs
    cost_df = analyzer.analyze_network_cost(routes_df, demand_df, train_type)
    
    # Convert currency if needed
    if currency == 'USD':
        cost_columns = ['infrastructure_cost', 'rolling_stock_cost', 'total_capex', 
                       'annual_opex', 'npv']
        for col in cost_columns:
            if col in cost_df.columns:
                cost_df[col] = cost_df[col] * analyzer.params.eur_to_usd
                
    # Add summary statistics
    summary_row = {
        'route_id': 'TOTAL',
        'origin_city': 'ALL',
        'destination_city': 'ALL',
        'distance_km': cost_df['distance_km'].sum(),
        'infrastructure_cost': cost_df['infrastructure_cost'].sum(),
        'rolling_stock_cost': cost_df['rolling_stock_cost'].sum(),
        'total_capex': cost_df['total_capex'].sum(),
        'annual_opex': cost_df['annual_opex'].sum(),
        'npv': cost_df['npv'].sum()
    }
    
    # Append summary
    cost_df = pd.concat([cost_df, pd.DataFrame([summary_row])], ignore_index=True)
    
    # Save results
    cost_df.to_csv(output_path, index=False)
    logger.info(f"Cost analysis saved to {output_path}")
    
    # Log summary
    total_capex = summary_row['total_capex']
    total_opex = summary_row['annual_opex']
    logger.info(f"Total CAPEX: {currency} {total_capex:,.0f}")
    logger.info(f"Total Annual OPEX: {currency} {total_opex:,.0f}")
    logger.info(f"Total NPV: {currency} {summary_row['npv']:,.0f}")
    
    return cost_df


def generate_cost_report(cost_analysis_csv: str,
                       output_dir: str,
                       include_charts: bool = True) -> None:
    """
    Generate comprehensive cost report with visualizations.
    
    Args:
        cost_analysis_csv: Path to cost analysis CSV
        output_dir: Directory to save report files
        include_charts: Whether to generate charts
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load cost data
    cost_df = pd.read_csv(cost_analysis_csv)
    
    # Remove summary row for analysis
    analysis_df = cost_df[cost_df['route_id'] != 'TOTAL'].copy()
    
    # Generate summary report
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'total_routes': len(analysis_df),
        'total_distance_km': analysis_df['distance_km'].sum(),
        'total_capex': analysis_df['total_capex'].sum(),
        'total_annual_opex': analysis_df['annual_opex'].sum(),
        'average_cost_per_km': analysis_df['total_capex'].sum() / analysis_df['distance_km'].sum(),
        'profitable_routes': len(analysis_df[analysis_df['npv'] > 0]),
        'average_payback_period': analysis_df['payback_period'].mean()
    }
    
    # Save summary
    with open(os.path.join(output_dir, 'cost_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
        
    # Generate charts if requested
    if include_charts:
        import matplotlib.pyplot as plt
        
        # Chart 1: Cost breakdown by route
        fig, ax = plt.subplots(figsize=(12, 6))
        routes = analysis_df['route_id'].head(10)
        infrastructure = analysis_df['infrastructure_cost'].head(10)
        rolling_stock = analysis_df['rolling_stock_cost'].head(10)
        
        x = np.arange(len(routes))
        width = 0.35
        
        ax.bar(x - width/2, infrastructure/1e6, width, label='Infrastructure')
        ax.bar(x + width/2, rolling_stock/1e6, width, label='Rolling Stock')
        
        ax.set_xlabel('Route')
        ax.set_ylabel('Cost (Million USD)')
        ax.set_title('Capital Cost Breakdown by Route')
        ax.set_xticks(x)
        ax.set_xticklabels(routes, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cost_breakdown.png'), dpi=300)
        plt.close()
        
        # Chart 2: NPV distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        npv_values = analysis_df['npv'] / 1e6  # Convert to millions
        
        ax.hist(npv_values, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', label='Break-even')
        ax.set_xlabel('NPV (Million USD)')
        ax.set_ylabel('Number of Routes')
        ax.set_title('Net Present Value Distribution')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'npv_distribution.png'), dpi=300)
        plt.