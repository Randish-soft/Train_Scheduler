"""
BCPC Pipeline - Cost Analysis Module
====================================

This module provides comprehensive cost estimation for railway construction and operations,
including infrastructure, rolling stock, operational expenses, and lifecycle costs.

Based on EU 2019 cost coefficients with provisions for localization.
Integrates with the BCPC pipeline for terrain-aware cost calculations.

Author: BCPC Pipeline Team
License: Open Source
"""

import json
import logging
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import yaml
import numpy as np
from pathlib import Path
from terrain_analysis import TerrainComplexity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrackGauge(Enum):
    """Track gauge specifications"""
    STANDARD = "1435mm"  # Standard gauge
    METRIC = "1000mm"    # Metric gauge
    BROAD = "1676mm"     # Broad gauge (Indian Railways)
    
class TrainType(Enum):
    """Train technology types"""
    DIESEL = "diesel"
    ELECTRIC_EMU = "electric_emu"
    ELECTRIC_LOCOMOTIVE = "electric_locomotive"
    HYBRID = "hybrid"

@dataclass
class CostCoefficients:
    """
    Cost coefficients based on EU 2019 averages
    All costs in EUR unless specified otherwise
    """
    # Infrastructure costs per km
    track_construction_flat: float = 2_500_000     # EUR/km for flat terrain
    track_construction_rolling: float = 4_000_000  # EUR/km for rolling terrain
    track_construction_mountainous: float = 8_000_000  # EUR/km for mountainous terrain
    track_construction_urban: float = 12_000_000   # EUR/km for urban areas
    
    # Electrification costs
    electrification_per_km: float = 750_000       # EUR/km for overhead lines
    signaling_per_km: float = 500_000             # EUR/km for modern signaling
    
    # Station costs
    basic_station_cost: float = 2_000_000         # EUR per basic station
    major_station_cost: float = 10_000_000        # EUR per major station/terminus
    platform_cost_per_meter: float = 15_000      # EUR per meter of platform
    
    # Rolling stock costs (per unit)
    diesel_trainset_cost: float = 3_500_000       # EUR per diesel trainset
    electric_emu_cost: float = 4_200_000          # EUR per electric EMU
    electric_locomotive_cost: float = 5_000_000   # EUR per electric locomotive
    passenger_car_cost: float = 1_800_000         # EUR per passenger car
    
    # Operational costs (annual)
    maintenance_track_per_km: float = 50_000      # EUR/km/year
    maintenance_rolling_stock_factor: float = 0.08  # % of purchase price/year
    energy_cost_electric: float = 0.12            # EUR/kWh
    energy_cost_diesel: float = 1.2               # EUR/liter
    staff_cost_per_employee: float = 65_000       # EUR/year including benefits
    
    # Terrain adjustment factors
    terrain_multipliers = {
        TerrainComplexity.FLAT: 1.0,
        TerrainComplexity.ROLLING: 1.6,
        TerrainComplexity.MOUNTAINOUS: 3.2,
        TerrainComplexity.URBAN: 4.8
    }

@dataclass
class NetworkDesign:
    """
    Network design specification output from optimization module
    """
    route_length_km: float
    track_gauge: TrackGauge
    train_type: TrainType
    number_of_trainsets: int
    electrification_required: bool
    number_of_stations: int
    major_stations: int
    terrain_complexity: TerrainComplexity
    daily_passengers_per_direction: int
    operating_speed_kmh: float = 120
    service_frequency_minutes: int = 30
    
@dataclass
class InfrastructureCosts:
    """Infrastructure construction costs breakdown"""
    track_construction: float
    electrification: float
    signaling: float
    stations: float
    total: float
    
@dataclass
class RollingStockCosts:
    """Rolling stock acquisition costs"""
    trainsets: float
    total: float
    spare_parts_factor: float = 0.15  # 15% of trainset cost for spare parts
    
@dataclass
class OperationalCosts:
    """Annual operational costs breakdown"""
    track_maintenance: float
    rolling_stock_maintenance: float
    energy: float
    staff: float
    total_annual: float
    
@dataclass
class CostSummary:
    """Complete cost analysis summary"""
    infrastructure: InfrastructureCosts
    rolling_stock: RollingStockCosts
    operational_annual: OperationalCosts
    total_capex: float
    lifecycle_cost_20_years: float  # NPV over 20 years
    cost_per_passenger_km: float
    cost_per_route_km: float

class CostAnalyzer:
    """
    Main cost analysis engine for BCPC railway projects
    """
    
    def __init__(self, cost_coefficients: Optional[CostCoefficients] = None, 
                 localization_factor: float = 1.0):
        """
        Initialize cost analyzer
        
        Args:
            cost_coefficients: Custom cost coefficients, uses defaults if None
            localization_factor: Factor to adjust EU costs to local market (default 1.0)
        """
        self.coeffs = cost_coefficients or CostCoefficients()
        self.localization_factor = localization_factor
        self.discount_rate = 0.04  # 4% discount rate for NPV calculations
        
    def analyze_full_cost(self, network_design: NetworkDesign, 
                         budget_constraint: Optional[float] = None) -> CostSummary:
        """
        Perform comprehensive cost analysis for a railway network design
        
        Args:
            network_design: Specifications of the proposed railway
            budget_constraint: Optional budget limit in EUR
            
        Returns:
            Complete cost breakdown and summary
        """
        logger.info(f"Analyzing costs for {network_design.route_length_km:.1f}km railway")
        
        # Calculate infrastructure costs
        infrastructure = self._calculate_infrastructure_costs(network_design)
        
        # Calculate rolling stock costs
        rolling_stock = self._calculate_rolling_stock_costs(network_design)
        
        # Calculate operational costs
        operational = self._calculate_operational_costs(network_design)
        
        # Calculate totals and lifecycle costs
        total_capex = infrastructure.total + rolling_stock.total
        lifecycle_cost = self._calculate_lifecycle_cost(total_capex, operational.total_annual)
        
        # Calculate unit costs
        annual_passenger_km = (network_design.daily_passengers_per_direction * 2 * 
                              network_design.route_length_km * 365)
        cost_per_passenger_km = lifecycle_cost / (annual_passenger_km * 20) if annual_passenger_km > 0 else 0
        cost_per_route_km = total_capex / network_design.route_length_km
        
        summary = CostSummary(
            infrastructure=infrastructure,
            rolling_stock=rolling_stock,
            operational_annual=operational,
            total_capex=total_capex,
            lifecycle_cost_20_years=lifecycle_cost,
            cost_per_passenger_km=cost_per_passenger_km,
            cost_per_route_km=cost_per_route_km
        )
        
        # Check budget constraint
        if budget_constraint and total_capex > budget_constraint:
            logger.warning(f"Project exceeds budget: €{total_capex:,.0f} > €{budget_constraint:,.0f}")
            logger.info(f"Budget overrun: €{total_capex - budget_constraint:,.0f}")
            
        return summary
    
    def _calculate_infrastructure_costs(self, design: NetworkDesign) -> InfrastructureCosts:
        """Calculate infrastructure construction costs"""
        
        # Base track construction cost based on terrain
        terrain_multiplier = self.coeffs.terrain_multipliers[design.terrain_complexity]
        base_cost_per_km = self.coeffs.track_construction_flat * terrain_multiplier
        track_cost = base_cost_per_km * design.route_length_km * self.localization_factor
        
        # Electrification costs
        electrification_cost = 0
        if design.electrification_required:
            electrification_cost = (self.coeffs.electrification_per_km * 
                                  design.route_length_km * self.localization_factor)
        
        # Signaling costs
        signaling_cost = (self.coeffs.signaling_per_km * 
                         design.route_length_km * self.localization_factor)
        
        # Station costs
        basic_stations = max(0, design.number_of_stations - design.major_stations)
        station_cost = (basic_stations * self.coeffs.basic_station_cost + 
                       design.major_stations * self.coeffs.major_station_cost) * self.localization_factor
        
        total_infrastructure = track_cost + electrification_cost + signaling_cost + station_cost
        
        return InfrastructureCosts(
            track_construction=track_cost,
            electrification=electrification_cost,
            signaling=signaling_cost,
            stations=station_cost,
            total=total_infrastructure
        )
    
    def _calculate_rolling_stock_costs(self, design: NetworkDesign) -> RollingStockCosts:
        """Calculate rolling stock acquisition costs"""
        
        # Select appropriate cost per trainset
        if design.train_type == TrainType.DIESEL:
            cost_per_unit = self.coeffs.diesel_trainset_cost
        elif design.train_type == TrainType.ELECTRIC_EMU:
            cost_per_unit = self.coeffs.electric_emu_cost
        elif design.train_type == TrainType.ELECTRIC_LOCOMOTIVE:
            # Locomotive + cars
            locomotive_cost = self.coeffs.electric_locomotive_cost
            cars_per_train = 6  # Typical configuration
            car_cost = cars_per_train * self.coeffs.passenger_car_cost
            cost_per_unit = locomotive_cost + car_cost
        else:  # HYBRID
            cost_per_unit = self.coeffs.electric_emu_cost * 1.2  # 20% premium for hybrid
            
        trainset_cost = (cost_per_unit * design.number_of_trainsets * 
                        self.localization_factor)
        
        # Add spare parts
        spare_parts_cost = trainset_cost * 0.15  # 15% for spare parts
        
        total_rolling_stock = trainset_cost + spare_parts_cost
        
        return RollingStockCosts(
            trainsets=trainset_cost,
            total=total_rolling_stock
        )
    
    def _calculate_operational_costs(self, design: NetworkDesign) -> OperationalCosts:
        """Calculate annual operational costs"""
        
        # Track maintenance
        track_maintenance = (self.coeffs.maintenance_track_per_km * 
                           design.route_length_km * self.localization_factor)
        
        # Rolling stock maintenance (% of purchase price)
        rolling_stock_purchase_value = self._calculate_rolling_stock_costs(design).trainsets
        rolling_stock_maintenance = (rolling_stock_purchase_value * 
                                   self.coeffs.maintenance_rolling_stock_factor)
        
        # Energy costs
        energy_cost = self._calculate_energy_costs(design)
        
        # Staff costs
        staff_cost = self._calculate_staff_costs(design)
        
        total_annual = (track_maintenance + rolling_stock_maintenance + 
                       energy_cost + staff_cost)
        
        return OperationalCosts(
            track_maintenance=track_maintenance,
            rolling_stock_maintenance=rolling_stock_maintenance,
            energy=energy_cost,
            staff=staff_cost,
            total_annual=total_annual
        )
    
    def _calculate_energy_costs(self, design: NetworkDesign) -> float:
        """Calculate annual energy costs"""
        
        # Estimate annual train-km
        trains_per_day = 24 * 60 / design.service_frequency_minutes  # Both directions
        annual_train_km = trains_per_day * design.route_length_km * 365
        
        if design.train_type in [TrainType.ELECTRIC_EMU, TrainType.ELECTRIC_LOCOMOTIVE]:
            # Electric traction: kWh consumption per train-km
            kwh_per_train_km = 6.0  # Typical for electric trains
            annual_kwh = annual_train_km * kwh_per_train_km
            return annual_kwh * self.coeffs.energy_cost_electric * self.localization_factor
            
        elif design.train_type == TrainType.DIESEL:
            # Diesel traction: liters per train-km
            liters_per_train_km = 2.5  # Typical for diesel trains
            annual_liters = annual_train_km * liters_per_train_km
            return annual_liters * self.coeffs.energy_cost_diesel * self.localization_factor
            
        else:  # HYBRID
            # Split between electric and diesel
            electric_portion = 0.7
            diesel_portion = 0.3
            
            kwh_cost = (annual_train_km * electric_portion * 6.0 * 
                       self.coeffs.energy_cost_electric)
            diesel_cost = (annual_train_km * diesel_portion * 2.5 * 
                          self.coeffs.energy_cost_diesel)
            
            return (kwh_cost + diesel_cost) * self.localization_factor
    
    def _calculate_staff_costs(self, design: NetworkDesign) -> float:
        """Calculate annual staff costs"""
        
        # Estimate required staff
        # Rule of thumb: 1 driver + 1 conductor per active trainset
        # Plus maintenance and administrative staff
        
        active_trainsets = min(design.number_of_trainsets, 
                              max(2, int(24 * 60 / design.service_frequency_minutes / 2)))
        
        # Operating staff (drivers, conductors) - 3 shifts coverage
        operating_staff = active_trainsets * 2 * 3  # 2 crew per trainset, 3 shifts
        
        # Maintenance staff (10% of operating staff)
        maintenance_staff = max(2, int(operating_staff * 0.1))
        
        # Administrative staff
        admin_staff = max(1, int((operating_staff + maintenance_staff) * 0.05))
        
        total_staff = operating_staff + maintenance_staff + admin_staff
        
        return total_staff * self.coeffs.staff_cost_per_employee * self.localization_factor
    
    def _calculate_lifecycle_cost(self, capex: float, annual_opex: float, 
                                 years: int = 20) -> float:
        """Calculate Net Present Value of lifecycle costs"""
        
        npv_opex = 0
        for year in range(1, years + 1):
            npv_opex += annual_opex / ((1 + self.discount_rate) ** year)
        
        return capex + npv_opex
    
    def generate_cost_report(self, summary: CostSummary, 
                           network_design: NetworkDesign) -> str:
        """Generate a human-readable cost report"""
        
        report = []
        report.append("=" * 60)
        report.append("BCPC RAILWAY COST ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Project summary
        report.append("PROJECT SUMMARY:")
        report.append(f"  Route Length: {network_design.route_length_km:.1f} km")
        report.append(f"  Track Gauge: {network_design.track_gauge.value}")
        report.append(f"  Train Type: {network_design.train_type.value.title()}")
        report.append(f"  Number of Trainsets: {network_design.number_of_trainsets}")
        report.append(f"  Terrain: {network_design.terrain_complexity.value.title()}")
        report.append(f"  Daily Passengers (per direction): {network_design.daily_passengers_per_direction:,}")
        report.append("")
        
        # Infrastructure costs
        report.append("INFRASTRUCTURE COSTS (CAPEX):")
        report.append(f"  Track Construction: €{summary.infrastructure.track_construction:,.0f}")
        report.append(f"  Electrification: €{summary.infrastructure.electrification:,.0f}")
        report.append(f"  Signaling: €{summary.infrastructure.signaling:,.0f}")
        report.append(f"  Stations: €{summary.infrastructure.stations:,.0f}")
        report.append(f"  Total Infrastructure: €{summary.infrastructure.total:,.0f}")
        report.append("")
        
        # Rolling stock costs
        report.append("ROLLING STOCK COSTS (CAPEX):")
        report.append(f"  Trainsets & Equipment: €{summary.rolling_stock.trainsets:,.0f}")
        report.append(f"  Spare Parts (15%): €{summary.rolling_stock.total - summary.rolling_stock.trainsets:,.0f}")
        report.append(f"  Total Rolling Stock: €{summary.rolling_stock.total:,.0f}")
        report.append("")
        
        # Total CAPEX
        report.append("TOTAL CAPITAL EXPENDITURE:")
        report.append(f"  €{summary.total_capex:,.0f}")
        report.append(f"  €{summary.cost_per_route_km:,.0f} per km")
        report.append("")
        
        # Operational costs
        report.append("ANNUAL OPERATIONAL COSTS (OPEX):")
        report.append(f"  Track Maintenance: €{summary.operational_annual.track_maintenance:,.0f}")
        report.append(f"  Rolling Stock Maintenance: €{summary.operational_annual.rolling_stock_maintenance:,.0f}")
        report.append(f"  Energy: €{summary.operational_annual.energy:,.0f}")
        report.append(f"  Staff: €{summary.operational_annual.staff:,.0f}")
        report.append(f"  Total Annual OPEX: €{summary.operational_annual.total_annual:,.0f}")
        report.append("")
        
        # Lifecycle costs
        report.append("LIFECYCLE COSTS (20-year NPV @ 4%):")
        report.append(f"  Total Lifecycle Cost: €{summary.lifecycle_cost_20_years:,.0f}")
        report.append(f"  Cost per Passenger-km: €{summary.cost_per_passenger_km:.3f}")
        report.append("")
        
        # Financial indicators
        annual_passenger_km = (network_design.daily_passengers_per_direction * 2 * 
                              network_design.route_length_km * 365)
        if annual_passenger_km > 0:
            revenue_per_passenger_km = 0.10  # Assumed €0.10 per passenger-km
            annual_revenue = annual_passenger_km * revenue_per_passenger_km
            operating_ratio = summary.operational_annual.total_annual / annual_revenue
            
            report.append("FINANCIAL INDICATORS:")
            report.append(f"  Annual Passenger-km: {annual_passenger_km:,.0f}")
            report.append(f"  Assumed Revenue per Passenger-km: €{revenue_per_passenger_km:.3f}")
            report.append(f"  Estimated Annual Revenue: €{annual_revenue:,.0f}")
            report.append(f"  Operating Ratio (OPEX/Revenue): {operating_ratio:.2f}")
            report.append("")
        
        report.append("=" * 60)
        report.append("Report generated by BCPC Cost Analysis Module")
        report.append("Based on EU 2019 cost coefficients")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def export_cost_breakdown(self, summary: CostSummary, 
                            network_design: NetworkDesign,
                            output_path: str = "cost_breakdown.json") -> None:
        """Export detailed cost breakdown to JSON"""
        
        export_data = {
            "network_design": asdict(network_design),
            "cost_summary": asdict(summary),
            "analysis_metadata": {
                "localization_factor": self.localization_factor,
                "discount_rate": self.discount_rate,
                "cost_coefficients": asdict(self.coeffs)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Cost breakdown exported to {output_path}")

def estimate_cost(network_design: NetworkDesign, 
                 localization_factor: float = 1.0,
                 budget_constraint: Optional[float] = None) -> CostSummary:
    """
    Convenience function for cost estimation
    
    Args:
        network_design: Railway network specifications
        localization_factor: Factor to adjust EU costs to local market
        budget_constraint: Optional budget limit in EUR
        
    Returns:
        Complete cost analysis summary
    """
    analyzer = CostAnalyzer(localization_factor=localization_factor)
    return analyzer.analyze_full_cost(network_design, budget_constraint)

def load_cost_coefficients_from_yaml(yaml_path: str) -> CostCoefficients:
    """Load custom cost coefficients from YAML file"""
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    return CostCoefficients(**data)

# Example usage and testing
if __name__ == "__main__":
    # Example network design for testing
    example_design = NetworkDesign(
        route_length_km=150.0,
        track_gauge=TrackGauge.STANDARD,
        train_type=TrainType.ELECTRIC_EMU,
        number_of_trainsets=8,
        electrification_required=True,
        number_of_stations=12,
        major_stations=2,
        terrain_complexity=TerrainComplexity.ROLLING,
        daily_passengers_per_direction=5000,
        operating_speed_kmh=120,
        service_frequency_minutes=30
    )
    
    # Perform cost analysis
    analyzer = CostAnalyzer(localization_factor=0.8)  # 20% lower costs for developing country
    cost_summary = analyzer.analyze_full_cost(example_design, budget_constraint=500_000_000)
    
    # Generate and print report
    report = analyzer.generate_cost_report(cost_summary, example_design)
    print(report)
    
    # Export detailed breakdown
    analyzer.export_cost_breakdown(cost_summary, example_design, "example_cost_breakdown.json")
    
    logger.info("Cost analysis completed successfully!")