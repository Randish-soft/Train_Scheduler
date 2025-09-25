import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class CostEstimator(nn.Module):
    def __init__(self, route_features: int, terrain_features: int, 
                 economic_features: int, hidden_dim: int = 256):
        super(CostEstimator, self).__init__()
        
        self.route_encoder = nn.Sequential(
            nn.Linear(route_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2)
        )
        
        self.terrain_encoder = nn.Sequential(
            nn.Linear(terrain_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2)
        )
        
        self.economic_encoder = nn.Sequential(
            nn.Linear(economic_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2)
        )
        
        self.cost_components = nn.ModuleDict({
            'construction': nn.Linear(hidden_dim * 3, 1),
            'land_acquisition': nn.Linear(hidden_dim * 3, 1),
            'environmental': nn.Linear(hidden_dim * 3, 1),
            'maintenance': nn.Linear(hidden_dim * 3, 1),
            'operational': nn.Linear(hidden_dim * 3, 1)
        })
        
        self.total_cost = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.uncertainty = nn.Linear(hidden_dim * 3, 1)
        
    def forward(self, route: torch.Tensor, terrain: torch.Tensor, 
                economic: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        route_features = self.route_encoder(route)
        terrain_features = self.terrain_encoder(terrain)
        economic_features = self.economic_encoder(economic)
        
        combined = torch.cat([route_features, terrain_features, economic_features], dim=-1)
        
        cost_breakdown = {}
        component_costs = []
        
        for component, layer in self.cost_components.items():
            cost = torch.relu(layer(combined))
            cost_breakdown[f'{component}_cost'] = cost
            component_costs.append(cost)
        
        all_costs = torch.cat(component_costs, dim=-1)
        combined_with_components = torch.cat([combined, all_costs], dim=-1)
        
        total = self.total_cost(combined_with_components)
        uncertainty = torch.sigmoid(self.uncertainty(combined))
        
        return {
            **cost_breakdown,
            'total_cost': total,
            'cost_uncertainty': uncertainty
        }

class TimeEstimator(nn.Module):
    def __init__(self, project_features: int, complexity_features: int):
        super(TimeEstimator, self).__init__()
        
        self.phases = ['planning', 'approval', 'construction', 'testing', 'operational']
        
        self.phase_estimators = nn.ModuleDict({
            phase: nn.Sequential(
                nn.Linear(project_features + complexity_features, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for phase in self.phases
        })
        
        self.delay_predictor = nn.Sequential(
            nn.Linear(project_features + complexity_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, project: torch.Tensor, complexity: torch.Tensor) -> Dict[str, torch.Tensor]:
        combined = torch.cat([project, complexity], dim=-1)
        
        phase_durations = {}
        total_duration = 0
        
        for phase, estimator in self.phase_estimators.items():
            duration = torch.relu(estimator(combined))
            phase_durations[f'{phase}_months'] = duration
            total_duration = total_duration + duration
        
        delay_factor = self.delay_predictor(combined)
        
        return {
            **phase_durations,
            'total_months': total_duration,
            'delay_probability': delay_factor,
            'adjusted_total': total_duration * (1 + delay_factor * 0.5)
        }

class ROICalculator(nn.Module):
    def __init__(self, financial_features: int, usage_features: int):
        super(ROICalculator, self).__init__()
        
        self.revenue_streams = nn.ModuleDict({
            'ticket_sales': nn.Linear(financial_features + usage_features, 1),
            'freight': nn.Linear(financial_features + usage_features, 1),
            'subsidies': nn.Linear(financial_features + usage_features, 1),
            'ancillary': nn.Linear(financial_features + usage_features, 1)
        })
        
        self.payback_period = nn.Sequential(
            nn.Linear(financial_features + usage_features + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.irr_calculator = nn.Sequential(
            nn.Linear(financial_features + usage_features + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, financial: torch.Tensor, usage: torch.Tensor) -> Dict[str, torch.Tensor]:
        combined = torch.cat([financial, usage], dim=-1)
        
        revenues = []
        revenue_breakdown = {}
        
        for stream, layer in self.revenue_streams.items():
            revenue = torch.relu(layer(combined))
            revenue_breakdown[f'{stream}_revenue'] = revenue
            revenues.append(revenue)
        
        total_revenue = torch.cat(revenues, dim=-1)
        combined_with_revenue = torch.cat([combined, total_revenue], dim=-1)
        
        payback = torch.relu(self.payback_period(combined_with_revenue))
        irr = self.irr_calculator(combined_with_revenue) * 0.3
        
        return {
            **revenue_breakdown,
            'total_annual_revenue': sum(revenues),
            'payback_years': payback,
            'internal_rate_return': irr
        }