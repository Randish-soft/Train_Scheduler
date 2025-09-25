import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class NIMBYAnalyzer(nn.Module):
    def __init__(self, demographic_features: int, land_features: int, 
                 heritage_features: int, hidden_dim: int = 128):
        super(NIMBYAnalyzer, self).__init__()
        
        self.demographic_encoder = nn.Sequential(
            nn.Linear(demographic_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2)
        )
        
        self.land_encoder = nn.Sequential(
            nn.Linear(land_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2)
        )
        
        self.heritage_encoder = nn.Sequential(
            nn.Linear(heritage_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2)
        )
        
        self.resistance_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.solution_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 5)  # 5 solution types
        )
        
        self.cost_impact = nn.Linear(hidden_dim * 3, 1)
        
    def forward(self, demographic: torch.Tensor, land: torch.Tensor, 
                heritage: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        demo_features = self.demographic_encoder(demographic)
        land_features = self.land_encoder(land)
        heritage_features = self.heritage_encoder(heritage)
        
        combined = torch.cat([demo_features, land_features, heritage_features], dim=-1)
        
        resistance_score = self.resistance_predictor(combined)
        solution_probs = F.softmax(self.solution_classifier(combined), dim=-1)
        additional_cost = self.cost_impact(combined)
        
        return {
            'resistance_score': resistance_score,
            'solution_probabilities': solution_probs,
            'additional_cost_factor': torch.sigmoid(additional_cost)
        }
    
    def get_solution_type(self, solution_probs: torch.Tensor) -> List[str]:
        solution_types = [
            'underground_tunnel',
            'elevated_track',
            'noise_barriers',
            'route_deviation',
            'mixed_solution'
        ]
        
        indices = torch.argmax(solution_probs, dim=-1)
        return [solution_types[idx] for idx in indices]

class LandValueEstimator(nn.Module):
    def __init__(self, location_features: int, economic_features: int):
        super(LandValueEstimator, self).__init__()
        
        self.estimator = nn.Sequential(
            nn.Linear(location_features + economic_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.uncertainty = nn.Linear(location_features + economic_features, 1)
        
    def forward(self, location: torch.Tensor, economic: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([location, economic], dim=-1)
        value = self.estimator(combined)
        uncertainty = torch.sigmoid(self.uncertainty(combined))
        return value, uncertainty

class HeritageImpactAssessor(nn.Module):
    def __init__(self, site_features: int, route_features: int):
        super(HeritageImpactAssessor, self).__init__()
        
        self.impact_levels = ['no_impact', 'minimal', 'moderate', 'severe', 'unacceptable']
        
        self.assessor = nn.Sequential(
            nn.Linear(site_features + route_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.impact_levels))
        )
        
        self.mitigation_suggester = nn.Sequential(
            nn.Linear(site_features + route_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 mitigation strategies
        )
        
    def forward(self, site: torch.Tensor, route: torch.Tensor) -> Dict[str, torch.Tensor]:
        combined = torch.cat([site, route], dim=-1)
        
        impact_scores = F.softmax(self.assessor(combined), dim=-1)
        mitigation_scores = torch.sigmoid(self.mitigation_suggester(combined))
        
        return {
            'impact_levels': impact_scores,
            'mitigation_strategies': mitigation_scores
        }