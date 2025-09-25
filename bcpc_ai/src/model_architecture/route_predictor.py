import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_model import RailwayBaseModel, GraphRailwayModel

class RoutePredictor(nn.Module):
    def __init__(self, station_features: int, terrain_features: int, 
                 hidden_dim: int = 256, num_layers: int = 4):
        super(RoutePredictor, self).__init__()
        
        self.station_encoder = nn.Sequential(
            nn.Linear(station_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.terrain_encoder = nn.Sequential(
            nn.Linear(terrain_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, 
                            batch_first=True, bidirectional=True)
        
        self.route_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # x, y, track_type
        )
        
        self.cost_predictor = nn.Linear(hidden_dim * 2, 1)
        self.speed_predictor = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, station_features: torch.Tensor, terrain_features: torch.Tensor,
                sequence_length: int = 100) -> Dict[str, torch.Tensor]:
        
        station_encoded = self.station_encoder(station_features)
        terrain_encoded = self.terrain_encoder(terrain_features)
        
        combined = torch.cat([station_encoded, terrain_encoded], dim=-1)
        combined = combined.unsqueeze(1).repeat(1, sequence_length, 1)
        
        lstm_out, _ = self.lstm(combined)
        
        route_points = self.route_decoder(lstm_out)
        route_cost = self.cost_predictor(lstm_out).squeeze(-1)
        route_speed = self.speed_predictor(lstm_out).squeeze(-1)
        
        return {
            'route': route_points,
            'cost': route_cost,
            'speed': route_speed
        }

class A3CRouteOptimizer(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(A3CRouteOptimizer, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared_layers(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_probs, value = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action), value

class RouteSegmentClassifier(nn.Module):
    def __init__(self, input_dim: int, num_segments: int = 5):
        super(RouteSegmentClassifier, self).__init__()
        
        self.segment_types = ['tunnel', 'bridge', 'ground_level', 'elevated', 'underground']
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_segments)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)