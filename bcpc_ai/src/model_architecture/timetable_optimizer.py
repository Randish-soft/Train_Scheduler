import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class TimetableOptimizer(nn.Module):
    def __init__(self, station_features: int, route_features: int, 
                 demand_features: int, hidden_dim: int = 256):
        super(TimetableOptimizer, self).__init__()
        
        self.station_encoder = nn.Sequential(
            nn.Linear(station_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.route_encoder = nn.Sequential(
            nn.Linear(route_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.demand_encoder = nn.Sequential(
            nn.Linear(demand_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.lstm = nn.LSTM(hidden_dim * 3, hidden_dim, 2, 
                           batch_first=True, bidirectional=True)
        
        self.departure_time = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 24 * 60)  # Minutes in a day
        )
        
        self.frequency_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dwell_time = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, stations: torch.Tensor, routes: torch.Tensor,
                demand: torch.Tensor, num_stops: int = 10) -> Dict[str, torch.Tensor]:
        
        station_features = self.station_encoder(stations)
        route_features = self.route_encoder(routes)
        demand_features = self.demand_encoder(demand)
        
        combined = torch.cat([station_features, route_features, demand_features], dim=-1)
        
        if len(combined.shape) == 2:
            combined = combined.unsqueeze(1).repeat(1, num_stops, 1)
        
        lstm_out, _ = self.lstm(combined)
        
        departures = torch.sigmoid(self.departure_time(lstm_out)) 
        frequency = torch.relu(self.frequency_predictor(lstm_out)) + 1
        dwell = torch.relu(self.dwell_time(lstm_out)) + 0.5
        
        return {
            'departure_times': departures,
            'train_frequency': frequency,
            'dwell_times': dwell
        }

class ConflictResolver(nn.Module):
    def __init__(self, platform_features: int, schedule_features: int):
        super(ConflictResolver, self).__init__()
        
        self.conflict_detector = nn.Sequential(
            nn.Linear(platform_features + schedule_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.resolution_strategy = nn.Sequential(
            nn.Linear(platform_features + schedule_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 4 resolution strategies
        )
        
        self.time_adjustment = nn.Sequential(
            nn.Linear(platform_features + schedule_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
    def forward(self, platform: torch.Tensor, schedule: torch.Tensor) -> Dict[str, torch.Tensor]:
        combined = torch.cat([platform, schedule], dim=-1)
        
        conflict_prob = self.conflict_detector(combined)
        strategy = F.softmax(self.resolution_strategy(combined), dim=-1)
        adjustment = self.time_adjustment(combined) * 30  # +/- 30 minutes
        
        return {
            'conflict_probability': conflict_prob,
            'resolution_strategy': strategy,
            'time_adjustment_minutes': adjustment
        }

class PassengerFlowPredictor(nn.Module):
    def __init__(self, temporal_features: int, station_features: int, 
                 external_features: int):
        super(PassengerFlowPredictor, self).__init__()
        
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        self.station_encoder = nn.Sequential(
            nn.Linear(station_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        self.external_encoder = nn.Sequential(
            nn.Linear(external_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        self.gru = nn.GRU(128 + 128 + 64, 256, 2, batch_first=True)
        
        self.flow_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.peak_detector = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, temporal: torch.Tensor, station: torch.Tensor,
                external: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        temp_features = self.temporal_encoder(temporal)
        stat_features = self.station_encoder(station)
        ext_features = self.external_encoder(external)
        
        combined = torch.cat([temp_features, stat_features, ext_features], dim=-1)
        
        if len(combined.shape) == 2:
            combined = combined.unsqueeze(1)
        
        gru_out, _ = self.gru(combined)
        
        if len(gru_out.shape) == 3:
            gru_out = gru_out[:, -1, :]
        
        passenger_count = torch.relu(self.flow_predictor(gru_out))
        is_peak = self.peak_detector(gru_out)
        
        return {
            'passenger_count': passenger_count,
            'is_peak_hour': is_peak,
            'load_factor': torch.sigmoid(passenger_count / 1000)
        }