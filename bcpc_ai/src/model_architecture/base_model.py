import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

class RailwayBaseModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.3):
        super(RailwayBaseModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        output = self.output_layer(features)
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

class GraphRailwayModel(nn.Module):
    def __init__(self, node_features: int, edge_features: int, hidden_dim: int, num_classes: int):
        super(GraphRailwayModel, self).__init__()
        self.node_encoder = nn.Linear(node_features, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)
        
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(3)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, 
                edge_index: torch.Tensor) -> torch.Tensor:
        x = self.node_encoder(node_features)
        edge_attr = self.edge_encoder(edge_features)
        
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_attr)
            x = F.relu(x)
        
        out = self.classifier(x)
        return out

class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features * 2, out_features)
        self.edge_update = nn.Linear(in_features, out_features)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        row, col = edge_index
        
        if edge_attr is not None:
            edge_features = self.edge_update(edge_attr)
            agg = torch.zeros_like(x)
            agg = agg.index_add(0, col, edge_features)
        else:
            agg = torch.zeros_like(x)
            agg = agg.index_add(0, col, x[row])
        
        out = torch.cat([x, agg], dim=-1)
        out = self.linear(out)
        return out

class AttentionRailwayModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, output_dim: int):
        super(AttentionRailwayModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(x)
        
        for attention, norm in zip(self.attention_layers, self.norm_layers):
            attn_out, _ = attention(x, x, x, attn_mask=mask)
            x = norm(x + attn_out)
            x = x + self.feed_forward(x)
        
        output = self.output_layer(x)
        return output

class VAERailwayModel(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int]):
        super(VAERailwayModel, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar