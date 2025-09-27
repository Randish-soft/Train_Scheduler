#!/usr/bin/env python3
"""
Simplified training script that works with mock data
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRailwayModel(nn.Module):
    """Simple model that works with any input size"""
    def __init__(self, input_dim=30, hidden_dim=64, output_dim=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def create_mock_data(num_samples=100, input_dim=30, output_dim=3):
    """Create mock training data"""
    X = torch.randn(num_samples, input_dim)
    y = torch.randn(num_samples, output_dim)
    return X, y

def train_model(model, train_loader, val_loader, epochs=5):
    """Simple training loop"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return model

def main():
    logger.info("Starting simplified training...")
    
    # Create mock data
    logger.info("Creating mock data...")
    X_train, y_train = create_mock_data(100, 30, 3)
    X_val, y_val = create_mock_data(20, 30, 3)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create and train model
    logger.info("Initializing model...")
    model = SimpleRailwayModel(input_dim=30, hidden_dim=64, output_dim=3)
    
    logger.info("Training model...")
    trained_model = train_model(model, train_loader, val_loader, epochs=5)
    
    # Save model
    save_dir = Path("models/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / "simple_model.pth"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'model_config': {
            'input_dim': 30,
            'hidden_dim': 64,
            'output_dim': 3
        }
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info("Training complete!")

if __name__ == "__main__":
    main()