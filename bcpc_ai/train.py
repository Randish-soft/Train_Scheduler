#!/usr/bin/env python3
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.data_pipeline.data_loader import RailwayDataLoader
from src.data_pipeline.feature_extractor import RailwayFeatureExtractor
from src.data_pipeline.data_splitter import RailwayDataSplitter
from src.model_architecture.route_predictor import RoutePredictor
from src.training_loops.trainer import RailwayTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_datasets(config: dict):
    data_loader = RailwayDataLoader(
        data_dir=config['data']['data_dir'],
        config_path=config['data']['config_path']
    )
    
    feature_extractor = RailwayFeatureExtractor()
    data_splitter = RailwayDataSplitter(
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size']
    )
    
    logger.info("Loading training data...")
    train_data = data_loader.load_all_train_data()
    
    all_features = []
    all_labels = []
    
    for country, country_data in train_data.items():
        logger.info(f"Processing {country} data...")
        
        line_features = feature_extractor.extract_line_features(country_data['railways'])
        station_features = feature_extractor.extract_station_features(country_data['stations'])
        timetable_features = feature_extractor.extract_timetable_features(country_data['timetables'])
        terrain_features = feature_extractor.extract_terrain_features(
            country_data['terrain'], 
            [(0, 0)] * len(country_data['terrain'])
        )
        cost_features = feature_extractor.extract_cost_features(country_data['costs'])
        flow_features = feature_extractor.extract_passenger_flow_features(country_data['passenger_flow'])
        
        combined_features = feature_extractor.combine_features({
            'line': line_features,
            'station': station_features,
            'timetable': timetable_features,
            'terrain': terrain_features,
            'cost': cost_features,
            'flow': flow_features
        })
        
        if not combined_features.empty:
            all_features.append(combined_features)
            all_labels.append(torch.randn(len(combined_features), 3))
    
    if all_features:
        import pandas as pd
        all_features = pd.concat(all_features, ignore_index=True)
        all_labels = torch.cat(all_labels, dim=0)
        
        X_train, X_val, X_test, y_train, y_val, y_test = data_splitter.split_by_routes(
            all_features, all_labels
        )
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train.values),
            torch.FloatTensor(y_train.numpy() if hasattr(y_train, 'numpy') else y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val.values),
            torch.FloatTensor(y_val.numpy() if hasattr(y_val, 'numpy') else y_val)
        )
        
        return train_dataset, val_dataset, X_train.shape[1]
    
    return None, None, 0

def main():
    parser = argparse.ArgumentParser(description='Train BCPC Railway Model')
    parser.add_argument('--config', type=str, default='configs/training/schedule.yaml',
                       help='Path to training config file')
    parser.add_argument('--model-config', type=str, default='configs/model/architecture.yaml',
                       help='Path to model config file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    config = {
        'data': {
            'data_dir': 'data',
            'config_path': 'configs/data/train_config.yaml',
            'test_size': 0.2,
            'val_size': 0.1
        },
        'model': {
            'station_features': 10,
            'terrain_features': 8,
            'hidden_dim': 256,
            'num_layers': 4
        },
        'training': {
            'optimizer': 'adamw',
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'scheduler': 'cosine',
            'loss': 'mse',
            'gradient_clip': 1.0,
            'early_stopping_patience': 15,
            'use_wandb': False
        }
    }
    
    if Path(args.config).exists():
        config['training'].update(load_config(args.config))
    
    if Path(args.model_config).exists():
        config['model'].update(load_config(args.model_config))
    
    config['training']['epochs'] = args.epochs
    
    logger.info("Preparing datasets...")
    train_dataset, val_dataset, input_dim = prepare_datasets(config)
    
    if train_dataset is None:
        logger.error("No data available for training")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info("Initializing model...")
    model = RoutePredictor(
        station_features=config['model']['station_features'],
        terrain_features=config['model']['terrain_features'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers']
    )
    
    logger.info("Starting training...")
    trainer = RailwayTrainer(model, config['training'], device=args.device)
    trainer.train(train_loader, val_loader, args.epochs, save_dir='models/checkpoints')
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()