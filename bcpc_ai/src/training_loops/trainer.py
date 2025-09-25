import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import wandb
from pathlib import Path

logger = logging.getLogger(__name__)

class RailwayTrainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = self._setup_criterion()
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.epoch = 0
        
        if config.get('use_wandb', False):
            wandb.init(project="bcpc-railway", config=config)
        
    def _setup_optimizer(self) -> optim.Optimizer:
        opt_name = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if opt_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        sched_name = self.config.get('scheduler', None)
        
        if sched_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.get('epochs', 100)
            )
        elif sched_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif sched_name == 'reduce':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5
            )
        return None
    
    def _setup_criterion(self) -> nn.Module:
        loss_name = self.config.get('loss', 'mse')
        
        if loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'mae':
            return nn.L1Loss()
        elif loss_name == 'huber':
            return nn.HuberLoss()
        elif loss_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch in progress_bar:
            inputs, targets = self._prepare_batch(batch)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            if self.config.get('gradient_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip']
                )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        num_batches = 0
        predictions = []
        ground_truth = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = self._prepare_batch(batch)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                predictions.append(outputs.cpu().numpy())
                ground_truth.append(targets.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        metrics = self._calculate_metrics(
            np.concatenate(predictions),
            np.concatenate(ground_truth)
        )
        
        metrics['val_loss'] = avg_loss
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int, save_dir: str = 'models/checkpoints'):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            self._log_metrics({**train_metrics, **val_metrics})
            
            if val_metrics['val_loss'] < self.best_loss:
                self.best_loss = val_metrics['val_loss']
                self.patience_counter = 0
                self.save_checkpoint(save_path / 'best_model.pth')
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config.get('early_stopping_patience', 10):
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                self.save_checkpoint(save_path / f'checkpoint_epoch_{epoch}.pth')
    
    def _prepare_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
        else:
            inputs = batch['inputs'].to(self.device)
            targets = batch['targets'].to(self.device)
        
        return inputs, targets
    
    def _calculate_metrics(self, predictions: np.ndarray, 
                          ground_truth: np.ndarray) -> Dict[str, float]:
        mse = np.mean((predictions - ground_truth) ** 2)
        mae = np.mean(np.abs(predictions - ground_truth))
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }
    
    def _log_metrics(self, metrics: Dict[str, float]):
        logger.info(f"Epoch {self.epoch}: {metrics}")
        
        if self.config.get('use_wandb', False):
            wandb.log(metrics, step=self.epoch)
    
    def save_checkpoint(self, path: Path):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'config': self.config
        }, path)
        
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"Checkpoint loaded from {path}")