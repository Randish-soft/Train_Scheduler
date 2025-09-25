import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif self._is_better(val_score):
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
        
        return self.early_stop
    
    def _is_better(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

class ModelCheckpoint:
    def __init__(self, filepath: str, monitor: str = 'val_loss', mode: str = 'min', 
                 save_best_only: bool = True):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best = None
        
    def __call__(self, model: torch.nn.Module, metrics: Dict[str, float], epoch: int):
        current = metrics.get(self.monitor)
        
        if current is None:
            return
        
        save = False
        if self.best is None:
            save = True
            self.best = current
        elif self.mode == 'min' and current < self.best:
            save = True
            self.best = current
        elif self.mode == 'max' and current > self.best:
            save = True
            self.best = current
        elif not self.save_best_only:
            save = True
        
        if save:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_score': self.best,
                'metrics': metrics
            }, self.filepath)
            logger.info(f"Model checkpoint saved to {self.filepath}")

class LearningRateScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, mode: str = 'step', **kwargs):
        self.optimizer = optimizer
        self.mode = mode
        self.step_count = 0
        self.kwargs = kwargs
        
    def step(self, metrics: Optional[float] = None):
        self.step_count += 1
        
        if self.mode == 'step':
            if self.step_count % self.kwargs.get('step_size', 30) == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.kwargs.get('gamma', 0.1)
        
        elif self.mode == 'exponential':
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.kwargs.get('gamma', 0.95)
        
        elif self.mode == 'reduce_on_plateau' and metrics is not None:
            # Simple implementation
            if not hasattr(self, 'best'):
                self.best = metrics
            elif metrics > self.best * 1.01:  # If worse by 1%
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.5
                self.best = metrics
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]