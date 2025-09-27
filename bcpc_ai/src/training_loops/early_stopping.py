import numpy as np
import logging

logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0.0001, mode: str = 'min', verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf if mode == 'min' else -np.Inf
        
    def __call__(self, val_score: float, model=None, path: str = 'checkpoint.pt'):
        score = -val_score if self.mode == 'max' else val_score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model, path)
        elif score < self.best_score - self.min_delta:
            self.best_score = score
            self.save_checkpoint(val_score, model, path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def save_checkpoint(self, val_score: float, model, path: str):
        if self.verbose:
            if self.mode == 'min':
                logger.info(f'Validation score decreased ({self.val_score_min:.6f} --> {val_score:.6f}). Saving model...')
            else:
                logger.info(f'Validation score increased ({self.val_score_min:.6f} --> {val_score:.6f}). Saving model...')
        
        if model is not None:
            import torch
            torch.save(model.state_dict(), path)
        
        self.val_score_min = val_score
    
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf if self.mode == 'min' else -np.Inf