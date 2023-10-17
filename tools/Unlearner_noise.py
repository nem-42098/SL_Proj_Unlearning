from copy import deepcopy
import torch
from torch.nn import Module
import torch.nn as nn
from torch.utils.data import DataLoader
from.Unlearner import Unlearner


class NoisyUnlearner(Unlearner):
    def __init__(self, model: Module, device: str = 'cuda', lr: float = 1e-4, alpha: float = 0, beta : float = 1.):
        super().__init__(model, device, lr, alpha)
        # note that alpha is set to zero by default so no transfer learning will happen, only retrainig
        self.beta = beta
        
    
    def impair(self, model: Module, forget_set: DataLoader = None, epochs: int = None) -> Module:
        impaired = deepcopy(model)
        impaired.apply(NoisyUnlearner.reset_batch_norm)

        for param in impaired.parameters():
            std = param.data.std(dim=[0,1,2] if  param.data.ndim == 4 else None)
            param.data += torch.randn_like(param.data, device=self.device) * std * self.beta
            
        return impaired
    
    @staticmethod
    def reset_batch_norm(layer: nn.Module):
        if isinstance(layer, nn.BatchNorm2d):
            layer.reset_parameters()


