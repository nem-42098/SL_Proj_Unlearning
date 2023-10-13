from copy import deepcopy
from tqdm import tqdm
import torch
from torch.nn import Module
import torch.nn as nn
from torch.utils.data import DataLoader
from.Unlearner import Unlearner
def reset_batch_norm(layer: nn.Module):
    if isinstance(layer, nn.BatchNorm2d):
        layer.reset_parameters()

class NoisyUnlearner(Unlearner):
    def unlearn(self, retain_set: DataLoader, forget_set: DataLoader, retrain_epochs: int = 4, nu : float = 1.) -> Module:
        self.og_model.eval()
        student = deepcopy(self.og_model)
        student.train()

        student.apply(reset_batch_norm)
        
        # Impair step
        for param in student.parameters():
            std = param.data.std(dim=[0,1,2] if  param.data.ndim == 4 else None)
            param.data += torch.randn_like(param.data) * std * nu
        
        
        self.optimizer = torch.optim.Adam(student.parameters(), lr=self.lr,weight_decay=1e-2)            
        self.erased_model = deepcopy(student)
        for e in tqdm(range(retrain_epochs)):
            # Retrain
            self.knowledge_transfer(student, self.og_model, True, retain_set, e)
            
        self.retrained_model = student
        return self.retrained_model


