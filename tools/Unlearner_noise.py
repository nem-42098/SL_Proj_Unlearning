from copy import deepcopy
from tqdm import tqdm
import torch
from torch.nn import Module
import torch.nn as nn
from torch.utils.data import DataLoader
from.Unlearner import Unlearner


class NoisyUnlearner(Unlearner):
    def unlearn(self, retain_set: DataLoader, forget_set: DataLoader = None, retrain_epochs: int = 4, nu : float = 1.) -> Module:
        """Applies the unlearning through an impair step in which gaussian noise (mean=0, std=nu) is added to the
        model parameters, and then some retraining

        Args:
            retain_set (DataLoader): _description_
            forget_set (DataLoader, optional): Unused, here for compatibility reasons.
            retrain_epochs (int, optional): _description_. Defaults to 4.
            nu (float, optional): Standard deviation of the gaussian noise. Defaults to 1..

        Returns:
            Module: the new model which should've forgotten the old samples
        """
        self.og_model.eval()
        student = deepcopy(self.og_model)
        student.train()

        student.apply(NoisyUnlearner.reset_batch_norm)
        
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
    
    @staticmethod
    def reset_batch_norm(layer: nn.Module):
        if isinstance(layer, nn.BatchNorm2d):
            layer.reset_parameters()


