import torch 
import torch.nn as nn
from torch.nn import Module
from .Unlearner_FM import Unlearner_FM
from copy import deepcopy
import numpy as np

class Unlearner_SDD(Unlearner_FM):
    """https://arxiv.org/abs/2308.07707"""
    def __init__(self, Removal_Ratio: float, Pretrained_Model: Module, lr: float = 0.001, lamda: float = 1, device: str = 'cuda'):
        super().__init__(Removal_Ratio, Pretrained_Model, lr, device)
        self.lamda = lamda
    
    def change_params(self, masks, importances, named_layers) -> nn.Module:
        """Dampening of params"""
        new_model = deepcopy(self.model)
        
        importances = np.array(importances)
        importances = np.nan_to_num(importances)
        
        importance_cutoff = np.quantile(importances, 1-self.removal)
        
        with torch.no_grad():
            for layer, param, H in zip(named_layers, new_model.parameters(), masks):
                if layer.kind in [nn.Conv2d, nn.Linear] and not layer.is_bias:
                    mask = H > importance_cutoff
                    dampening = mask * H * self.lamda
                    dampening = dampening.clamp(max=1) + ~mask
                    print('Unchanged params:', (dampening==1).float().mean().item() )
                    for _ in range(param.ndim - dampening.ndim):
                    # Need to adjust mask to full param size 
                        dampening = dampening.unsqueeze(-1)
                    dampening = dampening.expand(param.size())

                    param.data = param * dampening
                
        return new_model
    
    
    def compute_importances(self, named_layers: list, compare_diff : bool = False) -> (list, list):
        importances = []    # List of importances values
        param_mask  = []    # List of Tensors: difference btween hessians
        
        for layer, hessian_f, hessian_ret in zip(named_layers, self.forget_hess.parameters(), self.retain_hess.parameters()):

            if layer.is_bias or layer.kind not in [nn.Conv2d, nn.Linear]:
                param_mask.append(None)
                continue
            
            # Difference between the hessian diff between the two dataloader. Hessain expectation gives us FIM(fisher Information matrix)
            if compare_diff:
                fisher_diff = hessian_f / hessian_ret
            else:
                fisher_diff = hessian_f

            if layer.kind is nn.Conv2d:
                # total number of parameters in the kernel
                kernel_size = fisher_diff.size(-1)*fisher_diff.size(-2)

                # Average contributions of fisher information by the kernel channels
                fisher_diff = fisher_diff.sum(dim=[-1, -2])/kernel_size

            # 1D flatten
            fisher_importances = fisher_diff.view(-1).cpu().detach().numpy().tolist()
            
            param_mask.append(fisher_diff)
            importances += fisher_importances
                
        return param_mask, importances