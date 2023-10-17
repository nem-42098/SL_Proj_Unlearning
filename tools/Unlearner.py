from copy import deepcopy
from tqdm import tqdm
import torch
from torch.nn import Module
import torch.nn as nn
from torch.utils.data import DataLoader
from .Custom_Loss import ReconstructionLoss


class Unlearner:
    """https://arxiv.org/abs/2308.14322
    """
    def __init__(self, model: Module, device: str = 'cuda', lr: float = 1e-6, alpha: float = 1):
        self.og_model = model
        self.device = device
        self.lr = lr
        self.alpha = alpha
        self.log = []
        self.dumb_model = None # Unused
        self.erased_model = None
        self.retrained_model = None
        self.criterion = ReconstructionLoss(alpha=self.alpha)
        self.optimizer = None
        
        self.distr = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.]*10))

    def unlearn(self, retain_set: DataLoader, forget_set: DataLoader, forget_epochs: int = 10, retrain_epochs: int = 4) -> Module:
        # Create stochastic network, completely at random
        # self.dumb_model = self.reset_weights(self.og_model)

        self.og_model.eval()
        
        self.erased_model = self.impair(self.og_model, forget_set, forget_epochs)
        self.retrained_model = self.recover(self.erased_model, retain_set, retrain_epochs)

        return self.retrained_model
    
    def impair(self, model:nn.Module, forget_set:DataLoader, epochs:int) -> nn.Module:
        impaired = deepcopy(model)
        impaired.train()
        self.optimizer = torch.optim.Adam(impaired.parameters(), lr=self.lr)
        for e in tqdm(range(epochs)):
            # Erasure
            self.knowledge_transfer(impaired, None, False, forget_set, e)
            
        return impaired
    
    def recover(self, model:nn.Module, retain_set:DataLoader, epochs:int) -> nn.Module:
        recovering =  deepcopy(model)
        recovering.train()
        self.optimizer = torch.optim.Adam(recovering.parameters(), lr=self.lr)
        
        for e in tqdm(range(epochs)):
            # Retrain
            self.knowledge_transfer(recovering, self.og_model, True, retain_set, e)
        return recovering
            

    def knowledge_transfer(self,
            student: Module, teacher: Module,
            include_target: bool,
            trainset: DataLoader, epoch:int) -> Module:

        for i, (inputs, targets) in enumerate(trainset):
            inputs, targets = inputs.to(
                self.device), targets.to(self.device)

            y_pred = student(inputs)

            self.optimizer.zero_grad()
            if include_target:
                with torch.no_grad():
                    y_teach = teacher(inputs)
                # in case of retraining
                loss = self.criterion(y_pred, y_teach, targets)
            else:
                # in case of erasure
                y_teach = self.distr.sample((y_pred.size(0),)).to(self.device)
                loss = self.criterion(y_pred, y_teach)
            loss.backward()
            self.optimizer.step()

            self.log_performance(y_pred, targets, loss.item(), epoch, i, phase='retrain' if include_target else 'erasure')

    def reset_weights(self, model: Module) -> Module:
        dumb_model = deepcopy(model)
        dumb_model = dumb_model.apply(Unlearner.init_weights)
        return dumb_model.to(self.device)

    def log_performance(self, y_pred: torch.Tensor, target: torch.Tensor, loss: float, epoch: int, batch: int, phase: str) -> None:
        tp = (y_pred.argmax(axis=1) == target).sum().item()
        n = target.size(0)
        self.log.append((phase, epoch, batch, tp, n, loss))

    @staticmethod
    def init_weights(layer: Module) -> None:
        reset_parameters = getattr(layer, "reset_parameters", None)
        if callable(reset_parameters):
            layer.reset_parameters()
        if isinstance(layer, nn.BatchNorm2d):
            # Reset BatchNorm parameters (gamma and beta)
            nn.init.ones_(layer.weight)  # Adjust 'a' and 'b' as needed
            nn.init.zeros_(layer.bias)
            # Reset BatchNorm running statistics (mean and variance)
            layer.reset_running_stats()
            return

        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.zeros_(layer.bias)
        if hasattr(layer, 'weight'):
            nn.init.xavier_normal_(layer.weight,10)
