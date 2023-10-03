from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import DataLoader
from .Custom_Loss import ReconstructionLoss

class Unlearner:
    def __init__(self, model : Module, device: str = 'cuda'):
        self.og_model = model
        self.device = device
        self.log = []
        self.dumb_model = None
        self.erased_model = None
        self.retrained_model = None
        
    def unlearn(self, retain_set:DataLoader, forget_set:DataLoader, forget_epochs:int=4, retrain_epochs:int=10) -> Module:
        # Create stochastic network, completely at random
        self.dumb_model = self.reset_weights(self.og_model)
        
        # Erase knowledge about forget set from original model
        print('Erasing information from original model...')
        self.erased_model = self.knowledge_transfer(
            self.og_model, self.dumb_model,
            include_target = False,
            trainset = forget_set,
            epochs = forget_epochs)
        
        # Retrain erased model on remaning set
        print('Retraining on the retrain set...')
        self.retrained_model = self.knowledge_transfer(
            self.erased_model, self.og_model,
            include_target = True,
            trainset = retain_set,
            epochs = retrain_epochs)
        
        
        return self.retrained_model
    
    def knowledge_transfer(self, 
            student: Module, teacher: Module,
            include_target: bool,
            trainset: DataLoader, 
            epochs:int = 10) -> Module:
        
        teacher.eval()
        student = deepcopy(student)
        student.train()
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-6)
        criterion = ReconstructionLoss()
                    
        for e in tqdm(range(epochs)):
            for i,(inputs,targets) in enumerate(trainset):
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                y_pred = student(inputs)

                with torch.no_grad():
                    y_teach = teacher(inputs)

                optimizer.zero_grad()
                if include_target:
                    loss = criterion(y_pred, y_teach, targets)  # in case of retraining
                else:
                    loss = criterion(y_pred, y_teach)           # in case of erasure
                loss.backward()
                optimizer.step()
                
                self.log_performance(y_pred, targets, loss.item(), e, i)
        return student
            
    def reset_weights(self, model: Module) -> Module:
        dumb_model = deepcopy(model)
        dumb_model = dumb_model.apply(Unlearner.init_weights)
        return dumb_model.to(self.device)
    
    def log_performance(self, y_pred:torch.Tensor, target:torch.Tensor, loss:float, epoch:int, batch:int) -> None:
        tp = torch.sum(torch.argmax(y_pred,axis=1)==target).item()
        n  = target.size(0)
        self.log.append((epoch, batch, tp, n, loss))

    @staticmethod
    def init_weights(layer: Module) -> None:
        reset_parameters = getattr(layer, "reset_parameters", None)
        if callable(reset_parameters):
            layer.reset_parameters()
            

