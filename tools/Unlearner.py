from copy import deepcopy
from tqdm import tqdm
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from .Custom_Loss import ReconstructionLoss


class Unlearner:
    def __init__(self, model: Module, device: str = 'cuda', lr: float = 1e-6, alpha: float = 1):
        self.og_model = model
        self.device = device
        self.lr = lr
        self.alpha = alpha
        self.log = []
        self.dumb_model = None
        self.erased_model = None
        self.retrained_model = None
        self.criterion = ReconstructionLoss(alpha=self.alpha)
        self.optimizer = None

    def unlearn(self, retain_set: DataLoader, forget_set: DataLoader, forget_epochs: int = 4, retrain_epochs: int = 10) -> Module:
        # Create stochastic network, completely at random
        self.dumb_model = self.reset_weights(self.og_model)

        # # Erase knowledge about forget set from original model
        # print('Erasing information from original model...')
        # self.erased_model = self.knowledge_transfer(
        #     self.og_model, self.dumb_model,
        #     include_target=False,
        #     trainset=forget_set,
        #     epochs=forget_epochs)

        # # Retrain erased model on remaning set
        # print('Retraining on the retrain set...')
        # self.retrained_model = self.knowledge_transfer(
        #     self.erased_model, self.og_model,
        #     include_target=True,
        #     trainset=retain_set,
        #     epochs=retrain_epochs)
        self.og_model.eval()
        self.dumb_model.eval()
        student = deepcopy(self.og_model)
        student.train()
        self.optimizer = torch.optim.Adam(student.parameters(), lr=self.lr)
        
        for e in tqdm(forget_epochs):
            # Erasure
            self.knowledge_transfer(student, self.dumb_model, False, forget_set, e)
            # Retrain
            self.knowledge_transfer(student, self.og_model, False, retain_set, e)
            
        self.retrained_model = student
        return self.retrained_model

    def knowledge_transfer(self,
            student: Module, teacher: Module,
            include_target: bool,
            trainset: DataLoader, epoch:int) -> Module:

        for i, (inputs, targets) in enumerate(trainset):
            inputs, targets = inputs.to(
                self.device), targets.to(self.device)

            y_pred = student(inputs)

            with torch.no_grad():
                y_teach = teacher(inputs)

            self.optimizer.zero_grad()
            if include_target:
                # in case of retraining
                loss = self.criterion(y_pred, y_teach, targets)
            else:
                # in case of erasure
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
