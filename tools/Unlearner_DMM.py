from copy import deepcopy
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader


class UnlearnerDMM:
    def __init__(self, model: Module,config:dict, device: str = 'cuda', lr: float = 1e-6, alpha: float = 0.5):
        self.combined_model = model
        self.device = device
        self.lr = lr
        self.alpha = alpha
        self.log = []
        self.forget_model = None
        self.retained_model = None
        self.criterion = CrossEntropyLoss(reduction='mean')
        self.optimizer = None
        self.epoch_log = []
        self.config=config
    def forget_learn(self, forget_dataloader: DataLoader,retain_dataloader:DataLoader):

        ### Training the forget model to get the weight: Mean estimates of the Gaussian Distribution for the Task A(Forget Dataset)

        ### Initialise the forget model
        self.forget_model = self.reset_weights(self.combined_model)

        ### intialising the optimiser
        optimizer =torch.optim.Adam(self.forget_model.parameters(),lr=1e-3,weight_decay=0.0001) #torch.optim.SGD(self.combined_model.parameters(), lr=self.config['optimizer'][2][1],weight_decay=self.config['optimizer'][-2][1],momentum=self.config['optimizer'][2][1],dampening=self.config['optimizer'][3][1])
        ### intialiasing the scheuler
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.config['scheduler'][1][1],eta_min=self.config['scheduler'][2][1])
        ### forget epochs
        forget_epochs=self.config['max_epochs']
        ### early stopping counter
        stop_counter = 0

        

        ###Iterations
        for epoch in tqdm(range(forget_epochs)):
            ### iterate over the forget_dataloader
            ### batch _loss
            loss_epoch = 0



            for i, (inputs, targets) in enumerate(forget_dataloader):

                
                ### GPU push
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                self.forget_model = self.forget_model.to(self.device)

                ### predictions from the model
                y_pred = self.forget_model(inputs)

                optimizer.zero_grad()

                ###Calculate the Loss

                loss = self.criterion(y_pred, targets)

                ## Backpropogate the Loss
                loss.backward()
                ### Update the weights
                optimizer.step()
                ### Logging the measures
                self.log_performance(y_pred, targets, loss.item(), epoch, i, phase='Training_forget_model')

                #### Train-set loss
                loss_epoch += loss.item()

            loss_epoch /= (i + 1)

            
            if epoch > 0:
                if (self.epoch_log[-1][0] - loss_epoch) < 1e-3:
                    stop_counter += 1

                else:
                    stop_counter = 0
            ### Check for early stopping: if the decrease in the loss is less than 1e-3 for straight 5 iterations
            self.epoch_log.append([loss_epoch,UnlearnerDMM.test(self.forget_model,forget_dataloader,'cuda'),UnlearnerDMM.test(self.forget_model,retain_dataloader,'cuda')])
            print(self.epoch_log[-1])

            if stop_counter == 5:
                break

    def log_performance(self, y_pred: torch.Tensor, target: torch.Tensor, loss: float, epoch: int, batch: int,
                        phase: str) -> None:
        tp = (y_pred.argmax(axis=1) == target).sum().item()
        n = target.size(0)
        self.log.append((phase, epoch, batch, tp, n, loss))

    def reset_weights(self, model: Module) -> Module:
        dumb_model = deepcopy(model)
        dumb_model = dumb_model.apply(UnlearnerDMM.init_weights)
        return dumb_model.to(self.device)

    @staticmethod 
    def test(model, dataloader,device):
        tp, n = 0,0
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)

            with torch.no_grad():
                y_pred = model(X)

            tp += (y_pred.argmax(axis=1) == y).sum().item()
            n  += y.size(0)

        return tp/n
    @staticmethod
    def init_weights(layer: Module) -> None:
        """
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
            nn.init.xavier_normal_(layer.weight, 1)
        """
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0, 0.01)

    def unlearn(self, forget_dataloader: DataLoader, retain_dataloader:DataLoader,forget_epochs: int = 10):

        self.forget_learn(forget_dataloader,retain_dataloader)

        ## Initialize the model for the retrained dataset
        self.retained_model = self.reset_weights(self.combined_model)


        for key in self.retained_model.state_dict():

            self.retained_model.state_dict()[key]=(1/self.alpha)*self.combined_model.state_dict()[key]-((1-self.alpha)/self.alpha)*self.forget_model.state_dict()[key]

        # for i, (name, parameters) in enumerate(self.combined_model.named_parameters()):

        #     name = name.split('.')
        #     ### Feature transformation

        #     if name[0] == 'features':

        #         if name[2] == 'weight':
        #             ### computing the weights
        #             self.retained_model.features[eval(name[1])].weight = nn.Parameter(
        #                 parameters * (1 / (1 - self.alpha)) - (
        #                         self.alpha / (1 - self.alpha)) * self.forget_model.features[eval(name[1])].weight)
        #         else:
        #             ### Computing the bias
        #             self.retained_model.features[eval(name[1])].bias = nn.Parameter(
        #                 parameters * (1 / (1 - self.alpha)) - (
        #                         self.alpha / (1 - self.alpha)) * self.forget_model.features[eval(name[1])].bias)

        #     ### Classifier
        #     elif name[0] == 'classifier':

        #         if name[2] == 'weight':

        #             self.retained_model.classifier[eval(name[1])].weight = nn.Parameter(
        #                 parameters * (1 / (1 - self.alpha)) - (
        #                         self.alpha / (1 - self.alpha)) * self.forget_model.classifier[
        #                     eval(name[1])].weight)
        #         else:

        #             self.retained_model.classifier[eval(name[1])].bias = nn.Parameter(
        #                 parameters * (1 / (1 - self.alpha)) - (
        #                         self.alpha / (1 - self.alpha)) * self.forget_model.classifier[eval(name

        #                                                                                            [1])].bias)
