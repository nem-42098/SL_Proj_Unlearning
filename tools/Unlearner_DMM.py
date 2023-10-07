from copy import deepcopy
from tqdm import tqdm
import torch
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader


class UnlearnerDMM:
    def __init__(self, model: Module, device: str = 'cuda', lr: float = 1e-6, alpha: float =0.5) :
        self.combined_model = model
        self.device = device
        self.lr = lr
        self.alpha = alpha
        self.log = []
        self.forget_model = None
        self.retained_Model = None
        self.criterion = CrossEntropyLoss(reduction='mean')
        self.optimizer = None
        self.epoch_log = []

    def forget_learn(self,forget_dataloader:DataLoader,forget_epochs:int=10):
                
            ### Training the forget model to get the weight: Mean estimates of the Gaussian Distribution for the Task A(Forget Dataset)

            ### Initialise the forget model
            self.forget_model=self.reset_weights(self.combined_model)

            ### intialising the optimiser
            self.optimizer=torch.optim.Adam(self.forget_model.parameters(), lr=self.lr)
            ### early stopping counter
            stop_counter=0
           
            ###Iterations
            for epoch in range(forget_epochs):
                ### iterate over the forget_dataloader
                ### batch _loss
                loss_epoch=0
                
                for i, (inputs, targets) in enumerate(forget_dataloader):
                    ### GPU push
                    inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                    self.forget_model=self.forget_model.to(self.device)

                    ### predictions from the model
                    y_pred=self.forget_model(inputs)

                
                    self.optimizer.zero_grad()

                    ###Calculate the Loss
             
                    loss = self.criterion(y_pred,targets)
                
                    ## Backpropogate the Loss                   
                    loss.backward()
                    ### Update the weights
                    self.optimizer.step()
                    ### Logging the measures
                    self.log_performance(y_pred, targets, loss.item(), epoch, i, phase='Training_forget_model')

                    #### Train-set loss
                    loss_epoch+=loss.item()
                

                
                loss_epoch/=(i+1)


                ### Check for early stopping: if the decrease in the loss is less than 1e-3 for straight 5 iterations
                if (self.epoch_log[-1]-loss_epoch)<1e-3:
                     stop_counter+=1
                     self.epoch_log.append(loss_epoch)
                else:
                     stop_counter=0
                     self.epoch_log.append(loss_epoch)

                if stop_counter==5:
                     break

    def log_performance(self, y_pred: torch.Tensor, target: torch.Tensor, loss: float, epoch: int, batch: int, phase: str) -> None:
        tp = (y_pred.argmax(axis=1) == target).sum().item()
        n = target.size(0)
        self.log.append((phase, epoch, batch, tp, n, loss))


    def reset_weights(self, model: Module) -> Module:
        dumb_model = deepcopy(model)
        dumb_model = dumb_model.apply(Unlearner_DMM.init_weights)
        return dumb_model.to(self.device)
    

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
            nn.init.xavier_normal_(layer.weight,1)    
    
         


    def unlearn(self):
        ### Initialse the model for the retrained dataset
        self.retained_Model=self.reset_weights(self.combined_model)

        for i,(name,parameters) in enumerate(self.combined_model.named_parameters()):
            
            name=name.split('.')
            ### Feature transformation

            if name[0]=='features':

                if name[2]=='weight':
                ### computing the weights
                    self.retained_model.features[eval(name[1])].weight=parameters*(1/(1-self.alpha))-(self.alpha/(1-self.alpha))*self.forget_model_Model.features[eval(name[1])].weight
                else:
                ### Computing the bias
                    self.retained_model.features[eval(name[1])].bias=parameters*(1/(1-self.alpha))-(self.alpha/(1-self.alpha))*self.forget_model_Model.features[eval(name[1])].bias

            ### Classifier
            elif name[0]=='classifier':

                if name[2]=='weight':

                    self.retained_model.classifier[eval(name[1])].weight=parameters*(1/(1-self.alpha))-(self.alpha/(1-self.alpha))*self.forget_model_Model.classifier[eval(name[1])].weight
                else:
                
                    self.retained_model.classifier[eval(name[1])].bias=parameters*(1/(1-self.alpha))-(self.alpha/(1-self.alpha))*self.forget_model_Model.classifier[eval(name
                                                                                                                                                                         
                                                                                                                                                                         [1])].bias