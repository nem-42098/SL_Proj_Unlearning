from copy import deepcopy
import torch.nn as nn
import torch
from torch.nn.functional import softmax
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from itertools import chain
import numpy as np
import os
from typing import Type
from dataclasses import dataclass


@dataclass(frozen=True)
class Layer:
    """Keep information about a layer"""
    kind: Type
    in_channels: int
    out_channels: int
    idx: int = 0
    is_bias: bool = False
    is_running_mean: bool = False
    is_running_var: bool = False
    is_num_batches_tracked: bool = False

    def __post_init__(self):
        assert self.is_bias + self.is_running_mean + \
            self.is_running_var + self.is_num_batches_tracked <= 1
        # assert self.kind in ['Conv2d', 'ConvT2d', 'BatchNorm2D', 'Linear']


class Unlearner_FM(Module):

    def __init__(self, Removal_Ratio: float, Pretrained_Model: Module, lr: float = 1e-3, device: str = 'cuda',):
        super(Unlearner_FM, self).__init__()
        # Removal Ratio of the paramters in the network
        self.removal = Removal_Ratio
        # Pretrain_model
        self.model = Pretrained_Model
        # Learning Rate in case of fine tuning the network after masking the parameters
        self.lr = lr
        ###
        self.device = device
        # Hessian model wrt  retain dataset
        self.retain_hess = None
        # Hessian Model wrt to forget Dataset
        self.forget_hess = None
        # logging fine tuning performance
        self.log = []

    @staticmethod
    def get_named_layers(net, is_state_dict=True):
        """
        Convert the Model structure into list of string whose individual element is of the form:
            "layer_name"+ "param_type"{weight or bias or running_mean or running_var}
        """
        named_layers = []
        # recursive iteration : [More efficeint than using mutiple model.named_children for loops]
        for mod in net.modules():
            if hasattr(mod, 'in_channels') and hasattr(mod, 'out_channels'):
                param = Layer(type(mod), mod.in_channels, mod.out_channels)
                named_layers.append(param)
            elif hasattr(mod, 'num_features'):
                param = Layer(type(mod), mod.num_features, mod.num_features)
            named_layers.append(param)

            if hasattr(mod, 'bias') and mod.bias is not None:
                named_layers.append(param.replace(is_bias=True))

            if is_state_dict:
                named_layers.append(param.replace(is_running_var=True))
                named_layers.append(param.replace(is_num_batches_tracked=True))
                named_layers.append(param.replace(is_running_mean=True))

        return named_layers

    def load_hessian(self, path: str, dataloader) -> nn.Module:
        if path is None or not os.path.exists(path):
            hess = Unlearner_FM.Hessian(
                dataloader, self.model, self.device)
            torch.save(hess.state_dict(), path)
        else:
            hess_state_dict = torch.load(path)
            hess = deepcopy(self.model)
            hess.load_state_dict(hess_state_dict)

        return hess

    def Fisher_Masking(self, retain_loader: DataLoader, forget_loader: DataLoader, forget_hess_path: str, retain_hess_path: str):

        # get the named layers
        named_layers = Unlearner_FM.get_named_layers(
            self.model, is_state_dict=False)

        # Only need to compute hessian once for given class removal
        self.forget_hess = self.load_hessian(forget_hess_path, forget_loader)

        # Keeping count of Masked Parameters
        count = []

        # We will cover two cases of unlearning: Retain Data Available and Not
        # get Hessain wrt to Retain dataloader
        if retain_loader is not None:
            self.retain_hess = self.load_hessian(
                retain_hess_path, retain_loader)
        else:
            # placeholder
            self.retain_hess = self.forget_hess

        for layer, (_, param1), (_, param2) in zip(named_layers, self.forget_hess.named_parameters(), self.retain_hess.named_parameters()):

            if not layer.is_bias and layer.kind in [nn.Conv2d, nn.Linear]:
                # Difference between the hessian diff between the two dataloader. Hessain expectation gives us FIM(fisher Information matrix)

                fisher_diff = param1.data
                if retain_loader is not None:
                    fisher_diff -= param2.data

                if layer.kind is nn.Conv2d:
                    # total number of parameters in the kernel
                    total_size = fisher_diff.size()[-1]*fisher_diff.size()[-2]

                    # Average contributions of fisher information by the kernel channels
                    fisher_diff = torch.sum(
                        fisher_diff, dim=[-1, -2])/total_size

                # 1D flatten
                fisher_diff = fisher_diff.view(-1).cpu().detach().numpy()

                count.append(fisher_diff)

        # Converting the list of list tp => list
        count = list(chain.from_iterable(count))

        mask_index = np.argsort(np.array(count))

        # Getting the index of the kernels channels for Convolution layers or param in Linear layer which have
        # high contribution on the forget dataset and less on retrain dataset

        mask_index = mask_index[-int(len(count)*self.removal):]

        print('Total Number of Kernels and Neurons:{}, Number of masked Paramters:{}'.format(
            len(count), int(len(count)*self.removal)))

        # sorting the index of the param in ascending order'
        mask_index.sort()

        named_layers = Unlearner_FM.get_named_layers(self.model)
        state_dict = deepcopy(self.model.state_dict())

        num, idx = 0, 0
        for layer, (k, v) in zip(named_layers, state_dict.items()):
            if layer.kind in [nn.Conv2d, nn.Linear] and not layer.is_bias:
                while idx < len(mask_index) and mask_index[idx] < num + v.size()[0]*v.size()[1]:
                    # get the filter number
                    row = (mask_index[idx] - num) // v.size()[1]
                    # Channel number
                    col = (mask_index[idx] - num) % v.size()[1]

                    if layer.kind is nn.Conv2d:
                        state_dict[k][row, col, :, :] = 0.0
                    elif layer.kind is nn.Linear:
                        state_dict[k][row, col] = 0.0
                    idx += 1
                if num < len(count):
                    num += v.size()[0]*v.size()[1]

        assert num == len(count)
        new_model = deepcopy(self.model)
        new_model.load_state_dict(state_dict)
        new_model.cuda()
        return new_model, mask_index, len(count)

    @staticmethod
    def Hessian(dataloader: DataLoader, model: Module, device: str):
        """""
          Compute the Diagonal of the Hessian Matrix for the given Dataloader. 
          Note: In the implementation of the Diagonal of the Hessian Matrix we have weighted the second order derivatives by the 
                confidence of the samples wrt to the class.The significance of multiplying the squared gradient by the probability for 
                that class lies in estimating the curvature of the loss function with respect to the parameters. The Hessian matrix 
                represents the second-order derivatives of the loss function with respect to the parameters and provides information 
                about the local geometric properties of the loss surface.

                In the context of training a model for classification tasks, the probability for a particular class in the mini-batch 
                measures the likelihood of that class being the correct label. By multiplying the squared gradient by this probability
                , the algorithm assigns a higher weight to gradients associated with classes that are more likely to be correct. This 
                weighting reflects the importance of each class in determining the curvature of the loss function.

                Remember while doing batch gradinet we approximate the gradient.

        """""
        
        model = model.to(device)

        # Model in eval mode:
        model.eval()

        # Criterion of the Loss
        criterion = CrossEntropyLoss(reduction='mean')

        # Creating a attribute for the model paramters which store the the second order derivative
        for param in model.parameters():
            param.grad2 = 0

        # Iterating over the dataloader
        for _, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)

            # logits from the model
            model = model.to(device)
            output = model(data)
            # Convert to prob
            prob = softmax(output, dim=1)

            # Contribution of the paramters gradients wrt to each class computed sequentially weight by the
            # confidence in prediction
            for y in range(output.shape[1]):
                class_target = torch.empty_like(targets).fill_(y)
                loss = criterion(prob, class_target)

                model.zero_grad()
                loss.backward(retain_graph=True)
                model.to('cpu')
                for param in model.parameters():

                    # those paramters only which required gradinet
                    if param.requires_grad:
                        # weighted by the confidence in prediction for that class
                        # Since we only account for the Diag
                        param.grad2 += prob[:, y].float().mean().detach().cpu() * \
                            param.grad.data.pow(2)
                model.to(device)

        model.zero_grad()
        model.to('cpu')
        # Copy of the model having Diag(hessian) with respect to the given dataloader
        model_hessian = deepcopy(model)

        # Averaging the COmpute Diag with the size of the Dataloader

        for param_hess, param in zip(model_hessian.parameters(), model.parameters()):
            param.grad2 /= len(dataloader)
            # the parameters of the model_hessian have hessian stored
            param_hess.data = param.grad2

        print('Finished Computing Hessian Diagonal')

        return model_hessian

    @staticmethod
    def test(model, dataloader, device):
        tp, n = 0, 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            with torch.no_grad():
                y_pred = model(X)

            tp += (y_pred.argmax(axis=1) == y).sum().item()
            n += y.size(0)

        return tp/n

    def fine_tune(self, model, dataloader, epochs: int = 5):

        # intialising the optimiser
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=0.01)
        # early stopping counter
        stop_counter = 0
        # criterion
        criterion = CrossEntropyLoss()

        # Epoch Log of losses
        epoch_log = []

        # Iterations
        for epoch in range(epochs):
            # iterate over the forget_dataloader
            # batch _loss
            loss_epoch = 0

            for i, (inputs, targets) in enumerate(dataloader):

                # GPU push
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                if next(model.parameters()).is_cuda:
                    print("Model is on CUDA (GPU)")
                else:
                    model = model.to(self.device)

                # predictions from the model
                y_pred = model(inputs)

                self.optimizer.zero_grad()

                # Calculate the Loss

                loss = criterion(y_pred, targets)

                # Backpropogate the Loss
                loss.backward()
                # Update the weights
                optimizer.step()
                # Logging the measures
                self.log_performance(y_pred, targets, loss.item(
                ), epoch, i, phase='Training_forget_model')

                # Train-set loss
                loss_epoch += loss.item()

            loss_epoch /= (i + 1)

            # Check for early stopping: if the decrease in the loss is less than 1e-3 for straight 5 iterations
            epoch_log.append(loss_epoch)
            if epoch > 0:
                if (self.epoch_log[-1] - loss_epoch) < 1e-3:
                    stop_counter += 1

                else:
                    stop_counter = 0

            if stop_counter == 5:

                break

        return model, epoch_log

    def log_performance(self, y_pred: torch.Tensor, target: torch.Tensor, loss: float, epoch: int, batch: int,
                        phase: str) -> None:
        tp = (y_pred.argmax(axis=1) == target).sum().item()
        n = target.size(0)
        self.log.append((phase, epoch, batch, tp, n, loss))