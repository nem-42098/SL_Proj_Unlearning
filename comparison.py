from tools.Unlearner import Unlearner
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from types import ClassMethodDescriptorType
from torchvision.datasets.inaturalist import DATASET_MD5
import time
import pandas as pd
import seaborn as sns
import pickle

def train_test_dataloaders(data):

  """
  This function outputs train and test datasets, and train 
  and set dataloaders for both CIFAR10 and CIFAR100 datasets.
  """
  if data == "CIFAR10":
         ### Transformation
      transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(
     (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
      train_dataset = torchvision.datasets.CIFAR10(
        root= './data', train = True,
        download =True, transform = transform)
      test_dataset = torchvision.datasets.CIFAR10(
        root= './data', train = False,
        download =True, transform = transform)

  elif data == "CIFAR100":
      transform = transforms.Compose([transforms.ToTensor(),
      transforms.Normalize(
     (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))])
      train_dataset = torchvision.datasets.CIFAR100(
        root= './data', train = True,
        download =True, transform = transform)
      test_dataset = torchvision.datasets.CIFAR100(
        root= './data', train = False,
        download =True, transform = transform)
    # DATALOADER
  trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
  return train_dataset, test_dataset, trainloader, testloader



def forget_retain_dataloaders(train_dataset,list_classes):

  """
  We create forget and retain datasets for both CIFAR10 
  and CIFAR100 datasets, for an scenario forgetting 1 class and another forgetting N.
  """
  #Class split
  classes = ['forget', 'retain']
  class_datasets = {class_name: [] for class_name in classes}
    # Iterate through the CIFAR-10 dataset and split it into class-specific subsets
  for image, label in train_dataset:
    if label in list_classes:
      class_datasets['forget'].append((image, label))
    else:
      class_datasets['retain'].append((image, label))
  # Data Loader
  retain_dataloader = torch.utils.data.DataLoader(class_datasets['retain'], batch_size=128, shuffle=True, num_workers=2)
  forget_dataloader = torch.utils.data.DataLoader(class_datasets['forget'], batch_size=128, shuffle=True, num_workers=2)

  return class_datasets , retain_dataloader , forget_dataloader


def test(model, dataloader):
  device=torch.device('cuda')
  tp, n = 0,0
  for X,y in dataloader:
    X,y = X.to(device), y.to(device)

    with torch.no_grad():
      y_pred = model(X)

    tp += (y_pred.argmax(axis=1) == y).sum().item()
    n  += y.size(0)
  return tp/n

"""
This function, given dataloader objects, a model and a number of
classes to forget, runs the unlearing and provides as an output a
dictionary containing accuracy on forget, retrain subsets and the
time of execution
"""

def metrics(data,model,classes):

  device=torch.device('cuda')
# Train and test datasets, and train and set dataloaders for both CIFAR10 and CIFAR100 datasets.
  train_dataset, test_dataset, trainloader , testloader = train_test_dataloaders(data)

 # Split between Forget and Retain
  class_dataset , retain_dataloader , forget_dataloader = forget_retain_dataloaders(train_dataset,classes)

  # Load model into GPU
  model = torch.hub.load("chenyaofo/pytorch-cifar-models", model, pretrained=True)
  model=model.to(device)

  # Initialize Unlearning Class
  unlearner = Unlearner(model, lr = 1e-2, alpha=1)

  # Measure training time
  start_time = time.time()

  # Start Unlearning
  unlearn_model = unlearner.unlearn(retain_dataloader, forget_dataloader, forget_epochs=20)

  # Store time and accuracy of the unlearning
  time_i = (time.time() - start_time)
  datasets = [forget_dataloader,retain_dataloader,testloader]
  models = [unlearn_model,model]
  datasets_names = ["forget_dataloader","retain_dataloader","testloader"]
  models_names = ["unlearn_model","model"]

  metric = {(z+"_"+v): test(models[j],datasets[i]) for i,v in enumerate(datasets_names) for j,z in enumerate(models_names)}
  metric.update({"unlearn_model_time": time_i})

  return metric
