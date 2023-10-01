
import pickle
import torch

def read(file,mode='tensor'):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    if mode!='tensor':
        return dict
    else:
        dict[b'data']=torch.tensor(dict[b'data'])
        return dict