import sys
sys.path.append('/home/sagar/Projects/matlab/quantized_spectrum_cartography')
from networks.ae import AutoencoderSelu
import torch
import numpy as np
import scipy.io
import torch.nn as nn
import os


PATH = '/scratch/sagar/Projects/matlab/quantized_spectrum_cartography/deep_prior/trained_models/ae/1bit_map_1.model'
full_map_model = AutoencoderSelu()

checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
full_map_model.load_state_dict(checkpoint['model_state_dict'])
full_map_model.eval()

def cost_func(X, X_from_slf):
    return ((X - X_from_slf)**2).sum()

def model(X, W):
    X = torch.from_numpy(X).type(torch.float32)
    W = torch.from_numpy(W).type(torch.float32)
    K = X.shape[2]

    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wx = W.repeat(K,1,1,1)
    Wx[Wx<0.5] = 0
    Wx[Wx>=0.5] = 1

    X = X.permute(2,0,1)
    X = X.unsqueeze(dim=1)
    X_sampled = X*Wx
    test_map = torch.cat((Wx,X_sampled), dim=1)

    full_map = full_map_model(test_map)
    full_map = full_map.squeeze()
    full_map = full_map.permute(1,2,0)
    full_map = full_map.detach().numpy()

    return full_map.copy()

if __name__ == '__main__':
    # X = np.random.rand(51,51,64)
    # W = np.ones((51,51))
    # z = np.random.rand(51,51,5)
    # C = np.random.rand(64,5)
    # R = 5
    # a = run_descent(W,X,z,C,R)
    ROOT = '/home/sagar/Projects/matlab/radio_map_deep_prior/psd_recovery/data'
    BASE = '/home/sagar/Projects/matlab/radio_map_deep_prior/deep_prior'

    X = scipy.io.loadmat(os.path.join(ROOT,'T.mat'))['T']
    W = scipy.io.loadmat(os.path.join(ROOT,'Om.mat'))['Om']
    full_map = model(X,W)
    print(cost_func(X,full_map))

    pass