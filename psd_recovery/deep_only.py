import sys
sys.path.append('/home/sagar/Projects/radio_map_deep_prior/deep_prior')
from networks.adversarial_model import EncoderDecoder
from networks.ae import Autoencoder, AutoencoderSelu
import torch
import numpy as np
import scipy.io
import torch.nn as nn
from networks.gan import Generator, Generator256, Generator512
import os

lr = 0.01
loop_count = 10
criterion = nn.MSELoss()

# PATH = '/home/sagar/Projects/deep_completion/deep_slf/trained-models/l1_6_unnorm_raw_map_rand_samp.model'
# full_map_model = EncoderDecoder()

PATH = '/home/sagar/Projects/radio_map_deep_prior/deep_prior/trained-models/ae/l1_map_genmodel_selu2.model'
full_map_model = AutoencoderSelu()

PATH_SERVER = '/nfs/stak/users/shressag/sagar/deep_completion/deep_slf/models/full_data_models/l1_6_unnorm_raw_map_rand_samp.model'
try:
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
except:
    checkpoint = torch.load(PATH_SERVER, map_location=torch.device('cpu'))
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
    ROOT = '/home/sagar/Projects/radio_map_deep_prior/psd_recovery/data'
    BASE = '/home/sagar/Projects/radio_map_deep_prior/deep_prior'

    X = scipy.io.loadmat(os.path.join(ROOT,'T.mat'))['T']
    W = scipy.io.loadmat(os.path.join(ROOT,'Om.mat'))['Om']
    full_map = model(X,W)
    print(cost_func(X,full_map))

    pass