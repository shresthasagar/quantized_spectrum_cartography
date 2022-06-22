from slf_dataset import SLFDataset1bit, DEFAULT_ROOT
import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.io as sio
import torch.nn as nn
import os
import sys
from networks.gan import Generator, Generator256, Generator512


def load_data():
    data = sio.loadmat('../backup/data/onebitdata1')
    S = torch.from_numpy(data['S']).type(torch.float32)
    T = torch.from_numpy(data['T']).type(torch.float32)
    C = torch.from_numpy(data['C']).type(torch.float32)
    S_true = torch.from_numpy(data['S_true']).type(torch.float32)
    C_true = torch.from_numpy(data['C_true']).type(torch.float32)
    T_true = torch.from_numpy(data['T_true']).type(torch.float32)
    return S, C, T, S_true, C_true, T_true

def load_generator():
    generator = Generator256()
    GAN_PATH = '/home/sagar/Projects/matlab/quantized_spectrum_cartography/deep_prior/trained_models/gan/sngan11_256_unnorm'
    GAN_PATH_SERVER = '/scratch/sagar/Projects/matlab/quantized_spectrum_cartography/deep_prior/trained_models/gan/sngan11_256_unnorm'
    try:
        checkpoint = torch.load(GAN_PATH, map_location=torch.device('cpu'))
    except:
        checkpoint = torch.load(GAN_PATH_SERVER, map_location=torch.device('cpu'))
        
    generator.load_state_dict(checkpoint['g_model_state_dict'])
    generator.eval()
    return generator
    
def init_z(generator, slf_target, z_dimension=256):
    max_iter = 100
    z = torch.randn((1,z_dimension), dtype=torch.float32, )
    criterion = nn.MSELoss()
    
    # First select a good random vector
    min_criterion = 9999999
    for i in range(200):
        temp = torch.randn((1,z_dimension), dtype=torch.float32)
        slf_out = generator(temp)
       
        temp_criterion = criterion(slf_out, slf_target) 
        if  temp_criterion < min_criterion:
            z.data = temp.clone()
            min_criterion = temp_criterion
            print('min_first', min_criterion.item())

    for i in range(200):
        temp = 0.2*torch.randn((1,z_dimension), dtype=torch.float32) + z
        slf_out = generator(temp)
            
        temp_criterion = criterion(slf_out, slf_target) 
        if  temp_criterion < min_criterion:
            z.data = temp.clone()
            min_criterion = temp_criterion
            print('min_second', min_criterion.item())

    z.requires_grad = True
    optimizer = torch.optim.Adam([z], lr=0.01)
    
    num_iter = 0
    loss_prev = 999999
    loss_current = 999
    while num_iter < max_iter: #and (loss_prev - loss_current > loss_change_threshold):
        loss_prev = loss_current
        optimizer.zero_grad()
        
        gen_out = generator(z)
        
        loss = criterion(gen_out, slf_target)
        
        loss.backward()
        optimizer.step()

        loss_current = loss.item()
        num_iter += 1
    return z