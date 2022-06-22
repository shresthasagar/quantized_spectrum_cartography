
from slf_dataset import SLFDataset1bit, DEFAULT_ROOT
import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.io
import torch.nn as nn
import os
import sys
from networks.gan import Generator, Generator256, Generator512


generator = Generator256()
GAN_PATH = '/home/sagar/Projects/matlab/quantized_spectrum_cartography/deep_prior/trained_models/gan/sngan11_256_unnorm'
GAN_PATH_SERVER = '/scratch/sagar/Projects/matlab/quantized_spectrum_cartography/deep_prior/trained_models/gan/sngan11_256_unnorm'

SLF_ROOT = '/scratch/sagar/slf/train_set/set_harsh_torch_raw_unnormalized'

train_set_slf = SLFDataset1bit(root_dir=os.path.join(SLF_ROOT, 'slf_mat'), 
                    csv_file=os.path.join(SLF_ROOT, 'details.csv'), total_data=10000, sample_size=[0.1, 0.11])


try:
    checkpoint = torch.load(GAN_PATH, map_location=torch.device('cpu'))
except:
    checkpoint = torch.load(GAN_PATH_SERVER, map_location=torch.device('cpu'))
    
generator.load_state_dict(checkpoint['g_model_state_dict'])
generator.eval()
z_dimension = 256