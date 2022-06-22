import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import os
import scipy.io
import numpy as np
import random
import time
import matplotlib.pyplot as plt

mean_slf = 0.0045
std_slf = 0.0191

DEFAULT_ROOT = '/scratch/sagar/slf/train_set/2m_data/slf_mat'
    
def plot_image(train_set, index, log=False):
    a = train_set[index]
    if not log:
        plt.imshow(a.detach().squeeze().numpy())
    else:
        plt.imshow(np.log(a.detach().squeeze().numpy()))
        

def plot_image_output(image, log=False):
    if log:
        plt.imshow(np.log(image.detach().squeeze().numpy()))
    else:
        plt.imshow(image.detach().squeeze().numpy())

class GANSample(Dataset):
    def __init__(self, generator_path = 'trained-models/gan/generator-gan-first',train=True, download=True, transform=None, total_data=None, sampling=False):
        self.generator = torch.load(generator_path)
        self.generator.eval()
        self.generator = self.generator.to('cpu')
        self.z_dim = next(self.generator.parameters()).shape[1]
        if not total_data is None:
            self.num_examples = total_data
        else:
            if train == True:
                self.num_examples = 50000
            else:
                self.num_examples = 2000
        
        self.sampling = sampling
        sample_size = [0.01,0.30]
        self.sampling_rate = sample_size[1] - sample_size[0]
        self.omega_start_point = 1.0 - sample_size[1]
        
    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        z = torch.randn((1,self.z_dim), dtype=torch.float32)
        with torch.no_grad():
            sample = self.generator(z)
        z = z.squeeze()
        sample = sample.squeeze(dim=0)
        if self.sampling:
            rand = self.sampling_rate*torch.rand(1).item()
            bool_mask = torch.FloatTensor(1,51,51).uniform_() > (self.omega_start_point+rand)
            int_mask = bool_mask*torch.ones((1,51,51), dtype=torch.float32)
            subsample = sample*bool_mask
            subsample.requires_grad = False
            z.requires_grad = False
            return subsample, z
        else:
            return sample, z


class SLFDataset(Dataset):
    """SLF loader"""

    def __init__(self, root_dir, csv_file=None, transform=None, total_data=None, normalize=True, sample_size=[0.01,0.20], fixed_size=None, fixed_mask=False, no_sampling=False):
        """
        Args:
            csv_file (string): Path to the csv file with params.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            total_data: Number of data points
            normalize: Whether to normalize such that the largest value is 1
            sample_size: range off sampling percentage
            fixed_size: if not none, fixed_size will be used as the sampling size
            fixed_mask: if true, the same mask will be used 
        """
        self.root_dir = root_dir
        self.transform = transform
        self.NUM_SAMPLES = int(0.20*51*51)
        self.nrow, self.ncol = (51, 51)
        if not total_data is None:
            self.num_examples = total_data
        else:
            self.num_examples = 500000
        self.sampling_rate = sample_size[1]-sample_size[0]
        self.omega_start_point = 1.0 - sample_size[1]
        
        if fixed_size:
            self.sampling_rate = 0
            self.omega_start_point = 1.0 - fixed_size
        
        self.fixed_mask = fixed_mask
        self.no_sampling = no_sampling
        if self.fixed_mask:
            rand = self.sampling_rate*torch.rand(1).item()
            self.bool_mask = torch.FloatTensor(1,51,51).uniform_() > (self.omega_start_point+rand)
            self.int_mask = self.bool_mask*torch.ones((1,51,51), dtype=torch.float32)
        
    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):

        filename = os.path.join(self.root_dir,
                                str(idx)+'.pt')
        sample = torch.load(filename)

        if self.no_sampling:
            return sample
        
        if not self.fixed_mask:
            rand = self.sampling_rate*torch.rand(1).item()
            bool_mask = torch.FloatTensor(1,51,51).uniform_() > (self.omega_start_point+rand)
            int_mask = bool_mask*torch.ones((1,51,51), dtype=torch.float32)
            sampled_slf = sample*bool_mask
        else:
            int_mask = self.int_mask
            sampled_slf = sample*self.bool_mask
        
        return torch.cat((int_mask,sampled_slf), dim=0), sample

class SLFDataset1bit(Dataset):
    """SLF loader"""

    def __init__(self, root_dir, csv_file=None, transform=None, total_data=None, normalize=True, sample_size=[0.01,0.40], fixed_size=None, fixed_mask=False, no_sampling=False):
        """
        Args:
            csv_file (string): Path to the csv file with params.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            total_data: Number of data points
            normalize: Whether to normalize such that the largest value is 1
            sample_size: range off sampling percentage
            fixed_size: if not none, fixed_size will be used as the sampling size
            fixed_mask: if true, the same mask will be used 
        """
        self.root_dir = root_dir
        self.transform = transform
        self.NUM_SAMPLES = int(0.20*51*51)
        self.nrow, self.ncol = (51, 51)
        if not total_data is None:
            self.num_examples = total_data
        else:
            self.num_examples = 500000
        self.sampling_rate = sample_size[1]-sample_size[0]
        self.omega_start_point = 1.0 - sample_size[1]
        
        if fixed_size:
            self.sampling_rate = 0
            self.omega_start_point = 1.0 - fixed_size
        
        self.fixed_mask = fixed_mask
        self.no_sampling = no_sampling
        if self.fixed_mask:
            rand = self.sampling_rate*torch.rand(1).item()
            self.bool_mask = torch.FloatTensor(1,self.nrow, self.ncol).uniform_() > (self.omega_start_point+rand)
            self.int_mask = self.bool_mask*torch.ones((1,self.nrow, self.ncol), dtype=torch.float32)
        
    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):

        filename = os.path.join(self.root_dir,
                                str(idx)+'.pt')
        ground_truth = torch.load(filename)

        sample = ground_truth.clone()
        sample[sample<mean_slf] = -1
        sample[sample!=-1] = 1
        if self.no_sampling:
            return sample
        
        if not self.fixed_mask:
            rand = self.sampling_rate*torch.rand(1).item()
            bool_mask = torch.FloatTensor(1,self.nrow, self.ncol).uniform_() > (self.omega_start_point+rand)
            int_mask = bool_mask*torch.ones((1,self.nrow, self.ncol), dtype=torch.float32)
            sampled_slf = sample*bool_mask
        else:
            int_mask = self.int_mask
            sampled_slf = sample*self.bool_mask
        


        return torch.cat((int_mask,sampled_slf), dim=0), ground_truth

