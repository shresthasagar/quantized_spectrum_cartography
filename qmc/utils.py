from slf_dataset import SLFDataset1bit, DEFAULT_ROOT
import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.io
import torch.nn as nn
import os
import sys

QUANTIZATION_BOUNDARIES_8_BINS_SAMPLE =  [0.0, 3.219041422308777e-10, 6.34243551758118e-05, 0.0001823223865358159, 0.00036289551644586027, 0.0006664704997092485, 0.0012639077613130212, 0.00301913358271122, 0.3312782347202301]

SD_8_BINS_SAMPLE = 3.219041422308777e-10

QUANTIZATION_BOUNDARIES_16_BINS = [0.0, 8.944017748646615e-10, 2.3812383005861193e-05, 6.808515900047496e-05, 0.00012131989933550358, 0.00018234866729471833, 0.00025588355492800474, 0.00034619917278178036, 0.0004588317824527621, 0.0006049227667972445, 0.0007961964583955705, 0.0010579598601907492, 0.001441714819520712, 0.0020772861316800117, 0.003326504724100232, 0.006930550094693899, 0.27432483434677124]
SD_16_BINS = 8.944017748646615e-10

QUANTIZATION_BOUNDARIES_8_BINS_UNIFORM = torch.arange(9)*0.3312/8
SD_8_BINS_UNIFORM = QUANTIZATION_BOUNDARIES_8_BINS_UNIFORM[1] - QUANTIZATION_BOUNDARIES_8_BINS_UNIFORM[0]

QUANTIZATION_BOUNDARIES_16_BINS_UNIFORM = torch.arange(17)*0.3312/16
SD_16_BINS_UNIFORM = QUANTIZATION_BOUNDARIES_16_BINS_UNIFORM[1] - QUANTIZATION_BOUNDARIES_16_BINS_UNIFORM[0]

QUANTIZATION_BOUNDARIES_256_BINS_UNIFORM = torch.arange(257)*0.3312/256
SD_256_BINS_UNIFORM = QUANTIZATION_BOUNDARIES_256_BINS_UNIFORM[1] - QUANTIZATION_BOUNDARIES_256_BINS_UNIFORM[0]

QUANTIZATION_BOUNDARIES_8_BINS_LOG = [-23.025850296020508, -23.000225067138672, -9.472214698791504, -8.490324974060059, -7.831082344055176, -7.240789890289307, -6.61128044128418, -5.762726783752441, -1.2379993200302124]
SD_8_BINS_LOG = 0.0256

QUANTIZATION_BOUNDARIES_7_BINS_LOG = [-23.025850296020508, -9.472214698791504, -8.490324974060059, -7.831082344055176, -7.240789890289307, -6.61128044128418, -5.762726783752441, -1.2379993200302124]
QUANTIZATION_BOUNDARIES_4_BINS =   [-23.025850296020508, -10.002398490905762, -7.980128765106201, -6.692554473876953, -1.0331487655639648]
LOG_OFFSET_4 = 1e-10
SD_4_BINS = 1.287

QUANTIZATION_BOUNDARIES_7_ADJUSTED = [-10.69232977, -9.35950321, -8.49230102, -7.86067357, -7.27999497, -6.65573177, -5.7952887, -1.10472809]

QUANTIZATION_BOUNDARIES_16_ADJUSTED = [-15.25285591, -10.63537803,  -9.59126825,  -9.01512351,  -8.60828803,
                                        -8.26986013,  -7.96781035,  -7.68630929,  -7.41001714,  -7.13536627,
                                        -6.85118837,  -6.54175727,  -6.17657863,  -5.70576175,  -4.97178181,
                                        -1.29344148]

LOG_OFFSET_7_ADJUSTED = 2.27e-05
LOG_OFFSET_16_ADJUSTED = 2.3755e-07

# QUANTIZATION_BOUNDARIES_16_BINS_UNIFORM = torch.arange(9)*0.3312/8
# SD_16_BINS_UNIFORM = QUANTIZATION_BOUNDARIES_8_BINS_UNIFORM[1] - QUANTIZATION_BOUNDARIES_8_BINS_UNIFORM[0]


def _find_boundaries(samples, num_bins=4):
    data = samples.reshape(-1)
    data = data.sort().values
    num_points = len(data)
    num_points_per_bin = int(num_points/num_bins)
    num_boundaries = num_bins-1
    boundaries = [data[0].item()]
    data_count = 0
    for i in range(len(data)):
        data_count += 1
        if data_count > num_points_per_bin and data[i] > boundaries[-1]:
            boundaries.append(data[i].item())
            num_points_per_bin = int((len(data)-i)/(num_bins-len(boundaries)+1))
            data_count = 0
    if not len(boundaries) > num_bins:
        boundaries.append(data[-1].item())
    sd = min([boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)])
    return boundaries, sd

def get_boundaries_from_samples(task='sc', num_bins=8, num_samples=10000):
    """
    Computes the boundaries and standard deviation from data samples
    """
    if task == 'sc':
        MAP_ROOT = '/scratch/sagar/slf/train_set/map_set_torch_raw_unnormalized'
        train_set_map = SLFDataset1bit(root_dir=os.path.join(MAP_ROOT, 'slf_mat'), 
                            csv_file=os.path.join(MAP_ROOT, 'details.csv'), total_data=100000, sample_size=[0.1, 0.11])
        
        loader = torch.utils.data.DataLoader(train_set_map, batch_size=10000, shuffle=True)
        loader = iter(loader)
        batch = next(loader)
        boundaries, std_dev = _find_boundaries(batch[1], num_bins=num_bins)
        # boundaries, std_dev = _find_boundaries(torch.log(batch[1]+1e-10), num_bins=num_bins)
        return boundaries, std_dev

def plot_histogram_map_values():
    """
    Plot histogram of the pixel values of radio map from a sample of radio maps
    """
    MAP_ROOT = '/scratch/sagar/slf/train_set/map_set_torch_raw_unnormalized'
    train_set_map = SLFDataset1bit(root_dir=os.path.join(MAP_ROOT, 'slf_mat'), 
                        csv_file=os.path.join(MAP_ROOT, 'details.csv'), total_data=10000, sample_size=[0.1, 0.11])
    
    loader = torch.utils.data.DataLoader(train_set_map, batch_size=10000, shuffle=True)
    loader = iter(loader)
    data = next(loader)
    n_bins = 10
    bins = torch.arange(n_bins+1)*data.max()/n_bins

    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(data.numpy(), bins = bins)
    ax.set_xlabel('pixel value', fontsize=16) 
    ax.set_ylabel('count of the values in the given sample', fontsize=16)
    ax.set_title('Histogram of the radio map data', fontsize=16)
    plt.show()
    # fig.savefig('src/data/histogram.pdf')

if __name__ == "__main__":
    b, sd = get_boundaries_from_samples(num_bins=4, num_samples=10000)
    print('boundaries {}'.format(b))
    print('standard deviation: {}'.format(sd))