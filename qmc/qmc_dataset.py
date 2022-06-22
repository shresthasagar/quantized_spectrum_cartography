from torch.utils.data import Dataset
import torch
import os


class SLFDatasetQmc(Dataset):
    """
    QMC dataset: Provides quantized observation, observation location and ground truth for each data sample
    """

    def __init__(self, root_dir, csv_file=None, transform=None, total_data=None, sample_size=[0.01,0.40], fixed_size=None, fixed_mask=False, no_sampling=False):
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

