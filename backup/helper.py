import torch
import torch.nn as nn
import time
from scipy.stats import norm
from torch.distributions import normal

def F_sigmoid(y):
    """
    Evaluates sigmoid function applied to the input
    """
    return 1/(1+torch.exp(-y))
    
def dither_sigmoid(y):
    """
    Sample {+1, -1} values from the bernoulli distribution with parameter given by the sigmoid of the input 
    z ~ F(y)
    """
    F_y = F_sigmoid(y)
    return torch.bernoulli(F_y)

def F_probit(y, std):
    """
    Evaluates the probit function value for the given input
    """
    return (1/2)*(1 + torch.erf(y/(std*1.414213)))

def dither_probit(y, std):
    """
    Returns sample z ~ F(y) using the probit model for F with variance std
    """
    F_y = F_probit(y, std)
    return torch.bernoulli(F_y)

def outer(mat : torch.Tensor, vec : torch.Tensor):
    """
    Compute outer product given a matrix and a vector
    """
    prod = torch.zeros(( *vec.shape,*mat.shape), dtype=torch.float32)
    for i in range(len(vec)):
        prod[i,:,:] = mat*vec[i]
    return prod

def get_tensor(S: torch.Tensor, C: torch.Tensor):
    """
    Returns  sum_i (S[i] o C[i])
    """
    prod = 0
    for i in range(C.shape[0]):
        prod += outer(S[i,0,:,:], C[i,:])
    return prod

def NMSE(T: torch.Tensor, T_target: torch.Tensor):
    """
    Normalized mean squared error
    """
    return torch.norm(T-T_target, 'fro')/torch.norm(T_target, 'fro')


class NegLikelihood(torch.nn.Module):
    def __init__(self, mean, std=None, probit=True):
        super(NegLikelihood, self).__init__()

        self.mean = mean
        if probit:
            assert (std is not None)
        self.std = std
        self.probit = probit
        self.criterion = nn.BCELoss()

    def forward(self, T_sample, T_target):
        if self.probit:
            T_sample = F_probit(T_sample-self.mean, self.std)
        else:
            T_sample = F_sigmoid(T_sample-self.mean)
        return self.criterion(T_sample, T_target) 

class DeterministicCost(torch.nn.Module):
    """
    Find the deterministic cost with frobenius norm constraints
    - lambda * ((T-mean)* T_target).sum() + || T - mean ||_F
    """
    def __init__(self, mean=0):
        super(DeterministicCost, self).__init__()
        self.lambda_reg = 0.001
        self.mean = mean
        
    def forward(self, S, C, T_target):
        T_hat = get_tensor(S,C)
        T_hat = T_hat-self.mean
        quant_loss = - self.lambda_reg*((T_hat*T_target).sum()) + torch.norm(T_hat, 'fro')
        return quant_loss
