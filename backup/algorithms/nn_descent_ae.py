import sys
sys.path.append('/home/sagar/Projects/matlab/radio_map_deep_prior/deep_prior')
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

slf_network = EncoderDecoder()
PATH1 = '/home/sagar/Projects/deep_completion/deep_slf/trained-models/l1_5_unnorm_raw_rand_samp.model'
PATH1_SERVER = '/nfs/stak/users/shressag/sagar/deep_completion/deep_slf/models/full_data_models/l1_5_unnorm_raw_rand_samp.model'
try:
    checkpoint = torch.load(PATH1, map_location=torch.device('cpu'))
except:
    checkpoint = torch.load(PATH1_SERVER, map_location=torch.device('cpu'))
slf_network.load_state_dict(checkpoint['model_state_dict'])
slf_network.eval()

# autoencoder = Autoencoder()
# PATH_ae = '/home/sagar/Projects/matlab/radio_map_deep_prior/deep_prior/trained-models/ae/l1_ae_gen1_raw.model'
# PATHae_SERVER = '/nfs/stak/users/shressag/sagar/deep_completion/deep_slf/models/full_data_models/l1_5_unnorm_raw_rand_samp.model'
# try:
#     checkpoint = torch.load(PATH_ae, map_location=torch.device('cpu'))
# except:
#     checkpoint = torch.load(PATHae_SERVER, map_location=torch.device('cpu'))
# autoencoder.load_state_dict(checkpoint['model_state_dict'])
# autoencoder.eval()

autoencoder = AutoencoderSelu()
PATH_selu_ae = '/home/sagar/Projects/matlab/radio_map_deep_prior/deep_prior/trained-models/ae/l1_ae_gen4_selu_raw.model'
checkpoint = torch.load(PATH_selu_ae, map_location=torch.device('cpu'))
autoencoder.load_state_dict(checkpoint['model_state_dict'])
autoencoder.eval()

slf_network_log = EncoderDecoder()
PATH2 = '/home/sagar/Projects/matlab/radio_map_deep_prior/deep_prior/trained-models/ae/mse2_ed_rand_samp.model'
PATH2_SERVER = '/nfs/stak/users/shressag/sagar/deep_completion/deep_slf/models/full_data_models/mse2_ed_rand_samp.model'
try:
    checkpoint = torch.load(PATH2, map_location=torch.device('cpu'))
except:
    checkpoint = torch.load(PATH2_SERVER, map_location=torch.device('cpu'))
    
slf_network_log.load_state_dict(checkpoint['model_state_dict'])
slf_network_log.eval()




def outer(mat, vec):
    prod = torch.zeros(( *vec.shape,*mat.shape), dtype=torch.float32)
    for i in range(len(vec)):
        prod[i,:,:] = mat*vec[i]
    return prod

def get_tensor(S, C):
    prod = 0
    for i in range(C.shape[0]):
        prod += outer(S[i,:,:], C[i,:])
    return prod

def cost_func(X, X_from_slf, Wx):
    return (((Wx*X) - (Wx*X_from_slf))**2).sum()


def run_descent(W, X, z, C, R):
    """
    Arguments:
        W : Mask 
        X : sampled tensor
        z : current latent vectors estimate for R emitters
        C : current psd estimate

    Returns:
        the updated latent vector estimate
    """
    # Prepare data
    W = torch.from_numpy(W).type(torch.float32)
    X = torch.from_numpy(X).type(torch.float32)
    z = torch.from_numpy(z).type(torch.float32)
    C = torch.from_numpy(C).type(torch.float32)
    R = int(R)

    K = X.shape[2]

    X = X.permute(2,0,1)
    # z = z.permute(2,0,1)
    # z = z.unsqueeze(dim=1)
    C = C.permute(1,0)
    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wr = W.repeat(R,1,1,1)
    Wx = W.repeat(K,1,1,1)

    Wr[Wr<0.5] = 0
    Wr[Wr>=0.5] = 1


    # test_slf = torch.cat((Wr, z), dim=1)
    test_slf = z    
    test_slf.requires_grad = True
    
    slf_complete = slf_network(test_slf)
    X_from_slf = get_tensor(slf_complete[:,0,:,:], C)
    previous_loss = cost_func(X, X_from_slf, Wx)

    optimizer = torch.optim.Adam([test_slf], lr=0.01)
    i = 0
    for i in range(loop_count):
        i=i+1
        optimizer.zero_grad()
        slf_complete = slf_network(test_slf)
        # slf_complete = slf_complete.view(R,51,51)
        # reconstruct the map from slf 
        # first normalize slf
        # slf_complete_norm = torch.zeros((slf_complete.shape))
        # for rr in range(R):
        #     slf_complete[rr] = slf_complete[rr]/(slf_complete[rr].norm())
        X_from_slf = get_tensor(slf_complete[:,0,:,:], C)
        
        # loss = criterion(Wx*X, Wx*X_from_slf)
        loss = cost_func(X, X_from_slf, Wx)
        if i>5 and previous_loss - loss.item() < 1e-5:
            print('change in loss too small')
            break
        previous_loss = loss.item()
        print(loss)
        loss.backward()
        optimizer.step()
        # with torch.no_grad():
        #     test_slf -= lr*test_slf.grad.data
    slf_comp = slf_complete[:,0,:,:]
    slf_comp = slf_comp.permute(1,2,0)
    slf_comp = slf_comp.detach().numpy()

    return test_slf.detach().numpy().copy(), slf_comp.copy()

def model(z, W, R, Strue):
    z = torch.from_numpy(z).type(torch.float32)
    W = torch.from_numpy(W).type(torch.float32)
    S_true = torch.from_numpy(Strue).type(torch.float32)

    R = int(R)
    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wr = W.repeat(R,1,1,1)
    Wr[Wr<0.5] = 0
    Wr[Wr>=0.5] = 1
    z = z.permute(2,0,1)
    S_true = S_true.permute(2,0,1)

    S_true = S_true.unsqueeze(dim=1)
    z = z.unsqueeze(dim=1)


    test_slf = torch.cat((Wr, z), dim=1)
    slf_complete = slf_network(test_slf)

    loss = criterion(slf_complete, S_true)
    print('actual_loss for ae', loss)

    slf_comp = slf_complete[:,0,:,:]
    slf_comp = slf_comp.permute(1,2,0)
    slf_comp = slf_comp.detach().numpy()
    return test_slf.detach().numpy().copy(), slf_comp.copy()

def model_ae(z, W, R, Strue):
    z = torch.from_numpy(z).type(torch.float32)
    W = torch.from_numpy(W).type(torch.float32)
    S_true = torch.from_numpy(Strue).type(torch.float32)

    R = int(R)
    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wr = W.repeat(R,1,1,1)
    Wr[Wr<0.5] = 0
    Wr[Wr>=0.5] = 1
    z = z.permute(2,0,1)
    S_true = S_true.permute(2,0,1)

    S_true = S_true.unsqueeze(dim=1)
    z = z.unsqueeze(dim=1)


    test_slf = torch.cat((Wr, z), dim=1)
    slf_complete = autoencoder(test_slf)
    z = autoencoder.encoder(test_slf)
    
    loss = criterion(slf_complete, S_true)
    print('actual_loss for ae', loss)

    slf_comp = slf_complete[:,0,:,:]
    slf_comp = slf_comp.permute(1,2,0)
    slf_comp = slf_comp.detach().numpy()
    return z.detach().numpy().copy(), slf_comp.copy()


def model_log(S_tilde, W, R, S_true):
    S_tilde = torch.from_numpy(S_tilde).type(torch.float32)
    W = torch.from_numpy(W).type(torch.float32)
    S_true = torch.from_numpy(S_true).type(torch.float32)

    R = int(R)
    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wr = W.repeat(R,1,1,1)
    Wr[Wr<0.5] = 0
    Wr[Wr>=0.5] = 1

    S_true = S_true.permute(2,0,1)
    S_true_raw = S_true
    S_true[S_true<0] = 0
    S_true = torch.log(S_true.unsqueeze(dim=1) + 1e-16)
    S_true[S_true<-30] = 0

    S_tilde = S_tilde.permute(2,0,1)
    S_tilde[S_tilde<0] = 0
    S_tilde = torch.log(S_tilde.unsqueeze(dim=1)+1e-16)
    S_tilde[S_tilde<-30] = 0


    normalizer = []
    for i in range(R):
        normalizer.append(S_tilde[i].min().item())
        S_tilde[i] = S_tilde[i]/S_tilde[i].min()
        S_true[i] = S_true[i]/S_true[i].min()

    # save normalization
    a = torch.ones((R,1,51,51), dtype=torch.float32)
    for i in range(R):
        a[i] = a[i]*normalizer[i]

    test_slf = torch.cat((Wr, S_tilde), dim=1)
    slf_complete = slf_network_log(test_slf)

    slf_complete = torch.exp(slf_complete*a)

    loss = criterion(slf_complete.squeeze(), S_true_raw)
    print('actual_loss for ae log', loss)
    
    slf_comp = slf_complete[:,0,:,:]
    slf_comp = slf_comp.permute(1,2,0)

    slf_comp = slf_comp.detach().numpy()
    return slf_comp.copy()

def run_descent_ae(W, X, z, C, R, lambda_reg=1e-5):
    """
    Arguments:
        W : Mask 
        X : sampled tensor
        z : current latent vectors estimate for R emitters
        C : current psd estimate

    Returns:
        the updated latent vector estimate
    """
    # Prepare data
    W = torch.from_numpy(W).type(torch.float32)
    X = torch.from_numpy(X).type(torch.float32)
    z = torch.from_numpy(z).type(torch.float32)
    C = torch.from_numpy(C).type(torch.float32)
    R = int(R)

    K = X.shape[2]

    X = X.permute(2,0,1)
    # z = z.permute(2,0,1)
    # z = z.unsqueeze(dim=1)
    C = C.permute(1,0)
    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wr = W.repeat(R,1,1,1)
    Wx = W.repeat(K,1,1,1)

    Wr[Wr<0.5] = 0
    Wr[Wr>=0.5] = 1


    # test_slf = torch.cat((Wr, z), dim=1)
    test_slf = z    
    test_slf.requires_grad = True
    
    slf_complete = autoencoder.decoder(test_slf)
    X_from_slf = get_tensor(slf_complete[:,0,:,:], C)
    previous_loss = cost_func(X, X_from_slf, Wx)

    optimizer = torch.optim.Adam([test_slf], lr=0.01)
    i = 0
    for i in range(loop_count):
        i=i+1
        optimizer.zero_grad()
        slf_complete = autoencoder.decoder(test_slf)
        # slf_complete = slf_complete.view(R,51,51)
        # reconstruct the map from slf 
        # first normalize slf
        # slf_complete_norm = torch.zeros((slf_complete.shape))
        # for rr in range(R):
        #     slf_complete[rr] = slf_complete[rr]/(slf_complete[rr].norm())
        X_from_slf = get_tensor(slf_complete[:,0,:,:], C)
        
        # loss = criterion(Wx*X, Wx*X_from_slf)
        loss = cost_func(X, X_from_slf, Wx) + lambda_reg*torch.norm(test_slf)
        # if i>5 and previous_loss - loss.item() < 1e-5:
        #     print('change in loss too small')
        #     break
        previous_loss = loss.item()
        print(loss)
        loss.backward()
        optimizer.step()
        # with torch.no_grad():
        #     test_slf -= lr*test_slf.grad.data
    slf_comp = slf_complete[:,0,:,:]
    slf_comp = slf_comp.permute(1,2,0)
    slf_comp = slf_comp.detach().numpy()

    return test_slf.detach().numpy().copy(), slf_comp.copy()


if __name__ == '__main__':
    # X = np.random.rand(51,51,64)
    # W = np.ones((51,51))
    # z = np.random.rand(51,51,5)
    # C = np.random.rand(64,5)
    # R = 5
    # a = run_descent(W,X,z,C,R)
    ROOT = '/home/sagar/Projects/matlab/radio_map_deep_prior/psd_recovery/data'
    BASE = '/home/sagar/Projects/matlab/radio_map_deep_prior/deep_prior'

    samp = torch.load(os.path.join(BASE, 'samp'))
    true = torch.load(os.path.join(BASE, 'true'))
    S_tilde = samp[1]
    S_true = true
    S_tilde = S_tilde.unsqueeze(dim=0)
    S_true = S_true.unsqueeze(dim=0)

    # print(S_tilde.shape)
    S_tilde = S_tilde.permute(1,2,0).numpy()
    S_true  = true.permute(1,2,0).numpy()
    W = samp[0]
    W = W.numpy()

    X = scipy.io.loadmat(os.path.join(ROOT,'T.mat'))['T']
    C = scipy.io.loadmat(os.path.join(ROOT,'C.mat'))['C']
    W = scipy.io.loadmat(os.path.join(ROOT,'Om.mat'))['Om']
    # S_tilde = scipy.io.loadmat(os.path.join(ROOT,'S_omega.mat'))['S_omega']
    # S_true = scipy.io.loadmat(os.path.join(ROOT,'Sc.mat'))['S_true']
    # Strue_raw = scipy.io.loadmat(os.path.join(ROOT,'Sc.mat'))['S_true']
    R = C.shape[1]
    out = model(S_tilde, W, 1, S_true)
    
    pass