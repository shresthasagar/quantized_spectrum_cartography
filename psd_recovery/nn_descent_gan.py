import torch
import numpy as np
import scipy.io
import torch.nn as nn
import os
import sys
sys.path.append('/home/sagar/Projects/radio_map_deep_prior/deep_prior')
from networks.gan import Generator, Generator256, Generator512

loop_count = 50
z_dimension = 256
criterion = nn.MSELoss()
lambda_reg = 1e-4

generator = Generator256()
GAN_PATH = '/home/sagar/Projects/radio_map_deep_prior/deep_prior/trained-models/gan/sngan11_256_unnorm'
GAN_PATH_SERVER = '/scratch/sagar/Projects/radio_map_deep_prior/deep_prior/trained-models/gan/sngan7_256'

try:
    checkpoint = torch.load(GAN_PATH, map_location=torch.device('cpu'))
except:
    checkpoint = torch.load(GAN_PATH_SERVER, map_location=torch.device('cpu'))
    
generator.load_state_dict(checkpoint['g_model_state_dict'])
generator.eval()

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

def NAE(S, Sr):
    nae = 0
    for i in range(S.shape[2]):
        a = S[:,:,i]/abs(S[:,:,i]).sum() - Sr[:,:,i]/abs(Sr[:,:,i]).sum()
        nae += abs(a).sum()
    return nae

def optimize_z_raw(W, X, S_tilde, C, R, Z, lr):
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
    Z = torch.from_numpy(Z).type(torch.float32)
    S_tilde = torch.from_numpy(S_tilde).type(torch.float32)
    C = torch.from_numpy(C).type(torch.float32)
    R = int(R)
    lr = float(lr)

    K = X.shape[2]

    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wr = W.repeat(R,1,1,1)
    Wx = W.repeat(K,1,1,1)
    Wx = Wx.squeeze(dim=1)
    X = X.permute(2,0,1)
    C = C.permute(1,0)
    S_tilde = S_tilde.permute(2,0,1)

    Ones = torch.ones((K,51,51), dtype=torch.float32)

    Wr[Wr<0.5] = 0
    Wr[Wr>=0.5] = 1
    Wx[Wx<0.5] = 0
    Wx[Wx>=0.5] = 1

    Z.requires_grad = True
    optimizer = torch.optim.Adam([Z], lr=lr)

    cost_list = []
    for i in range(loop_count):
        optimizer.zero_grad()

        gen_out = generator(Z)
        X_from_slf = get_tensor(gen_out[:,0,:,:], C)
        loss = cost_func(X, X_from_slf, Wx) #+ 1e-5* torch.norm(Z)
        actual_loss = cost_func(X, X_from_slf, Ones) 
        cost_list.append(loss.item())
        # print('magnitude of grad', Z.grad.norm())
        print(loss.item(), actual_loss.item())
        loss.backward()
        optimizer.step()

    gen_out = generator(Z)
    gen_out = gen_out[:,0,:,:]
    gen_out = gen_out.permute(1,2,0)
    gen_out = gen_out.detach().numpy()
    Z = Z.detach().numpy()

    return Z.copy(), gen_out.copy(), np.array(cost_list).copy()

def optimize_z(W, X, S_tilde, C, R, Z, lr):
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
    Z = torch.from_numpy(Z).type(torch.float32)
    X = torch.from_numpy(X).type(torch.float32)
    X = X.permute(2,0,1)
    S_tilde = torch.from_numpy(S_tilde).type(torch.float32)
    # S_true = torch.from_numpy(S_true).type(torch.float32)
    C = torch.from_numpy(C).type(torch.float32)
    C = C.permute(1,0)
    K = X.shape[0]
    R = int(R)
    lr = float(lr)

    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wr = W.repeat(R,1,1,1)
    Wx = W.repeat(K,1,1,1)
    Wr[Wr<0.5] = 0
    Wr[Wr>=0.5] = 1
    Wx[Wx<0.5] = 0
    Wx[Wx>=0.5] = 1
    Wx = Wx.squeeze()
    W = W.squeeze(dim=0)

    S_tilde = S_tilde.permute(2,0,1)
    S_tilde[S_tilde<0] = 0
    S_tilde = torch.log(S_tilde.unsqueeze(dim=1)+1e-16)
    S_tilde[S_tilde<-30] = 0

    normalizer = []
    for i in range(R):
        normalizer.append(S_tilde[i].min().item())
        S_tilde[i] = S_tilde[i]/S_tilde[i].min()

    # save normalization
    a = torch.ones((R,1,51,51), dtype=torch.float32)
    for i in range(R):
        a[i] = a[i]*normalizer[i]

    Z.requires_grad = True
    optimizer = torch.optim.Adam([Z], lr=lr)

    cost_list = []
    for i in range(loop_count):
        optimizer.zero_grad()

        gen_out = generator(Z)
        gen_out = torch.exp(gen_out*a)
        X_from_slf = get_tensor(gen_out[:,0,:,:], C)
        loss = cost_func(X, X_from_slf, Wx)

        cost_list.append(loss.item())
        # print('magnitude of grad', Z.grad.norm())
        # print(loss)
        loss.backward()
        optimizer.step()

    Z = Z.detach().numpy()
    gen_out = generate(Z, a)

    return Z.copy(), gen_out.copy(), np.array(cost_list).copy()

def inverse_gan(W, X, S_tilde, C, R, S_true):
    """
    Arguments:
        W : Mask 
        X : sampled tensor
        C : current psd estimate
        R : Number of emitters
    Returns:
        the latent vector estimate
    """
    loop_count = 400

    # z_dimension = 512

    # Prepare data
    W = torch.from_numpy(W).type(torch.float32)
    X = torch.from_numpy(X).type(torch.float32)
    S_tilde = torch.from_numpy(S_tilde).type(torch.float32)
    S_true = torch.from_numpy(S_true).type(torch.float32)

    C = torch.from_numpy(C).type(torch.float32)
    R = int(R)


    K = X.shape[2]

    X = X.permute(2,0,1)
    C = C.permute(1,0)

    S_true = S_true.permute(2,0,1)
    S_true[S_true<0] = 0
    S_true = torch.log(S_true.unsqueeze(dim=1) + 1e-16)
    S_true[S_true<-30] = 0

    S_tilde = S_tilde.permute(2,0,1)
    S_tilde[S_tilde<0] = 0
    S_tilde = torch.log(S_tilde.unsqueeze(dim=1)+1e-16)
    S_tilde[S_tilde<-30] = 0

    normalizer = []
    for i in range(R):
        normalizer.append(S_true[i].min().item())
        S_tilde[i] = S_tilde[i]/S_true[i].min()
        S_true[i] = S_true[i]/S_true[i].min()

    # save normalizer
    a = torch.ones((R,1,51,51), dtype=torch.float32)
    for i in range(R):
        a[i] = a[i]*normalizer[i]

    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wr = W.repeat(R,1,1,1)
    Wx = W.repeat(K,1,1,1)

    Wr[Wr<0.5] = 0
    Wr[Wr>=0.5] = 1
    Wx[Wx<0.5] = 0
    Wx[Wx>=0.5] = 1
    Wx = Wx.squeeze()
    print(Wx.shape)
    z = torch.randn((R,z_dimension), dtype=torch.float32)

    # First select a good random vector
    min_criterion = 9999999
    print('starting opt')
    for i in range(400):
        temp = torch.randn((R,z_dimension), dtype=torch.float32)
        temp_out = generator(temp)
        
        temp_out = torch.exp(temp_out*a)
        X_from_slf = get_tensor(temp_out[:,0,:,:], C)
        temp_criterion = (((Wx*X) - (Wx*X_from_slf))**2).sum()

        # temp_criterion = criterion(Wr*temp_out, Wr*S_tilde) 
        if  temp_criterion < min_criterion:
            z.data = temp.clone()
            min_criterion = temp_criterion
            print('min_first', min_criterion)

    for i in range(200):
        temp = 0.2*torch.randn((1,z_dimension), dtype=torch.float32) + z
        temp_out = generator(temp)

        temp_out = torch.exp(temp_out*a)
        X_from_slf = get_tensor(temp_out[:,0,:,:], C)
        temp_criterion = (((Wx*X) - (Wx*X_from_slf))**2).sum()

        # temp_criterion = criterion(Wr*temp_out, Wr*S_tilde) 
        if  temp_criterion < min_criterion:
            z.data = temp.clone()
            min_criterion = temp_criterion
            print('min_second', min_criterion)

    z.requires_grad = True
    optimizer = torch.optim.Adam([z], lr=0.01)

    for i in range(loop_count):
        optimizer.zero_grad()

        gen_out = generator(z)
        gen_out = torch.exp(gen_out*a)
        X_from_slf = get_tensor(gen_out[:,0,:,:], C)
        loss = (((Wx*X) - (Wx*X_from_slf))**2).sum()
        # loss = criterion(Wr*gen_out, Wr*S_tilde)
        # print('x from slf sum', X_from_slf.sum())
        # print('x sum', X.sum())
        # print('W sum', Wx.sum())
        # print('shape of X {}, X_r {}, W {}'.format(X.shape, X_from_slf.shape, Wx.shape))


        # print(loss)
        loss.backward()
        optimizer.step()

    z = z.detach().numpy()
    

    gen_out = gen_out[:,0,:,:]
    gen_out = gen_out.permute(1,2,0)
    gen_out = gen_out.detach().numpy()
    
    return z.copy(), gen_out.copy() 

def inverse_gan3(W, X, S_tilde, C, R, S_true, lr):

    # Prepare data
    W = torch.from_numpy(W).type(torch.float32)
    X = torch.from_numpy(X).type(torch.float32)
    X = X.permute(2,0,1)
    S_tilde = torch.from_numpy(S_tilde).type(torch.float32)
    S_true = torch.from_numpy(S_true).type(torch.float32)
    C = torch.from_numpy(C).type(torch.float32)
    C = C.permute(1,0)
    K = X.shape[0]
    R = int(R)
    lr = float(lr)

    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wr = W.repeat(R,1,1,1)
    Wx = W.repeat(K,1,1,1)
    Wr[Wr<0.5] = 0
    Wr[Wr>=0.5] = 1
    Wx[Wx<0.5] = 0
    Wx[Wx>=0.5] = 1
    Wx = Wx.squeeze()
    W = W.squeeze(dim=0)

    S_true_raw = S_true
    S_true = S_true.permute(2,0,1)
    # S_true[S_true<0] = 0
    S_true = torch.log(S_true.unsqueeze(dim=1)) # + 1e-16)
    # S_true[S_true<-30] = 0

    S_tilde = S_tilde.permute(2,0,1)
    S_tilde[S_tilde<0] = 0
    S_tilde = torch.log(S_tilde.unsqueeze(dim=1)+1e-16)
    S_tilde[S_tilde<-30] = 0



    normalizer = []
    for i in range(R):
        normalizer.append(S_true[i].min().item())
        S_tilde[i] = S_tilde[i]/S_tilde[i].min()
        S_true[i] = S_true[i]/S_true[i].min()

    # save normalization
    a = torch.ones((R,1,51,51), dtype=torch.float32)
    for i in range(R):
        a[i] = a[i]*normalizer[i]


    Z = torch.zeros((1,z_dimension), dtype=torch.float32)
    # First select a good random vector
    for r in range(R):
        cost_list_randomized = []
        cost_list_gd = []

        z = torch.randn((1,z_dimension), dtype=torch.float32)
        min_criterion = 9999999
        # print('starting opt')
        for i in range(400):
            temp = torch.randn((1,z_dimension), dtype=torch.float32)
            temp_out = generator(temp)

            temp_criterion = criterion((W*S_tilde[r]), (W*temp_out))

            if  temp_criterion < min_criterion:
                z.data = temp.clone()
                min_criterion = temp_criterion
                cost_list_randomized.append(min_criterion.item())
                # print('min_first', min_criterion)

        for i in range(200):
            temp = 0.2*torch.randn((1,z_dimension), dtype=torch.float32) + z
            temp_out = generator(temp)

            temp_criterion = criterion((W*S_tilde[r]), (W*temp_out))
            if  temp_criterion < min_criterion:
                z.data = temp.clone()
                min_criterion = temp_criterion
                cost_list_randomized.append(min_criterion.item())
                # print('min_second', min_criterion)

        z.requires_grad = True
        optimizer = torch.optim.Adam([z], lr=lr)

        for i in range(loop_count):
            optimizer.zero_grad()

            gen_out = generator(z)
            # gen_out = torch.exp(gen_out*a)
            loss = criterion((W*S_tilde[r]), (W*gen_out))
            cost_list_gd.append(loss.item())
            actual_loss = criterion(S_true[r], gen_out)
            # print(loss)
            loss.backward()
            optimizer.step()

        print('actual_loss', actual_loss)
        z = z.detach()
        Z = torch.cat((Z,z), dim=0)
    
    # Z = Z[1:]
    # for i in range(loop_count):
    #     optimizer.zero_grad()

    #     gen_out = generator(Z)
    #     gen_out = torch.exp(gen_out*a)
    #     X_from_slf = get_tensor(gen_out[:,0,:,:], C)
    #     loss = (((Wx*X) - (Wx*X_from_slf))**2).sum()

    #     cost_list_gd.append(loss.item())
    #     actual_loss = criterion(S_true[r], gen_out)
    #     # print(loss)
    #     loss.backward()
    #     optimizer.step()

    print('actual_loss', actual_loss)

    gen_out_log = generator(Z[1:])
    Z = Z[1:].numpy()
    print('shape gen_out_log {}, S_true {}'.format(gen_out_log.shape, S_true.shape))
    actual_loss_log = criterion(gen_out_log, S_true)
    print('actual log loss total for gan', actual_loss_log)

    gen_out = generate(Z, a)
    gen_out_torch = torch.from_numpy(gen_out).type(torch.float32)
    actual_loss = criterion(gen_out_torch, S_true_raw)
    print('unnorm loss',actual_loss)

    return Z.copy(), gen_out.copy(), np.array(cost_list_randomized).copy(), np.array(cost_list_gd).copy()

def inverse_gan2(W, X, S_tilde, C, R, S_true, lr):

    # Prepare data
    W = torch.from_numpy(W).type(torch.float32)
    X = torch.from_numpy(X).type(torch.float32)
    X = X.permute(2,0,1)
    S_tilde = torch.from_numpy(S_tilde).type(torch.float32)
    S_true = torch.from_numpy(S_true).type(torch.float32)
    C = torch.from_numpy(C).type(torch.float32)
    C = C.permute(1,0)
    K = X.shape[0]
    R = int(R)
    lr = float(lr)

    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wr = W.repeat(R,1,1,1)
    Wx = W.repeat(K,1,1,1)
    Wr[Wr<0.5] = 0
    Wr[Wr>=0.5] = 1
    Wx[Wx<0.5] = 0
    Wx[Wx>=0.5] = 1
    Wx = Wx.squeeze()
    W = W.squeeze(dim=0)

    S_true_raw = S_true
    S_true = S_true.permute(2,0,1)
    S_true[S_true<0] = 0
    S_true = torch.log(S_true.unsqueeze(dim=1) + 1e-16)
    S_true[S_true<-30] = 0

    S_tilde = S_tilde.permute(2,0,1)
    S_tilde[S_tilde<0] = 0
    S_tilde = torch.log(S_tilde.unsqueeze(dim=1)+1e-16)
    S_tilde[S_tilde<-30] = 0

    normalizer = []
    for i in range(R):
        normalizer.append(S_true[i].min().item())
        S_tilde[i] = S_tilde[i]/S_tilde[i].min()
        S_true[i] = S_true[i]/S_true[i].min()

    # save normalization
    a = torch.ones((R,1,51,51), dtype=torch.float32)
    for i in range(R):
        a[i] = a[i]*normalizer[i]


    Z = torch.zeros((1,z_dimension), dtype=torch.float32)
    # First select a good random vector
    for r in range(R):
        z = torch.randn((1,z_dimension), dtype=torch.float32)
        min_criterion = 9999999
        # print('starting opt')
        for i in range(400):
            temp = torch.randn((1,z_dimension), dtype=torch.float32)
            temp_out = generator(temp)

            temp_criterion = criterion((W*S_tilde[r]), (W*temp_out))

            if  temp_criterion < min_criterion:
                z.data = temp.clone()
                min_criterion = temp_criterion
                # print('min_first', min_criterion)

        for i in range(200):
            temp = 0.2*torch.randn((1,z_dimension), dtype=torch.float32) + z
            temp_out = generator(temp)

            temp_criterion = criterion((W*S_tilde[r]), (W*temp_out))
            if  temp_criterion < min_criterion:
                z.data = temp.clone()
                min_criterion = temp_criterion
                # print('min_second', min_criterion)

        z.requires_grad = True
        optimizer = torch.optim.Adam([z], lr=lr)

        for i in range(loop_count):
            optimizer.zero_grad()

            gen_out = generator(z)
            # gen_out = torch.exp(gen_out*a)
            loss = criterion((W*S_tilde[r]), (W*gen_out))
            actual_loss = criterion(S_true[r], gen_out)
            # print(loss)
            loss.backward()
            optimizer.step()

        print('actual_loss', actual_loss)
        z = z.detach()
        Z = torch.cat((Z,z), dim=0)

    gen_out_log = generator(Z[1:])
    Z = Z[1:].numpy()
    print('shape gen_out_log {}, S_true {}'.format(gen_out_log.shape, S_true.shape))
    actual_loss_log = criterion(gen_out_log, S_true)
    print('actual log loss total for gan', actual_loss_log)

    gen_out = generate(Z, a)
    gen_out_torch = torch.from_numpy(gen_out).type(torch.float32)
    actual_loss = criterion(gen_out_torch, S_true_raw)
    print('unnorm loss',actual_loss)

    return Z.copy(), gen_out.copy() 

def inverse_gan_raw(W, X, S_tilde, C, R, S_true, lr):
    # Prepare data
    W = torch.from_numpy(W).type(torch.float32)
    X = torch.from_numpy(X).type(torch.float32)
    S_tilde = torch.from_numpy(S_tilde).type(torch.float32)
    S_true = torch.from_numpy(S_true).type(torch.float32)
    C = torch.from_numpy(C).type(torch.float32)
    R = int(R)
    lr = float(lr)

    K = X.shape[2]
    Ones = torch.ones((K,51,51), dtype=torch.float32)

    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wr = W.repeat(R,1,1,1)
    Wx = W.repeat(K,1,1,1)
    X = X.permute(2,0,1)
    C = C.permute(1,0)
    S_tilde = S_tilde.permute(2,0,1)
    S_true = S_true.permute(2,0,1)
    
    Wr[Wr<0.5] = 0
    Wr[Wr>=0.5] = 1
    Wx[Wx<0.5] = 0
    Wx[Wx>=0.5] = 1
    Wx = Wx.squeeze(dim=1)

    W = W.squeeze()

    z_list = []
    # First select a good random vector
    for r in range(R):
        z = torch.randn((1,z_dimension), dtype=torch.float32)
        min_criterion = 9999999
        # print('starting opt')
        for i in range(400):
            temp = torch.randn((1,z_dimension), dtype=torch.float32)
            temp_out = generator(temp)

            temp_criterion = criterion((W*S_tilde[r]), (W*temp_out.squeeze()))

            if  temp_criterion < min_criterion:
                z.data = temp.clone()
                min_criterion = temp_criterion
                # print('min_first', min_criterion)

        for i in range(200):
            temp = 0.2*torch.randn((1,z_dimension), dtype=torch.float32) + z
            temp_out = generator(temp)
            temp_criterion = criterion((W*S_tilde[r]), (W*temp_out.squeeze()))
            if  temp_criterion < min_criterion:
                z.data = temp.clone()
                min_criterion = temp_criterion
                # print('min_second', min_criterion)

        
        # # Individual emitter slf optimization
        # z.requires_grad = True
        # optimizer = torch.optim.Adam([z], lr=lr)
        
        # for i in range(300):
        #     optimizer.zero_grad()

        #     gen_out = generator(z)
        #     loss = criterion((W*S_tilde[r]), (W*gen_out.squeeze())) +  1e-5* torch.norm(z)
        #     actual_loss = criterion(S_true[r], gen_out)
        #     print(loss)
        #     loss.backward()
        #     optimizer.step()

        # print('actual_loss', actual_loss)
        z_list.append(z.detach())

    Z = torch.cat(z_list, dim=0)

    # Joint optimization
    Z.requires_grad = True
    optimizer = torch.optim.Adam([Z], lr=lr)

    cost_list = []
    for i in range(loop_count):
        optimizer.zero_grad()

        gen_out = generator(Z)
        X_from_slf = get_tensor(gen_out[:,0,:,:], C)
        loss = cost_func(X, X_from_slf, Wx) + 1e-4* torch.norm(Z)
        actual_loss = cost_func(X, X_from_slf, Ones) 
        cost_list.append(loss.item())
        # print('magnitude of grad', Z.grad.norm())
        print(loss.item(), actual_loss.item())
        loss.backward()
        optimizer.step()

    gen_out = generator(Z)
    Z = Z.detach().numpy()
    
    print(gen_out.shape, S_true.shape)
    actual_loss = criterion(gen_out.squeeze(), S_true)
    print('unnorm loss',actual_loss)

    
    gen_out = gen_out[:,0,:,:]
    gen_out = gen_out.permute(1,2,0)
    gen_out = gen_out.detach().numpy()

    return Z.copy(), gen_out.copy() 


def generate(z, a):
    z = torch.from_numpy(z).type(torch.float32)
    gen_out = generator(z)
    # gen_out = torch.exp(gen_out)
    if a is not None:
        gen_out = torch.exp(gen_out*a)
    gen_out = gen_out[:,0,:,:]
    gen_out = gen_out.permute(1,2,0)
    gen_out = gen_out.detach().numpy()

    return gen_out.copy()


if __name__ == '__main__':
    ROOT = '/home/sagar/Projects/radio_map_deep_prior/psd_recovery/data'
    BASE = '/home/sagar/Projects/radio_map_deep_prior/deep_prior'

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
    S_tilde = scipy.io.loadmat(os.path.join(ROOT,'S_omega.mat'))['S_omega']
    S_true = scipy.io.loadmat(os.path.join(ROOT,'Sc.mat'))['S_true']
    Strue_raw = scipy.io.loadmat(os.path.join(ROOT,'Sc.mat'))['S_true']
    R = C.shape[1]
    # # ans = run_descent(O_mat, T, S_tensor, C, C.shape[1])
    ans = inverse_gan_raw(W, X, S_tilde, C, R, S_true, 0.01)
    # ans = inverse_gan2(O_mat, T, S_tensor, C, C.shape[1])
    # print(ans.shape)
