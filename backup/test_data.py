import scipy.io as sio
from params import mean_slf
    data = sio.loadmat('data/onebitdata')

    f = 0.5
    std_probit = 0.008
    # stds_probit = [0.00001, 0.0001, 0.001, 0.005, 0.008, 0.01, 0.02, 0.04]
    stds_probit = [0.008]
    nmse_list = []

    loss_model = 'probit'  # one of 'sigmoid', 'probit', and 'determ'

    S = torch.from_numpy(data['S']).type(torch.float32)
    T = torch.from_numpy(data['T']).type(torch.float32)
    C = torch.from_numpy(data['C']).type(torch.float32)
    S_true = torch.from_numpy(data['S_true']).type(torch.float32)
    C_true = torch.from_numpy(data['C_true']).type(torch.float32)
    T_true = torch.from_numpy(data['T_true']).type(torch.float32)

    Om = torch.from_numpy(data['Om']).type(torch.float32)

    T = T.permute(2,0,1)
    T_true = T_true.permute(2,0,1)
    S = S.permute(2,0,1)
    S_true = S_true.permute(2,0,1)
    C = C.permute(1,0)
    C_true = C_true.permute(1,0)

    W = Om.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wx = W.repeat(64,1,1,1)
