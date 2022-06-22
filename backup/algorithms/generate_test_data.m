% addpath algorithms
addpath ../../prepare_data
clear;

%% Generate a sample of radio map
% load python modules
% py.sys.setdlopenflags(int32(10));

R = 2;    % number of emitters
shadow_sigma = 4;
Xc = 90;  % Correlation Distance
f = 1;
snr = 20;
use_snr = false;

structure_c = true;
optimize_c = true;    % whether to generate the radio map using pre-determined emitter locations
use_gan = false;
show_plot = false;
normalize_input_columns = true;
is_separable = true;

maxIter = 0;
mean_slf = 0.0045;
std_slf = 0.0191;

I = 51;
J = 51;
K = 64;     % length of spectrum
gd_lr = 0.01;

z_dimension = 256;

folder_name = strcat('data/fake/slf_mat/');


expected_loss = [];
sample_loss = [];

norm_c = [];
norm_z = [];
norm_s = [];

objective = [];

% generate radio map
% the last argument determines the type of psd basis function 's': sinc 'g': gaussian
[T_true, Sc, C_true, peaks] = generate_map(false, K, R, shadow_sigma, Xc, structure_c, 'g', is_separable); 
S_true = zeros(I,J,R);
for rr=1:R
    S_true(:,:,rr) = Sc{rr};
end
C_true = ColumnNormalization(C_true);

if use_snr
    Ps = frob(T_true)^2;
    Pn = Ps*10^(-snr/10);
    sn = sqrt(Pn/I/J/K);
    if sn>=1e2
        sn =0;
    end
    T = T_true + sn*1.73*rand(I,J,K);
else
    T = T_true;
end
T(T<0) = 0;

T(T>mean_slf) = 1;
T(T<mean_slf) = -1;

[I,J,K] = size(T);
IJ = I*J;
num_samples = round(f*IJ);
Omega = randperm(IJ, num_samples)';

% sampling matrix
Ov = false(1,IJ);
Ov(Omega) = true;
Om = reshape(Ov,[I,J]);

S = zeros([I J R])
C = zeros([K R])
save('../data/onebitdata1.mat', 'C', 'T', 'S', 'C_true', 'S_true', 'Om', 'T_true');
