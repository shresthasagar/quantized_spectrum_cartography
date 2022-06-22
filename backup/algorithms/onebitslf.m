% Implementation of Joint optimization 
% This code uses capital letter such as 'A' with small m appended to denote the matricized version 'Am' 
% And Av to denote the vectorized version of the matrix A of Am


% addpath algorithms
addpath ../../prepare_data
clear;

%% Generate a sample of radio map
% load python modules
% py.sys.setdlopenflags(int32(10));

R = 2;    % number of emitters
shadow_sigma = 5;
Xc = 50;  % Correlation Distance
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

%% Initialize C and S using Deep completion and factorization

% Complete using deep completion
% get initial estimate of z vector
% W_py = py.numpy.array(Om);
% T_py = py.numpy.array(T);
% T_comp = double(py.deep_comp.model(T_py, W_py));

% % Mode-3 matrix unfolding, arrange fibers as columns of the matrix from tensor
% Tm = tens2mat(T_comp,3);
% [C, Sm] = NMF_SPA(Tm, R);

% % % Randomly initialize the SLF
% % if use_gan:
% %     cell_tuple = cell(py.gan_one_bit.init_slf())
% %     S_comp = double(cell_tuple{1})
% %     Z = double(cell_tuple{2})
% % end
% % S_omega = S_comp.*Om;
% % Sm = tens2mat(S_comp,3);
% % Sm_omega = Sm(:, Ov);

% S = mat2tens(Sm,[I J R], 3);

% sample_loss = [sample_loss, Cost(T, T_comp, Om)/f];
% expected_loss = [expected_loss, NMSE(T_true, T_comp)];

% norm_c = [norm_c, frob(C)];
% norm_s = [norm_s, frob(Sm)];
    
% %% Start joint Optimization of C and S

% step = 0;
% previous_loss = 9999;

S = zeros([I J R])
C = zeros([K R])
save('../data/onebitdata.mat', 'C', 'T', 'S', 'C_true', 'S_true', 'Om', 'T_true');

% while (step < steps) && (previous_loss-sample_loss(end) > 0.00003)
%     a = previous_loss - sample_loss(end)
%     previous_loss = sample_loss(end);
%     step = step+1;
%     %% plots
%     if show_plot
%         r = 1;
%         q = 40;
%         p = 6;
%         figure(1);
%         subplot(p,2,[1]);
%         contourf(10*log10(T(:,:,q)), 100, 'linecolor', 'None');
%         colormap jet;
%         title('true map');

%         subplot(p,2,[2]);
%         contourf(10*log10(T_comp(:,:,q)), 100, 'linecolor', 'None');
%         colormap jet;
%         title('reconstructed map');

%         subplot(p,2, [3 4])
%         % hold on;
%         plot(1:K, C_true(:,r),1:K, C(:,r),'--');
%         legend("true psd","reconstructed psd");

%         subplot(p,2, [5 6]);
%         plot(1:length(sample_loss), sample_loss, '-o', 1:length(expected_loss), expected_loss, '-s');
%         legend('sample loss (objective)', 'expected loss');

%         % subplot(p,2, [5 6]);
%         % plot(1:length(expected_loss), expected_loss, '-s',1:length(expected_loss), ones(length(expected_loss),1)*baseline_loss_ae, '-o', 1:length(expected_loss), ones(length(expected_loss),1)*baseline_deep_only);
%         % legend('expected loss', 'baseline ae', 'baseline deep');

%         subplot(p,2, [7 8]);
%         plot(1:length(sample_loss), sample_loss, '-o');
%         legend('sample loss (objective)');

%         subplot(p,2, [9]);
%         plot(1:length(norm_z), norm_z);
%         legend('Z energy');

%         subplot(p,2, [10]);
%         plot(1:length(norm_c), norm_c);
%         legend('C energy')

%         subplot(p,2, [11 12]);
%         plot(1:length(objective), objective, '-o');
%         legend('objective')

%     end
    
%     %% Step 1
%     % Cr optimization subprobplem.
%     % Non Linear Least Squares
%     tic
%     if optimize_c
%         Y = Tm*W;
%         Q = Sm*W;
%         norm_before_C = frob(Y - C*Q)^2
%         C_before_opt = C;
        
%         % with regularization
%         C = [];
%         lambdaI = lambda*eye(R);
%         A = [Q'; lambdaI;];
%         for k=1:K
%             b = [(Y(k,:))';zeros(R,1)];
%             % c = lsqnonneg(Q', (Y(k,:))');
%             c = lsqnonneg(A, b);
%             C = [C; c'];
%         end
        
%         % % without regularization
%         % Y = Tm_omega;
%         % Q = Sm*W;
%         % Q = Q(:, Ov);
%         % norm_before_C = frob(Y - C*Q)^2;
%         % C_before_opt = C;
        
%         % A = kron(Q', eye(K));
%         % vecC = lsqnonneg(A, Y(:));
%         % C = reshape(vecC, 64, R);
        
%         norm_before_column_norm = frob(Y - C*Q)^2

%         % C = ColumnNormalization(C);

%         norm_after_C = frob(Y-C*Q)^2
%     end
%     time_copt = toc;
%     T_comp = get_tensor(S_comp, C);
    
%     sample_loss = [sample_loss, Cost(T, T_comp, Om)/f];
%     expected_loss = [expected_loss, NMSE(T_true, T_comp)];

    
%     %% Step 2
%     % Stheta optimization subproblem
%     W_py = py.numpy.array(Om);
%     X_py = py.numpy.array(T);
%     C_py = py.numpy.array(C);
%     S_true_py = py.numpy.array(S_true);

%     % Call the nn gradient descent optimizer: returns optimized S_omega
%     tic;
%     if use_gan
%         tuple = py.nn_descent_gan.optimize_z_raw(W_py, X_py, S_py, C_py, R, Z_py, gd_lr);
%         cell_tuple = cell(tuple);
%         Z_py = cell_tuple{1};
%         S_comp = cell_tuple{2};
%         S_comp = double(S_comp);
%     else
%         tuple = py.nn_descent_ae.run_descent_ae(W_py, X_py, Z_py, C_py, R);
%         cell_tuple = cell(tuple);
%         Z_py = cell_tuple{1};
%         S_comp = cell_tuple{2};
%         S_comp = double(S_comp);
%     end

%     S_omega = S_comp.*Om;
%     Sm = tens2mat(S_comp,3);
%     Sm_omega = Sm(:, Ov);
%     time_sopt = toc;

%     T_comp = get_tensor(S_comp, C);
%     sample_loss = [sample_loss, Cost(T, T_comp, Om)/f];
%     expected_loss = [expected_loss, NMSE(T_true, T_comp)];
    
%     norm_z = [norm_z, frob(Z)];
%     norm_c = [norm_c, frob(C)];
%     norm_s = [norm_s, frob(S_comp)];
%     objective = [objective, (Cost(T, T_comp, Om)/f+lambda*frob(C)+lambda_z*frob(Z))];
% end


%% helper functions
function error = NMSE(T, T_recovered)
    error = (frob(T - T_recovered)^2)/(frob(T)^2);
end

function error = Cost(T, T_recovered, Om)
    error = frob(Om.*(T - T_recovered))^2;
end

function X = get_tensor(S_omega, C)
    sizec = size(C);
    X = zeros(51,51,sizec(1));

    for rr=1:sizec(2)
        X = X + outprod(S_omega(:,:,rr), C(:,rr));
    end
end