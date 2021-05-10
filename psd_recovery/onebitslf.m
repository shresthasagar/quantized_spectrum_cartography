% Implementation of Joint optimization 
% This code uses capital letter such as 'A' with small m appended to denote the matricized version 'Am' 
% And Av to denote the vectorized version of the matrix A of Am


clear;
%% Generate a sample of radio map
% load python modules
py.sys.setdlopenflags(int32(10));

Rs = [5,7,9,11,13];
% Rs = [7];
R = 2;    % number of emitters
sigmas = [4, 5, 6, 7, 8];
% sigmas = [4];
shadow_sigma = 5;       % shadowing variance
Xcs = [10, 30, 50, 70, 90];
% Xcs = [50];
Xc = 50;  % Correlation Distance
% fs = [0.01, 0.05, 0.1, 0.15, 0.2];
fs = [0.05];
f = 0.05;
snr = 20;
snrs = [0,10,20,30,40];
use_snr = false;

structure_c = true;
optimize_c = true;    % whether to generate the radio map using pre-determined emitter locations
use_gan = false;
show_plot = false;
normalize_input_columns = true;
compute_detection_probability = false;
is_separable = true;
test_btd = true;

num_egs = 1;
steps = 0;

I = 51;
J = 51;
K = 64;     % length of spectrum
gd_lr = 0.01;

z_dimension = 256;
lambda = 0.00000;
lambda_z = 1e-7;
folder_name = strcat('data/fake/slf_mat/');

miss_threshold = 0.25;
misdetect_threshold = 1.75;


for l = 1:length(fs)
    % shadow_sigma = sigmas(l);
    % R = Rs(l);
    f = fs(l);
    % Xc = Xcs(l);
    % snr = snrs(l);

    count_valid = 0;
    for m = 1:num_egs
        m

        expected_loss = [];
        sample_loss = [];
        
        norm_c = [];
        norm_z = [];
        norm_s = [];
        
        objective = [];
        % generate radio map
        % the last argument determines the type of psd basis function 's': sinc 'g': gaussian
        [T_true, Sc, C_true, peaks] = generate_map(false, K, R, shadow_sigma, Xc, structure_c, 's', is_separable); 
        % T_true = zeros(I,J,K);
        % T_true(:,:,:) = T_true_list(m,:,:,:);

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
        
        [I,J,K] = size(T);
        IJ = I*J;
        num_samples = round(f*IJ);
        Omega = randperm(IJ, num_samples)';
        
        % Mode-3 matrix unfolding, arrange fibers as columns of the matrix from tensor
        Tm = tens2mat(T,3);
        
        % sampling matrix
        Ov = false(1,IJ);
        Ov(Omega) = true;
        Om = reshape(Ov,[I,J]);

        Tm_omega = Tm(:, Ov);
        
        S_true = zeros(I,J,R);
        for rr=1:R
            S_true(:,:,rr) = Sc{rr};
        end
        
        
        % Randomly initialize the SLF
        if use_gan:
            cell_tuple = cell(py.gan_one_bit.init_slf())
            S_comp = double(cell_tuple{1})
            Z = double(cell_tuple{2})
        end
        S_omega = S_comp.*Om;
        Sm = tens2mat(S_comp,3);
        Sm_omega = Sm(:, Ov);

        C = zeros(K, R);

        % get initial estimate of z vector
        S_true_py = py.numpy.array(S_true);
        W_py = py.numpy.array(Om);
        X_py = py.numpy.array(T);
        
        % tic;
        % % output of deep_only network
        % T_deep = double(py.deep_only.model(X_py, W_py));
        % Tr{l} = T(:,:,48);
        % Tr_deep{l} = T_deep(:,:,48);
        % time_deep = toc;

        %% reconstruct the recovered tensor for joint_optimization
        T_comp = get_tensor(S_comp, C);
        Cost_ae = Cost(T, T_comp, Om);
        
        sample_loss = [sample_loss, Cost_ae/f];
        expected_loss = [expected_loss, SRE(T_true, T_comp)];
        norm_z = [norm_z, frob(Z)];
        norm_c = [norm_c, frob(C)];
        norm_s = [norm_s, frob(S_comp)];

        initial_loss = SRE(T_true,T_comp);

        objective = [objective, ((cost_comp/f)+lambda*frob(C)+lambda_z*frob(Z))];
        
        
        %% Start joint Optimization of C and S
        W = zeros(IJ, IJ);
        for i=1:IJ
            if Ov(i)
                W(i,i) = 1;
            end
        end
        % baseline_deep_only = SRE(T_true, T_deep);
        step = 0;
        previous_loss = 9999;

        while (step < steps) && (previous_loss-sample_loss(end) > 0.00003)
            a = previous_loss - sample_loss(end)
            previous_loss = sample_loss(end);
            step = step+1;
            %% plots
            if show_plot
                r = 1;
                q = 40;
                p = 6;
                figure(1);
                subplot(p,2,[1]);
                contourf(10*log10(T(:,:,q)), 100, 'linecolor', 'None');
                colormap jet;
                title('true map');
    
                subplot(p,2,[2]);
                contourf(10*log10(T_comp(:,:,q)), 100, 'linecolor', 'None');
                colormap jet;
                title('reconstructed map');
    
                subplot(p,2, [3 4])
                % hold on;
                plot(1:K, C_true(:,r),1:K, C(:,r),'--');
                legend("true psd","reconstructed psd");
    
                subplot(p,2, [5 6]);
                plot(1:length(sample_loss), sample_loss, '-o', 1:length(expected_loss), expected_loss, '-s');
                legend('sample loss (objective)', 'expected loss');

                % subplot(p,2, [5 6]);
                % plot(1:length(expected_loss), expected_loss, '-s',1:length(expected_loss), ones(length(expected_loss),1)*baseline_loss_ae, '-o', 1:length(expected_loss), ones(length(expected_loss),1)*baseline_deep_only);
                % legend('expected loss', 'baseline ae', 'baseline deep');

                subplot(p,2, [7 8]);
                plot(1:length(sample_loss), sample_loss, '-o');
                legend('sample loss (objective)');

                subplot(p,2, [9]);
                plot(1:length(norm_z), norm_z);
                legend('Z energy');

                subplot(p,2, [10]);
                plot(1:length(norm_c), norm_c);
                legend('C energy')

                subplot(p,2, [11 12]);
                plot(1:length(objective), objective, '-o');
                legend('objective')

            end
            
            %% Step 1
            % Cr optimization subprobplem.
            % Non Linear Least Squares
            tic
            if optimize_c
                Y = Tm*W;
                Q = Sm*W;
                norm_before_C = frob(Y - C*Q)^2
                C_before_opt = C;
                
                % with regularization
                C = [];
                lambdaI = lambda*eye(R);
                A = [Q'; lambdaI;];
                for k=1:K
                    b = [(Y(k,:))';zeros(R,1)];
                    % c = lsqnonneg(Q', (Y(k,:))');
                    c = lsqnonneg(A, b);
                    C = [C; c'];
                end
                
                % % without regularization
                % Y = Tm_omega;
                % Q = Sm*W;
                % Q = Q(:, Ov);
                % norm_before_C = frob(Y - C*Q)^2;
                % C_before_opt = C;
                
                % A = kron(Q', eye(K));
                % vecC = lsqnonneg(A, Y(:));
                % C = reshape(vecC, 64, R);
                
                norm_before_column_norm = frob(Y - C*Q)^2

                % C = ColumnNormalization(C);

                norm_after_C = frob(Y-C*Q)^2
            end
            time_copt = toc;
            T_comp = get_tensor(S_comp, C);
            
            sample_loss = [sample_loss, Cost(T, T_comp, Om)/f];
            expected_loss = [expected_loss, SRE(T_true, T_comp)];

            
            %% Step 2
            % Stheta optimization subproblem
            W_py = py.numpy.array(Om);
            X_py = py.numpy.array(T);
            C_py = py.numpy.array(C);
            S_true_py = py.numpy.array(S_true);

            % Call the nn gradient descent optimizer: returns optimized S_omega
            tic;
            if use_gan
                tuple = py.nn_descent_gan.optimize_z_raw(W_py, X_py, S_py, C_py, R, Z_py, gd_lr);
                cell_tuple = cell(tuple);
                Z_py = cell_tuple{1};
                S_comp = cell_tuple{2};
                S_comp = double(S_comp);
            else
                tuple = py.nn_descent_ae.run_descent_ae(W_py, X_py, Z_py, C_py, R);
                cell_tuple = cell(tuple);
                Z_py = cell_tuple{1};
                S_comp = cell_tuple{2};
                S_comp = double(S_comp);
            end

            S_omega = S_comp.*Om;
            Sm = tens2mat(S_comp,3);
            Sm_omega = Sm(:, Ov);
            time_sopt = toc;

            T_comp = get_tensor(S_comp, C);
            sample_loss = [sample_loss, Cost(T, T_comp, Om)/f];
            expected_loss = [expected_loss, SRE(T_true, T_comp)];
            
            norm_z = [norm_z, frob(Z)];
            norm_c = [norm_c, frob(C)];
            norm_s = [norm_s, frob(S_comp)];
            objective = [objective, (Cost(T, T_comp, Om)/f+lambda*frob(C)+lambda_z*frob(Z))];
        end

end    

%% functions
function error = SRE(T, T_recovered)
    error = (frob(T - T_recovered)^2)/(frob(T)^2);
end

function error = NAE(T, T_recovered, R)
    error = (1/R)*sum(abs(T/sum(abs(T),'all') - T_recovered/sum(abs(T_recovered),'all')), 'all');
end

function error = Cost(T, T_recovered, Om)
    error = sum((Om.*T - Om.*T_recovered).^2, 'all');
end

function error = MSE(A, B)
    num_el = prod(size(A));
    error = 1.0/num_el *sum((A - B).^2, 'all');
end

function X = get_tensor(S_omega, C)
    sizec = size(C);
    X = zeros(51,51,sizec(1));

    for rr=1:sizec(2)
        X = X + outprod(S_omega(:,:,rr), C(:,rr));
    end
end