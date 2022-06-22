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

sample_cost_joint = [];

cost_tps = [];
cost_btd = [];
cost_deep = [];
cost_nmf = [];
cost_joint = [];

naec_btd = [];
naec_nmf = [];
naec_joint = [];

naes_btd = [];
naes_nmf = [];
naes_joint = [];

misses_deep = [];
misses_ae = [];
misses_joint = [];

misdetect_deep = [];
misdetect_ae = [];
misdetect_joint = [];

Tr = [];
Tr_deep = [];

%% save cost for individual example for analysis
costs = zeros(length(fs), 5, num_egs);

for l = 1:length(fs)
    % shadow_sigma = sigmas(l);
    % R = Rs(l);
    f = fs(l);
    % Xc = Xcs(l);
    % snr = snrs(l);
    
    sum_sample_cost_joint = 0;

    sum_cost_tps = 0;
    sum_cost_btd = 0;
    sum_cost_deep = 0;
    sum_cost_nmf = 0;
    sum_cost_joint = 0;

    sum_naec_btd = 0;
    sum_naec_nmf = 0;
    sum_naec_joint = 0;
    
    sum_naes_btd = 0;
    sum_naes_nmf = 0;
    sum_naes_joint = 0;

    % for each condition
    total_peaks = 0;
    total_lows = 0;

    sum_misses_deep = 0;
    sum_misses_ae = 0;
    sum_misses_joint = 0;

    sum_misdetect_deep = 0;
    sum_misdetect_ae = 0;
    sum_misdetect_joint = 0;

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
            
%         Sc  = {};
%         for rr=1:R
%             filename = sprintf('%07d.mat', 10+m*R+rr); 
%             file_path = strcat(folder_name, filename);
%             a = load(file_path);
%             Sc{rr} = double(a.Sc);
%         end
% 
%         T = zeros(I,J,K);
%         for rr=1:R
%             T = T + outprod(Sc{rr},C_true(:,rr));
%         end

        %% TPS baseline
        
        %% Initialization
        
        % Use NMF factorization to get C and S from the sampled tensor
        
        [I,J,K] = size(T);
        IJ = I*J;
        num_samples = round(f*IJ);
        Omega = randperm(IJ, num_samples)';
        
        % Mode-3 matrix unfolding, arrange fibers as columns of the matrix from tensor
        Tm = tens2mat(T,3);
        
        % sampling matrix
        Ov = false(1,IJ);
        Ov(Omega) = true;

        % TPS
        tic;
        T_tps = full_tps(T,Ov, false);
        time_tps = toc;

        Tm_omega = Tm(:, Ov);

        % if normalize_input_columns
        %     [Tm_omega_norm, Normalizer] = ColumnSumNormalization(Tm_omega);
        %     Tm_omega = Tm_omega_norm;
        % end

        % % apply SPA algorithm to obtain the indices of factor C
        % indices_C = SPA(Tm_omega, R);
        % C = Tm_omega(:, indices_C);

        % if normalize_input_columns
        %     C = C.*Normalizer(:,indices_C);
        %     Tm_omega = Tm_omega;
        % end

        % %% remove permutation
        % [cpderrc,per,~]=cpderr(C_true,C);
        % C_noperm = C*per;
        % C_p = ColumnPositive(C_noperm);
        % C_p(C_p<0)=0;
        % C = ColumnNormalization(C_p);

        % % obtain S matrix whose rows are the spatial loss field for emitters
        % pseudo_inverse_C = pseudo_inverse(C);
        % Sm_omega = pseudo_inverse_C*Tm_omega;

        % Alternative SPA on T transpose
        Tm_omega = Tm_omega';

        tic;

        if normalize_input_columns
            [Tm_omega_norm, Normalizer] = ColumnSumNormalization(Tm_omega);
            Tm_omega = Tm_omega_norm;    
        end

        indices_S = SPA(Tm_omega, R);
        Sm_omega = Tm_omega(:,indices_S);

        if normalize_input_columns
            Sm_omega = Sm_omega.*Normalizer(:, indices_S);
            Tm_omega = Tm_omega.*Normalizer;
        end
        
        % obtain the C matrix 
        pseudo_inverse_S = pseudo_inverse(Sm_omega);
        C = pseudo_inverse_S*Tm_omega;
        C = C';
        
        % remove permutation
        [cpderrc,per,~]=cpderr(C_true,C);
        C_noperm = C*per;
        C_p = ColumnPositive(C_noperm);
        C_p(C_p<0)=0;
        [C, d] = ColumnNormalization(C_p);
        
        Sm_omega = Sm_omega*per;
        Sm_omega = Sm_omega.*d;
        Sm_omega = Sm_omega';
        Tm_omega = Tm_omega';
        
        S_true = zeros(I,J,R);
        for rr=1:R
            S_true(:,:,rr) = Sc{rr};
        end
        
        
        %% Reconstruct spatial loss field for each emitter from the Sm_omega matrix
        S_omega = zeros(R, I*J);
        j = 1;
        for i=1:I*J
            if Ov(i)
                S_omega(:,i) = Sm_omega(:,j);
                j = j+1;
            end
        end
        S_omega = mat2tens(S_omega,[I J R], 3);
        
        Om = reshape(Ov,[I,J]);
        time_spa = toc;
        
        % get initial estimate of z vector
        S_true_py = py.numpy.array(S_true);
        W_py = py.numpy.array(Om);
        X_py = py.numpy.array(T);
        S_py = py.numpy.array(S_omega);
        C_py = py.numpy.array(C);
        
        tic;
        % output of NMF-AE 
        tuple = py.nn_descent_ae.model_ae(S_py, W_py, R, S_true_py);
        cell_tuple = cell(tuple);
        S_nmf = double(cell_tuple{2});
        T_nmf = get_tensor(S_nmf, C);
        Tr_nmf{l} = T_nmf(:,:,48);
        baseline_loss_ae = SRE(T_true,T_nmf);
        C_nmf = C;
        
        time_nasdac = toc;
        
        if test_btd
            [T_btd, S_btd, C_btd] = BTD(T, S_true, C_true, f);
        else
            T_btd = T;
            S_btd = S_true;
            C_btd = C_true;
        end
        
        tic;
        % output of deep_only network
        T_deep = double(py.deep_only.model(X_py, W_py));
        Tr{l} = T(:,:,48);
        Tr_deep{l} = T_deep(:,:,48);
        time_deep = toc;

        % initialization for joint optimization
        if use_gan
            tuple = py.nn_descent_gan.inverse_gan_raw(W_py, X_py, S_py, C_py, R, S_true_py, gd_lr);
            cell_tuple = cell(tuple);
            Z_py = cell_tuple{1};
            S_joint = cell_tuple{2};
            S_joint = double(S_joint);
            Z = double(Z_py);
        else
            tuple = py.nn_descent_ae.model_ae(S_py, W_py, R, S_true_py);
            cell_tuple = cell(tuple);
            Z_py = cell_tuple{1};
            S_joint = cell_tuple{2};
            S_joint = double(S_joint);
            Z = double(Z_py);
        end

        S_omega = S_joint.*Om;
        Sm = tens2mat(S_joint,3);
        Sm_omega = Sm(:, Ov);

        %% reconstruct the recovered tensor for joint_optimization
        T_joint = get_tensor(S_joint, C);
        Cost_ae = Cost(T, T_joint, Om);
        
        sample_loss = [sample_loss, Cost_ae/f];
        expected_loss = [expected_loss, SRE(T_true, T_joint)];
        norm_z = [norm_z, frob(Z)];
        norm_c = [norm_c, frob(C)];
        norm_s = [norm_s, frob(S_joint)];

        initial_loss = SRE(T_true,T_joint);

        objective = [objective, ((cost_joint/f)+lambda*frob(C)+lambda_z*frob(Z))];
        
        
        %% Start joint Optimization of C and S
        W = zeros(IJ, IJ);
        for i=1:IJ
            if Ov(i)
                W(i,i) = 1;
            end
        end
        baseline_deep_only = SRE(T_true, T_deep);
        step=0;
        previous_loss = 9999;
        while (step < steps) && (previous_loss-sample_loss(end) > 0.003)
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
                contourf(10*log10(T_joint(:,:,q)), 100, 'linecolor', 'None');
                colormap jet;
                title('reconstructed map');
    
                subplot(p,2, [3 4])
                % hold on;
                plot(1:K, C_true(:,r),1:K, C(:,r),'--');
                legend("true psd","reconstructed psd");
    
                % subplot(p,2, [5 6]);
                % plot(1:length(sample_loss), sample_loss, '-o', 1:length(expected_loss), expected_loss, '-s');
                % legend('sample loss (objective)', 'expected loss');

                subplot(p,2, [5 6]);
                plot(1:length(expected_loss), expected_loss, '-s',1:length(expected_loss), ones(length(expected_loss),1)*baseline_loss_ae, '-o', 1:length(expected_loss), ones(length(expected_loss),1)*baseline_deep_only);
                legend('expected loss', 'baseline ae', 'baseline deep');

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
            T_joint = get_tensor(S_joint, C);
            
            sample_loss = [sample_loss, Cost(T, T_joint, Om)/f];
            expected_loss = [expected_loss, SRE(T_true, T_joint)];

            
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
                S_joint = cell_tuple{2};
                S_joint = double(S_joint);
            else
                tuple = py.nn_descent_ae.run_descent_ae(W_py, X_py, Z_py, C_py, R);
                cell_tuple = cell(tuple);
                Z_py = cell_tuple{1};
                S_joint = cell_tuple{2};
                S_joint = double(S_joint);
            end

            S_omega = S_joint.*Om;
            Sm = tens2mat(S_joint,3);
            Sm_omega = Sm(:, Ov);
            time_sopt = toc;

            T_joint = get_tensor(S_joint, C);
            sample_loss = [sample_loss, Cost(T, T_joint, Om)/f];
            expected_loss = [expected_loss, SRE(T_true, T_joint)];
            
            norm_z = [norm_z, frob(Z)];
            norm_c = [norm_c, frob(C)];
            norm_s = [norm_s, frob(S_joint)];
            objective = [objective, (Cost(T, T_joint, Om)/f+lambda*frob(C)+lambda_z*frob(Z))];
        end

        %% In this order: TPS, BTD, DEEP, NMF, JOINT
        costs(l, 1, m) = SRE(T_true, T_tps);
        costs(l, 2, m) = SRE(T_true, T_btd);
        costs(l, 3, m) = SRE(T_true, T_deep);
        costs(l, 4, m) = SRE(T_true, T_nmf);
        costs(l, 5, m) = SRE(T_true, T_joint);
        
        sum_naes_nmf = sum_naes_nmf + NAE(S_true, S_nmf, R);
        sum_naes_joint = sum_naes_joint + NAE(S_true, S_joint, R);
        
        sum_naec_nmf = sum_naec_nmf + NAE(C_true, C_nmf, R);
        sum_naec_joint = sum_naec_joint + NAE(C_true, C, R);
        
        
        if costs(l, 2, m) < 3
            count_valid = count_valid +1 ;
            sum_cost_btd = sum_cost_btd + costs(l, 2, m);   
            sum_naes_btd = sum_naes_btd + NAE(S_true, S_btd, R);
            sum_naec_btd = sum_naec_btd + NAE(C_true, C_btd, R);
        end
        sum_cost_tps = sum_cost_tps + costs(l, 1, m);
        sum_cost_deep = sum_cost_deep + costs(l, 3, m);
        sum_cost_nmf = sum_cost_nmf + costs(l, 4, m) ;
        sum_cost_joint = sum_cost_joint + costs(l, 5, m) ;
        
        % if(opt_cost<initial_loss)
        %     sum_cost_joint = sum_cost_joint + opt_cost;
        % else
        %     sum_cost_joint = sum_cost_joint + initial_loss;
        % end


        % Detection Probability Computation
        if compute_detection_probability
            for k=1:K
                for pk=1:length(peaks)
                    peak_value = T(peaks{pk}(2), peaks{pk}(1), k);
                    if peak_value > 0.01
                        total_peaks = total_peaks + 1;
                        if T_deep(peaks{pk}(2), peaks{pk}(1), k) < miss_threshold*peak_value
                            sum_misses_deep = sum_misses_deep + 1;
                        end
                        if T_joint(peaks{pk}(2), peaks{pk}(1), k) < miss_threshold*peak_value
                            sum_misses_joint = sum_misses_joint + 1;
                        end
                        if T_nmf(peaks{pk}(2), peaks{pk}(1), k) < miss_threshold*peak_value
                            sum_misses_ae = sum_misses_ae + 1;
                        end
                    else
                        total_lows = total_lows + 1;
                        if T_deep(peaks{pk}(2), peaks{pk}(1), k) > max(0.01, misdetect_threshold*peak_value)
                            sum_misdetect_deep = sum_misdetect_deep + 1;
                        end
                        if T_joint(peaks{pk}(2), peaks{pk}(1), k) > max(0.01, misdetect_threshold*peak_value)
                            sum_misdetect_joint = sum_misdetect_joint + 1;
                        end
                        if T_nmf(peaks{pk}(2), peaks{pk}(1), k) > max(0.01, misdetect_threshold*peak_value)
                            sum_misdetect_ae = sum_misdetect_ae + 1;
                        end
                    end
                end
            end
        end

    end

    misses_deep = [misses_deep sum_misses_deep/(total_peaks+1)];
    misses_ae = [misses_ae sum_misses_ae/(total_peaks+1)];
    misses_joint = [misses_joint sum_misses_joint/(total_peaks+1)];
    misdetect_deep = [misdetect_deep sum_misdetect_deep/(total_lows+1)];
    misdetect_ae = [misdetect_ae sum_misdetect_ae/(total_lows+1)];
    misdetect_joint = [misdetect_joint sum_misdetect_joint/(total_lows+1)];

    naec_btd = [naec_btd sum_naec_btd/count_valid];
    naec_nmf = [naec_nmf sum_naec_nmf/num_egs];
    naec_joint = [naec_joint sum_naec_joint/num_egs];

    naes_btd = [naes_btd sum_naes_btd/count_valid];
    naes_nmf = [naes_nmf sum_naes_nmf/num_egs];
    naes_joint = [naes_joint sum_naes_joint/num_egs];

    cost_tps = [cost_tps, sum_cost_tps/num_egs];
    cost_btd = [cost_btd, sum_cost_btd/count_valid]
    cost_deep = [cost_deep, sum_cost_deep/num_egs];
    cost_nmf = [cost_nmf, sum_cost_nmf/num_egs];
    cost_joint = [cost_joint, sum_cost_joint/num_egs];

end    
%% show plot of PSD and SLF
show_plot=true;
show_psd = false;
show_slf = true;
show_map = false;
if show_plot

    if show_map
        r = 48;
        fig = figure(2);

        tiledlayout(2,3, 'Padding', 'none', 'TileSpacing', 'compact', 'Position', [0.0 0.05 0.87 0.89]); 
        
        max_limit = max(max(10*log10(T(:,:,r)+1e-16)))+1;
        min_limit = min(min(10*log10(T(:,:,r)+1e-16)));
        
        nexttile         
        % subplot(231);
        contourf(10*log10(T(:,:,r)), 100, 'linecolor', 'None');
        caxis([min_limit max_limit])
        set(gca,'XTick',[],'YTick',[])
        % header = sprintf('true slf, %d , R=%d, $\sigma$=%d, $X_c$=%d', uint8(f*100), R, shadow_sigma, Xc)
        % title("true slf, %d% sampling, R=%d, sigma=10, Xc=30");
        header = sprintf('True Map')
        title('True map at $48$-th bin', 'Interpreter', 'latex');
        xlabel('(a)');
        colormap jet;


        nexttile    
        % subplot(233);
        contourf(10*log10(T_joint(:,:,r)), 100, 'linecolor', 'None');
        caxis([min_limit max_limit]);
        set(gca,'XTick',[],'YTick',[]);
        header = sprintf('Joint-Opt');
        title('\texttt{DowJons} SRE$=0.3163$', 'Interpreter', 'latex');
        xlabel('(b)');
        colormap jet;
        
        nexttile    
        % subplot(233);
        contourf(10*log10(T_nmf(:,:,r)), 100, 'linecolor', 'None');
        caxis([min_limit max_limit]);
        set(gca,'XTick',[],'YTick',[]);
        header = sprintf('NMF-DAE');
        xlabel('(c)');
        title('\texttt{Nasdac} SRE$=1.1751$', 'Interpreter', 'latex');
        colormap jet;

        nexttile    
        % subplot(232);
        contourf(10*log10(T_deep(:,:,r)), 100, 'linecolor', 'None');
        caxis([min_limit max_limit]);
        set(gca,'XTick',[],'YTick',[]);
        header = sprintf('Deep-Only');
        title('\texttt{DeepComp} SRE$=0.4201$', 'Interpreter', 'latex');
        xlabel('(d)');
        colormap jet;
        
        nexttile    
        % subplot(235);
        contourf(10*log10(T_btd(:,:,r)), 100, 'linecolor', 'None');
        caxis([min_limit max_limit]);
        set(gca,'XTick',[],'YTick',[]);
        header = sprintf('BTD');
        title('\texttt{LL1} SRE$=1.2288$', 'Interpreter', 'latex');
        xlabel('(e)');
        colormap jet;

        nexttile    
        % subplot(236);
        contourf(real(10*log10(T_tps(:,:,r))), 100, 'linecolor', 'None');
        caxis([min_limit max_limit]);
        set(gca,'XTick',[],'YTick',[]);
        header = sprintf('TPS');
        title('\texttt{TPS} SRE$=1.9181$', 'Interpreter', 'latex');
        xlabel('(f)');
        colormap jet;
        
        cb = colorbar;
        cb.Label.String = 'dB';
        cb.Layout.Tile = 'east';
    end

    if show_slf
        k = 20
        figure(3);
        tiledlayout(4,3, 'Padding', 'none', 'TileSpacing', 'compact', 'Position', [0.01 0.03 0.86 0.94]); 
        
        nexttile
        % X = (T(:,:,k));
        % X_omega = X.*Om;
        % contourf(Om, 100, 'linecolor', 'None');
        imshow(Om);

        set(gca,'XTick',[],'YTick',[])
        title('Measurement locations $\rho=5\%$', 'Interpreter', 'latex');
        xlabel('(a1)', 'Interpreter', 'latex');

        % colormap jet;

        nexttile    
        % subplot(231);
        plot(1:K, squeeze(C_true(:,1)), 'linewidth',2, 'linewidth', 2, 'color', 'red')
        set(gca,'XTick',[],'YTick',[])
        % header = sprintf('true slf, %d , R=%d, $\sigma$=%d, $X_c$=%d', uint8(f*100), R, shadow_sigma, Xc)
        % title("true slf, %d% sampling, R=%d, sigma=10, Xc=30");
        header = sprintf('True $\mathbf{c}_1$')
        title('True PSD $\mathbf{c}_1$', 'Interpreter', 'latex');
        xlabel('(a2)', 'Interpreter', 'latex');
        % ylabel('PSD');
        
        nexttile    
        % subplot(232);
        plot(1:K, squeeze(C_true(:,2)), 'linewidth', 2, 'color', 'blue')
        set(gca,'XTick',[],'YTick',[])
        title('True PSD $\mathbf{c}_2$', 'Interpreter', 'latex');
        xlabel('(a3)', 'Interpreter', 'latex');

        
        nexttile(4)    
        % subplot(231);
        contourf(10*log10(T(:,:,k)), 100, 'linecolor', 'None');
        set(gca,'XTick',[],'YTick',[])
        % header = sprintf('true slf, %d , R=%d, $\sigma$=%d, $X_c$=%d', uint8(f*100), R, shadow_sigma, Xc)
        % title("true slf, %d% sampling, R=%d, sigma=10, Xc=30");
        title('Aggregated SLF at 5-th bin', 'Interpreter', 'latex');
        xlabel('(b1)', 'Interpreter', 'latex');
        % colormap jet;
        
        nexttile(5)    
        % subplot(232);
        contourf(10*log10(S_true(:,:,1)), 100, 'linecolor', 'None');
        set(gca,'XTick',[],'YTick',[])
        title('True $\mathbf{S}_1$','Interpreter', 'latex');
        xlabel('(b2)', 'Interpreter', 'latex');
        % colormap jet;

        nexttile(6)   
        % subplot(233);
        contourf(10*log10(S_true(:,:,2)), 100, 'linecolor', 'None');
        set(gca,'XTick',[],'YTick',[])
        title('True $\mathbf{S}_2$', 'Interpreter', 'latex');
        xlabel('(b3)', 'Interpreter', 'latex');
        % colormap jet;
        
        nexttile(7) 
        % subplot(234);
        contourf(10*log10(T_nmf(:,:,k)), 100, 'linecolor', 'None');
        set(gca,'XTick',[],'YTick',[]);
        sre1 = SRE(T, T_nmf);
        title('Estimated map SRE$=0.0840$', 'Interpreter', 'latex');
        xlabel('(c1) $\texttt{Nasdac}$', 'Interpreter', 'latex');
        % colormap jet;
        
        nexttile(8)    
        % subplot(235);
        contourf(10*log10(S_nmf(:,:,1)), 100, 'linecolor', 'None');
        set(gca,'XTick',[],'YTick',[]);
        naes1 = NAE(S_true(:,:,1), S_nmf(:,:,1),1); 
        title('Estimated $\widehat{\mathbf{S}}_1$ NAE$_{\mathbf{S}}=0.2230$', 'Interpreter', 'latex');
        xlabel('(c2) $\texttt{Nasdac}$', 'Interpreter', 'latex');
        % colormap jet;
        
        nexttile(9)  
        % subplot(236);
        contourf(real(10*log10(S_nmf(:,:,2))), 100, 'linecolor', 'None');
        set(gca,'XTick',[],'YTick',[])
        naes2 = NAE(S_true(:,:,2), S_nmf(:,:,2),1); 
        title('Estimated $\widehat{\mathbf{S}}_2$ NAE$_{\mathbf{S}}=0.2212$', 'Interpreter', 'latex');
        xlabel('(c3) $\texttt{Nasdac}$', 'Interpreter', 'latex');
        % colormap jet;

        nexttile(10)  
        % subplot(234);
        contourf(10*log10(T_joint(:,:,k)), 100, 'linecolor', 'None');
        set(gca,'XTick',[],'YTick',[]);
        sre2 = SRE(T, T_joint)
        title('Estimated map SRE$=0.0850$', 'Interpreter', 'latex');
        xlabel('(d1) $\texttt{DowJons}$', 'Interpreter', 'latex');
        % colormap jet;
        

        nexttile(11) 
        % subplot(235);
        contourf(10*log10(S_joint(:,:,1)), 100, 'linecolor', 'None');
        set(gca,'XTick',[],'YTick',[]);
        naes3 = NAE(S_true(:,:,1), S_joint(:,:,1),1); 
        title('Estimated $\widehat{\mathbf{S}}_1$ NAE$_{\mathbf{S}}=0.2452$', 'Interpreter', 'latex');
        xlabel('(d2) $\texttt{DowJons}$', 'Interpreter', 'latex');
        % colormap jet;
        
        nexttile(12) 
        % subplot(236);
        contourf(real(10*log10(S_joint(:,:,2))), 100, 'linecolor', 'None');
        set(gca,'XTick',[],'YTick',[])
        naes4 = NAE(S_true(:,:,2), S_joint(:,:,2),1); 
        title('Estimated $\widehat{\mathbf{S}}_2$ NAE$_{\mathbf{S}}=0.2089$', 'Interpreter', 'latex');
        xlabel('(d3) $\texttt{DowJons}$', 'Interpreter', 'latex');
        % colormap jet;
        
        cb = colorbar;
        cb.Label.String = 'dB';
        cb.Layout.Tile = 'east';
    end
    
    if show_psd
        figure(4);
        tiledlayout(3,2,  'Padding', 'none', 'TileSpacing', 'compact', 'Position', [0.08 0.09 0.9 0.85]); 

        nexttile    
        % subplot(231);
        plot(1:K, squeeze(C_true(:,1)), 'linewidth',2, 'linewidth', 2, 'color', 'red')
        % set(gca,'XTick',[],'YTick',[])
        % header = sprintf('true slf, %d , R=%d, $\sigma$=%d, $X_c$=%d', uint8(f*100), R, shadow_sigma, Xc)
        % title("true slf, %d% sampling, R=%d, sigma=10, Xc=30");
        header = sprintf('True $\mathbf{c}_1$')
        title('True $\mathbf{c}_1$', 'Interpreter', 'latex');
        ylabel('PSD');
        
        nexttile    
        % subplot(232);
        plot(1:K, squeeze(C_true(:,2)), 'linewidth', 2, 'color', 'blue')
        % set(gca,'XTick',[],'YTick',[])
        title('True $\mathbf{c}_2$', 'Interpreter', 'latex');
        
        nexttile    
        % subplot(233);
        plot(1:K, squeeze(C_nmf(:,1)), 'linewidth', 2, 'color', 'red')
        % set(gca,'XTick',[],'YTick',[])
        naec1 = NAE(C_true(:,1), C_nmf(:,1),1);
        title('\texttt{Nasdac} estimated $\widehat{\mathbf{c}}_2$ NAE$=0.0055$', 'Interpreter', 'latex');
        ylabel('PSD');
        
        
        nexttile    
        % subplot(234);
        plot(1:K, squeeze(C_nmf(:,2)), 'linewidth', 2, 'color', 'blue')
        % set(gca,'XTick',[],'YTick',[])
        naec2 = NAE(C_true(:,2), C_nmf(:,2),1);
        title('\texttt{Nasdac} estimated $\widehat{\mathbf{c}}_2$ NAE$=0.5542$', 'Interpreter', 'latex');
        
        nexttile    
        % subplot(233);
        plot(1:K, squeeze(C(:,1)), 'linewidth', 2, 'color', 'red')
        % set(gca,'XTick',[],'YTick',[])
        naec3 = NAE(C_true(:,1), C(:,1),1);
        title('\texttt{DowJons} estimated $\widehat{\mathbf{c}}_1$ NAE$=0.010$', 'Interpreter', 'latex');
        xlabel('frequency bins');
        ylabel('PSD');
        
        nexttile    
        % subplot(234);
        plot(1:K, squeeze(C(:,2)), 'linewidth', 2, 'color', 'blue')
        % set(gca,'XTick',[],'YTick',[])
        naec4 = NAE(C_true(:,2), C(:,2),1);
        title('\texttt{DowJons} estimated $\widehat{\mathbf{c}}_2$ NAE$=0.0510$', 'Interpreter', 'latex');
        xlabel('frequency bins');
        naec = NAE(C_true, C, R);
    end
end    

%%
show_plot = false;
if show_plot
    r = 48;
    figure(2);

    tiledlayout(3,4, 'Padding', 'none', 'TileSpacing', 'compact'); 
     
    max_limit = max(max(10*log10(Tr{1})));
    min_limit = min(min(10*log10(Tr{1})));
    
    nexttile    
    % subplot(231);
    contourf(10*log10(Tr{1}), 100, 'linecolor', 'None');
    caxis([min_limit max_limit])
    set(gca,'XTick',[],'YTick',[])
    % header = sprintf('true slf, %d , R=%d, $\sigma$=%d, $X_c$=%d', uint8(f*100), R, shadow_sigma, Xc)
    % title("true slf, %d% sampling, R=%d, sigma=10, Xc=30");
    header = sprintf('True Map: R=5')
    title(header);
    colormap jet;
    
    
    nexttile    
    % subplot(231);
    contourf(10*log10(Tr{2}), 100, 'linecolor', 'None');
    caxis([min_limit max_limit])
    set(gca,'XTick',[],'YTick',[])
    % header = sprintf('true slf, %d , R=%d, $\sigma$=%d, $X_c$=%d', uint8(f*100), R, shadow_sigma, Xc)
    % title("true slf, %d% sampling, R=%d, sigma=10, Xc=30");
    header = sprintf('R=8')
    title(header);
    colormap jet;
    
    
    nexttile    
    % subplot(231);
    contourf(10*log10(Tr{3}), 100, 'linecolor', 'None');
    caxis([min_limit max_limit])
    set(gca,'XTick',[],'YTick',[])
    % header = sprintf('true slf, %d , R=%d, $\sigma$=%d, $X_c$=%d', uint8(f*100), R, shadow_sigma, Xc)
    % title("true slf, %d% sampling, R=%d, sigma=10, Xc=30");
    header = sprintf('R=11')
    title(header);
    colormap jet;
    
    
    nexttile    
    % subplot(231);
    contourf(10*log10(Tr{4}), 100, 'linecolor', 'None');
    caxis([min_limit max_limit])
    set(gca,'XTick',[],'YTick',[])
    % header = sprintf('true slf, %d , R=%d, $\sigma$=%d, $X_c$=%d', uint8(f*100), R, shadow_sigma, Xc)
    % title("true slf, %d% sampling, R=%d, sigma=10, Xc=30");
    header = sprintf('R=14')
    title(header);
    colormap jet;
    
    nexttile    
    % subplot(232);
    contourf(10*log10(Tr_deep{1}), 100, 'linecolor', 'None');
    caxis([min_limit max_limit]);

    set(gca,'XTick',[],'YTick',[]);
    header = sprintf('Deep-Only');
    title(header);
    colormap jet;
    

    nexttile    
    % subplot(232);
    contourf(10*log10(Tr_deep{2}), 100, 'linecolor', 'None');
    caxis([min_limit max_limit]);

    set(gca,'XTick',[],'YTick',[]);
    header = sprintf('Deep-Only');
    % title(header);
    colormap jet;


    nexttile    
    % subplot(232);
    contourf(10*log10(Tr_deep{3}), 100, 'linecolor', 'None');
    caxis([min_limit max_limit]);

    set(gca,'XTick',[],'YTick',[]);
    header = sprintf('Deep-Only');
    % title(header);
    colormap jet;


    nexttile    
    % subplot(232);
    contourf(10*log10(Tr_deep{4}), 100, 'linecolor', 'None');
    caxis([min_limit max_limit]);

    set(gca,'XTick',[],'YTick',[]);
    header = sprintf('Deep-Only');
    % title(header);
    colormap jet;

    nexttile    
    % subplot(232);
    contourf(10*log10(Tr_nmf{1}), 100, 'linecolor', 'None');
    caxis([min_limit max_limit]);

    set(gca,'XTick',[],'YTick',[]);
    header = sprintf('Joint-Opt');
    title(header);
    colormap jet;
    

    nexttile    
    % subplot(232);
    contourf(10*log10(Tr_nmf{2}), 100, 'linecolor', 'None');
    caxis([min_limit max_limit]);

    set(gca,'XTick',[],'YTick',[]);
    header = sprintf('Joint-Opt');
    % title(header);
    colormap jet;


    nexttile    
    % subplot(232);
    contourf(10*log10(Tr_nmf{3}), 100, 'linecolor', 'None');
    caxis([min_limit max_limit]);

    set(gca,'XTick',[],'YTick',[]);
    header = sprintf('Joint-Opt');
    % title(header);
    colormap jet;


    nexttile    
    % subplot(232);
    contourf(10*log10(Tr_nmf{4}), 100, 'linecolor', 'None');
    caxis([min_limit max_limit]);

    set(gca,'XTick',[],'YTick',[]);
    header = sprintf('Joint-Opt');
    % title(header);
    colormap jet;

end

%%
misses = [          0.2323    0.0675    0.1017
    0.1741    0.1045    0.1240
    0.4184    0.1174    0.2017
    0.3976    0.1752    0.2442
    0.4062    0.1250    0.1552
    0.3818    0.1444    0.1937
    0.5519    0.1508    0.2492

 ];

misses = [    0.2085    0.0709    0.0709
    0.1712    0.0446    0.0647
    0.3187    0.1663    0.2430
    0.3290    0.1208    0.1907
    0.3712    0.2421    0.3238
    0.3933    0.1421    0.2074
    0.4919    0.1815    0.2760
];

misses = [     0.5360    0.2882    0.2688
    0.2554    0.1115    0.0952
    0.0977    0.0262    0.0292
    0.0244    0.0089    0.0038
    0.0213    0.0069    0.0007
];
rho = [1, 2.5, 5, 7.5, 10];


plot(rho, misses(:,1), '-s ', 'linewidth', 2)
legend('DeepComp');
hold on;

plot(rho, misses(:,2), 'linewidth', 2)
hold on;

plot(rho, misses(:,3), '-o ', 'linewidth', 2)
legend('\texttt{DeepComp}', '\texttt{Nasdac}', '\texttt{DowJons}', 'Interpreter', 'latex');

xlabel('$\rho(\%)$', 'Interpreter', 'latex')
ylabel('Miss detection probability', 'Interpreter', 'latex')
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