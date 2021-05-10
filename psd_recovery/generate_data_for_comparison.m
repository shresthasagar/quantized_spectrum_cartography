function [X Sc Ctrue peaks] = generate_data_for_comparison(dB, K, R, shadow_sigma, Xc, structured_c, basis, separable)
    seed = sum(100*clock);
    s = RandStream('mt19937ar','Seed',seed,'NormalTransform','Polar');
    
    indK = [1:K]';
    if nargin <7
        basis = 's'; 
    end
    
    if basis == 'g'
        Sx = @(f0,sigma) exp(-(indK-f0).^2 /(2*sigma^2)); 
    else
        Sx =@(f0,a) sinc((indK-f0)/a).^2.*( abs((indK-f0)/a)<=1);
    end
    % ind_psd = 4:8:K-4;

    % % psd_peaks = ind_psd(randperm(length(ind_psd), 7));
    % psd_peaks = [ 44 60 20 28 12 52 4 ];

    % Ctrue=[];
    % % for rr = 1:R
    % %     am = 0.5 + 1.5*rand(3,1);
    % %     c{rr} = am(1)*Sx(psd_peaks((rr-1)*3+1),2+2*rand) + (rand>0.5)*am(2)*Sx(psd_peaks((rr-1)*3+2),2+2*rand) + (rand>0.5)*am(3)*Sx(psd_peaks((rr-1)*3+3),2+2*rand);
    % %     Ctrue = [Ctrue,c{rr}];
    % % end

    Ctrue=[];
    
    
    % if structured_c
    %     ind_psk = 24:1:K-2;
    %     for rr=1:R
    %         am = 0.5 + 1.5*rand(s,9,1);
    %         if separable
    %             c{rr} = am(1)*Sx(1+(rr-1)*3,2+2*rand(s));% + am(2)*Sx(28,2+2*rand(s)) + am(3)*Sx(40,2+2*rand(s)) + am(4)*Sx(48,2+2*rand(s));
    %         else
    %             c{rr} = am(1)*Sx(2,2+2*rand(s));
    %         end
            
    %         for q=1:6
    %             c{rr} = c{rr} + am(q)*Sx(24+q*4,2+2*rand(s));
    %         end
    %         Ctrue = [Ctrue,c{rr}];
    %     end
    % else
    %     ind_psd = 2:1:K-2;
    
    %     psd_peaks = ind_psd(randperm(length(ind_psd), 40));
    %     % psd_peaks = [ 44 60 20 28 12 52 4 ];
    
    %     for rr = 1:R
    %         am = 0.5 + 1.5*rand(s,3,1);
    %         c{rr} = am(1)*Sx(psd_peaks((rr-1)*3+1),2+2*rand(s)) + (rand(s)>0)*am(2)*Sx(psd_peaks((rr-1)*3+2),2+2*rand(s)) + (rand(s)>0)*am(3)*Sx(psd_peaks((rr-1)*3+3),2+2*rand(s));
    %         Ctrue = [Ctrue,c{rr}];
    %     end

    % end

    if structured_c
        ind_psk = 24:1:K-2;
        for rr=1:R
            am = 0.5+ 1.5*rand(s,9,1);
            if separable
                c{rr} = am(1)*Sx(5+(rr-1)*2,2+2*rand(s));% + am(2)*Sx(28,2+2*rand(s)) + am(3)*Sx(40,2+2*rand(s)) + am(4)*Sx(48,2+2*rand(s));
            else
                c{rr} = am(1)*Sx(8,3+2*rand(s));
            end
            
            for q=1:3
                c{rr} = c{rr} + am(q)*Sx(10+q*15,2+2*rand(s));
            end
            Ctrue = [Ctrue,c{rr}];
        end
    else
        ind_psd = 2:1:K-2;
    
        psd_peaks = ind_psd(randperm(length(ind_psd), 55));
        % psd_peaks = [ 44 60 20 28 12 52 4 ];
    
        for rr = 1:R
            am = 1.5 + 0.5*rand(s,4,1);

            c{rr} = 1.5*Sx(5,3+2*rand(s)) + am(1)*Sx(psd_peaks((rr-1)*3+1),3+2*rand(s)) + (rand(s)>0)*am(2)*Sx(psd_peaks((rr-1)*3+2),3+2*rand(s)) + (rand(s)>0)*am(3)*Sx(psd_peaks((rr-1)*3+3),3+2*rand(s)) + (rand(s)>0)*am(4)*Sx(psd_peaks((rr-1)*3+3),3+2*rand(s));

            Ctrue = [Ctrue,c{rr}];
        end

    end


    
    if structured_c
        ind_psd = 24:3:K-2;
        psd_peaks = ind_psd(randperm(length(ind_psd)), )
        for rr=1:R
            am = 0.5+ 1.5*rand(s,9,1);
            if separable
                c{rr} = am(1)*Sx(5+(rr-1)*2,2+2*rand(s));% + am(2)*Sx(28,2+2*rand(s)) + am(3)*Sx(40,2+2*rand(s)) + am(4)*Sx(48,2+2*rand(s));
            else
                c{rr} = am(1)*Sx(8,3+2*rand(s));
            end
            
            for q=1:3
                c{rr} = c{rr} + am(q)*Sx(10+q*15,2+2*rand(s));
            end
            Ctrue = [Ctrue,c{rr}];
        end
    else
        ind_psd = 2:1:K-2;
    
        psd_peaks = ind_psd(randperm(length(ind_psd), 55));
        % psd_peaks = [ 44 60 20 28 12 52 4 ];
    
        for rr = 1:R
            am = 1.5 + 0.5*rand(s,4,1);

            c{rr} = 1.5*Sx(5,3+2*rand(s)) + am(1)*Sx(psd_peaks((rr-1)*3+1),3+2*rand(s)) + (rand(s)>0)*am(2)*Sx(psd_peaks((rr-1)*3+2),3+2*rand(s)) + (rand(s)>0)*am(3)*Sx(psd_peaks((rr-1)*3+3),3+2*rand(s)) + (rand(s)>0)*am(4)*Sx(psd_peaks((rr-1)*3+3),3+2*rand(s));

            Ctrue = [Ctrue,c{rr}];
        end

    end

    
    Ctrue = ColumnNormalization(Ctrue);

    loss_f = @(x,d,alpha) min(1,(x/d).^(-alpha));
    d0 = 2;

    gridLen = 50;
    gridResolution = 1;%ja 
    x_grid = [0:gridResolution:gridLen];
    y_grid = [0:gridResolution:gridLen];
    [Xmesh_grid, Ymesh_grid] = meshgrid(x_grid, y_grid);
    Xgrid = Xmesh_grid + 1i*Ymesh_grid;
    % X1 = transpose(X1);
    [I,J] = size(Xgrid);


    Svec = [];
    % locations = [5+5i, 20+5i, 35+5i, 48+5i, 10+20i, 30+20i, 45+20i, 5+35i, 40+35i,  2+48i, 48+48i, 22+29i, 32+47i, 16+45i];

    % locations = locations(randperm(length(locations), R));
    for rr=1:R
        % location = 10+30*rand(s) + 1i*(10+30*rand(s));
        location = (50*rand(s)) + 1i*(50*rand(s));
        peaks{rr} = [uint8(real(location)), uint8(imag(location))];
        % location = locations(rr);
        loss_mat = abs(Xgrid - location);
        alpha = 2.1+0.5*rand;
        p = exp(-1/Xc);
        shadow = Shadowing(Xgrid,shadow_sigma,p);
        shadow_linear = 10.^(shadow/10);
        Sc{rr} = loss_f(loss_mat,d0,alpha).*shadow_linear;
        Sc{rr} = Sc{rr}/norm(Sc{rr},'fro');
        Svec = [Svec,Sc{rr}(:)];
    end

    if dB
        for rr=1:R
            Sc{rr} = real(10*log10(Sc{rr}));  % Remove all the fibers with nan entries along the column

        end
    end

    X = zeros(I,J,K);
    for rr=1:R
        X = X + outprod(Sc{rr},Ctrue(:,rr));
    end


    
end