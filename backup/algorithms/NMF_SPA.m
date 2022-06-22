function [C, Sm] = NMF_SPA(T, R)
    % Alternative SPA on T transpose
    Tm = T';

    % column normalization
    [Tm_norm, Normalizer] = ColumnSumNormalization(Tm);
    Tm = Tm_norm;    

    indices_S = SPA(Tm, R);

    Sm = Tm(:,indices_S);

    % column unnormalization
    Sm = Sm.*Normalizer(:, indices_S);
    Tm = Tm.*Normalizer;

    % obtain the C matrix 
    pseudo_inverse_S = (inv(Sm'*Sm))*Sm';
    C = pseudo_inverse_S*Tm;
    C = C';

    C_p = ColumnPositive(C);
    C_p(C_p<0)=0;
    [C, d] = ColumnNormalization(C_p);

    Sm = Sm.*d;
    Sm = Sm';
end

function K = SPA(X,r)
    R = X; 
    normX = sum(X.^2); 
    normR = normX; 
    
    K = zeros(1,r);
    
    i = 1; 
    % Perform r steps (unless the residual is zero)
    while i <= r && max(normR) > 1e-12 
        
        % Select the column of R with the greatest 2-norm
        [~ , max_index] = max(sum(R.^2)); 
        K(i) = max_index;
        
        % unit vector u for the bth colum
        u = R(:,K(i))/norm(R(:,K(i))); 
        R = R - u*(u'*R);
        
        normR = sum(R.^2); 
        i = i + 1; 
    end
  
    K = K(1, 1:i-1);
    
end 

function [Y,D] = ColumnNormalization( X )
    %UNTITLED normalize each column of X
    %   To make its 2-norm to 1
    [m,n]=size(X);
    Y = zeros(m,n);
    D = zeros(1,n);
    
    for ii = 1:n
        D(ii) = norm(X(:,ii));
        if D(ii) ==0
            Y(:,ii) = X(:,ii);
        else
            Y(:,ii) = X(:,ii)/D(ii);
        end
         
    end
        
end
    
    