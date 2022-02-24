function dist = LS_tensor(X, Y, J)
%LS_tensor Estimate distance between two CP tensors using estimated
%leverage score sampling
%
%dist = LS_tensor(X, Y, J) computes an estimate of the distance between X
%and Y, which are in Tensor Toolbox CP tensor format. The estimate is done
%by applying estimated leverage score sampling with J rows. This code
%requires Tensor Toolbox [Ba15]. For more info on the estimated leverage
%score sampling approach, see [Ch16].
%
%REFERENCES:
%
%[Ba15]  BW Bader, TG Kolda and others. MATLAB Tensor Toolbox Version 2.6,
%        Available online, February 2015. 
%        URL: http://www.sandia.gov/~tgkolda/TensorToolbox/.
%   
%[Ch16]  D Cheng, R Peng, I Perros, Y Liu. SPALS: Fast Alternating Least 
%        Squares via Implicit Leverage Scores Sampling. NeurIPS, 2016.

% Get degree, dimension sizes and rank
degree  = length(X.U);
sz      = zeros(degree,1);
R       = size(X.U{1},2);
for d = 1:degree
    sz(d) = size(X.U{d},1);
end

% Construct empty sketches
X_sketched  = ones(J, R)/sqrt(J) .* X.lambda.';
Y_sketched  = ones(J, R)/sqrt(J) .* Y.lambda.';

% Compute sketches
for d = 1:degree
    [Q, ~]      = qr([X.U{d} Y.U{d}], 0);
    q           = sum(Q.^2, 2);
    q           = q/sum(q);
    S           = randsample(sz(d), J, true, q);
    X_sketched  = X_sketched .* X.U{d}(S,:) ./ sqrt(q(S));
    Y_sketched  = Y_sketched .* Y.U{d}(S,:) ./ sqrt(q(S));
end

% Compute distance
dist = norm(sum(X_sketched,2) - sum(Y_sketched,2));

end