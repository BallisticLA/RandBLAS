function dist = KFJLT_tensor(X, Y, J)
%KFJLT_tensor Estimate distance between two CP tensors using KFJLT
%
%dist = KFJLT_tensor(X, Y, J) computes an estimate of the distance between
%X and Y, which are in Tensor Toolbox CP tensor format. The estimate is
%done by applying a KFJLT sketch with J rows. This code requires Tensor
%Toolbox [Ba15].
%
%REFERENCES:
%
%[Ba15]  BW Bader, TG Kolda and others. MATLAB Tensor Toolbox Version 2.6,
%        Available online, February 2015. 
%        URL: http://www.sandia.gov/~tgkolda/TensorToolbox/.

% Get degree, dimension sizes and rank
degree  = length(X.U);
sz      = zeros(degree,1);
R       = size(X.U{1},2);
for d = 1:degree
    sz(d) = size(X.U{d},1);
end

% Construct mixing matrices and empty sketches
HD = cell(degree,1);
for d = 1:degree
    HD{d} = hadamard(sz(d))/sqrt(sz(d)) .* (round(rand(1,sz(d)))*2-1);
end
X_sketched  = sqrt(prod(sz)/J)*ones(J, R).*X.lambda.';
Y_sketched  = sqrt(prod(sz)/J)*ones(J, R).*Y.lambda.';

% Compute sampling without replacement
S               = cell(degree,1);
large_sample    = randsample(prod(sz), J, false);
for d = 1:degree
    S{d} = floor(mod((large_sample-1)/prod(sz(d+1:end)),sz(d)))+1;
end

% Mix tensor factor matrices
for d = 1:degree
    X.U{d}  = HD{d}*X.U{d};
    Y.U{d}  = HD{d}*Y.U{d};
end

% Compute sampled matrices corresponding to tensors
for d = 1:degree
    X_sketched  = X_sketched.*X.U{d}(S{d},:);
    Y_sketched  = Y_sketched.*Y.U{d}(S{d},:);
end

% Compute distance
dist = norm(sum(X_sketched,2) - sum(Y_sketched,2));

end