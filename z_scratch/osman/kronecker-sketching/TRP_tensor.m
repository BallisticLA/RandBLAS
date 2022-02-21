function dist = TRP_tensor(X, Y, J)
%TRP_tensor Estimate distance between two CP tensors using TRP
%
%dist = TRP_tensor(X, Y, J) computes an estimate of the distance between X
%and Y, which are in Tensor Toolbox CP tensor format. The estimate is done
%by applying a TRP sketch with J rows. This code requires Tensor Toolbox
%[Ba15]. For more info on TRP, see [Su18].
%
%REFERENCES:
%
%[Ba15]  BW Bader, TG Kolda and others. MATLAB Tensor Toolbox Version 2.6,
%        Available online, February 2015. 
%        URL: http://www.sandia.gov/~tgkolda/TensorToolbox/.
%
%[Su18]  Y Sun, Y Guo, JA Tropp, M Udell. Tensor Random Projection for Low
%        Memory Dimension Reduction. NeurIPS Workshop on Relational
%        Representation Learning, 2019.

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
    S           = randn(J, sz(d));
    X_sketched  = X_sketched .* (S*X.U{d});
    Y_sketched  = Y_sketched .* (S*Y.U{d});
end

% Compute distance
dist = norm(sum(X_sketched,2) - sum(Y_sketched,2));

end