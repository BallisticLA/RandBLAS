function dist = TS_tensor(X, Y, J)
%TS_tensor Estimate distance between two CP tensors using TensorSketch
%
%dist = TS_tensor(X, Y, J) computes an estimate of the distance between X
%and Y, which are in Tensor Toolbox CP tensor format. The estimate is done
%by applying a TensorSketch with J rows. This code requires Tensor Toolbox
%[Ba15]. For more info on TensorSketch, see e.g. [Di18].
%
%REFERENCES:
%
%[Ba15]  BW Bader, TG Kolda and others. MATLAB Tensor Toolbox Version 2.6,
%        Available online, February 2015. 
%        URL: http://www.sandia.gov/~tgkolda/TensorToolbox/.
%
%[Di18]  H Diao, Z Song, W Sun, DP Woodruff. Sketching for Kronecker 
%        Product Regression and P-splines. AISTATS, 2018.

% Get degree and rank
degree  = length(X.U);
R       = size(X.U{1},2);

% Compute input matrix for TensorSketch
input_mat   = cell(degree,1);
X.U{1}      = X.U{1} .* X.lambda.';
Y.U{1}      = Y.U{1} .* Y.lambda.';
for d = 1:degree
    input_mat{d} = [X.U{d} Y.U{d}];
end

% Compute sketches
SA          = TensorSketch(input_mat, J);
X_sketched  = SA(:,1:R);
Y_sketched  = SA(:,R+1:end);

% Compute distance
dist = norm(sum(X_sketched,2) - sum(Y_sketched,2));

end