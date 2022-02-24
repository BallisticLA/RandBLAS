function dist = GS_tensor(X, Y, J)
%GS_tensor Estimate distance between two CP tensors using Gaussian sketch
%
%dist = GS_tensor(X, Y, J) computes an estimate of the distance between X
%and Y, which are in Tensor Toolbox CP tensor format. The estimate is done
%by applying a Gaussian sketch with J rows, after first reshaping the
%tensors into full matrices. This code requires Tensor Toolbox [Ba15]. For
%more information on Gaussian sketching, see e.g. [Wo14].
%
%REFERENCES:
%
%[Ba15]  BW Bader, TG Kolda and others. MATLAB Tensor Toolbox 
%        Version 2.6, Available online, February 2015. 
%        URL: http://www.sandia.gov/~tgkolda/TensorToolbox/.
%
%[Wo14]  DP Woodruff. Sketching as a Tool for Numerical Linear Algebra.
%        Foundations and Trends in Theoretical Computer Science 10(1-2),
%        pp. 1-157, 2014.

% Compute full-sized matrices
X_full  = khatrirao(X.U) .* X.lambda.';
Y_full  = khatrirao(Y.U) .* Y.lambda.';

% Compute sketches
S           = randn(J, size(X_full,1))/sqrt(J);
X_sketched  = S*X_full;
Y_sketched  = S*Y_full;

% Compute distance
dist = norm(sum(X_sketched,2) - sum(Y_sketched,2));

end