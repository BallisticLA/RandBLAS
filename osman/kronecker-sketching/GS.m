function dist = GS(X, Y, J)
%GS Estimate distance between pairs of Kronecker vectors using Gaussian
%sketch
%
%   dist = GS(X, Y, J) computes Gaussian sketches of the column vectors of
%   khatrirao(X) and khatrirao(Y) with a target sketch dimension of J, and
%   then computes the distance between each corresponding vector in
%   khatrirao(X) and khatrirao(Y). These estimated distances are then
%   returned in the vector dist. Note that X and Y should be cells of
%   matrices, and J should be a positive integer. For more information on
%   Gaussian sketching, see e.g. [Wo14].
%
%   This function uses the khatrirao function from Tensor Toolbox [Ba15].
%
%   REFERENCES:
%
%   [Ba15]  BW Bader, TG Kolda and others. MATLAB Tensor Toolbox 
%           Version 2.6, Available online, February 2015. 
%           URL: http://www.sandia.gov/~tgkolda/TensorToolbox/.
%
%   [Wo14]  DP Woodruff. Sketching as a Tool for Numerical Linear Algebra.
%           Foundations and Trends in Theoretical Computer Science 10(1-2),
%           pp. 1-157, 2014.

% Get degree, size and number of trials
degree          = length(X);
[sz, no_trials] = size(X{1});

% Compute full-sized matrices and construct empty sketches
X_full      = khatrirao(X);
Y_full      = khatrirao(Y);
X_sketched  = zeros(J, no_trials);
Y_sketched  = zeros(J, no_trials);

% Compute sketches
for tr = 1:no_trials
    S                   = randn(J, sz^degree)/sqrt(J);
    X_sketched(:, tr)   = S*X_full(:, tr);
    Y_sketched(:, tr)   = S*Y_full(:, tr);
end

% Compute distances
dist    = sqrt(sum( (X_sketched-Y_sketched).^2, 1 ));

end