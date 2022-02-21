function dist = TRP(X, Y, J)
%TRP Estimate distance between pairs of Kronecker vectors using TRP
%   
%   dist = TRP(X, Y, J) computes tensor random projection (TRP) sketches of
%   the column vectors of khatrirao(X) and khatrirao(Y) with a target 
%   sketch dimension of J, and then computes the distance between each 
%   corresponding vector in khatrirao(X) and khatrirao(Y). These estimated 
%   distances are then returned in the vector dist. Note that X and Y
%   should be cells of matrices, and J should be a positive integer. For
%   more info on TRP, see [Su18].
%
%   REFERENCES:
%   
%   [Su18]  Y Sun, Y Guo, JA Tropp, M Udell. Tensor Random Projection for 
%           Low Memory Dimension Reduction. NeurIPS Workshop on Relational 
%           Representation Learning, 2019.

% Get degree, size and number of trials
degree          = length(X);
[sz, no_trials] = size(X{1});

% Construct empty sketches
X_sketched  = ones(J, no_trials)/sqrt(J);
Y_sketched  = ones(J, no_trials)/sqrt(J);

% Compute sketches
for d = 1:degree
    for tr = 1:no_trials
        S                   = randn(J, sz);
        Xd_sketched         = S*X{d}(:, tr);
        Yd_sketched         = S*Y{d}(:, tr);
        X_sketched(:, tr)   = X_sketched(:, tr).*Xd_sketched;
        Y_sketched(:, tr)   = Y_sketched(:, tr).*Yd_sketched;
    end
end

% Compute distances
dist    = sqrt(sum((X_sketched-Y_sketched).^2, 1));

end