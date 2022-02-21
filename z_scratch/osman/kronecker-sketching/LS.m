function dist = LS(X, Y, J)
%LS Estimate distance between pairs of Kronecker vectors using estimated
%leverage score sampling
%
%   dist = LS(X, Y, J) computes sketches of the column vectors of
%   khatrirao(X) and khatrirao(Y) with a target sketch dimension of J, and
%   then computes the distance between each corresponding vector in
%   khatrirao(X) and khatrirao(Y). The sketches are computed by using
%   estimated leverage score sampling. These estimated distances are then
%   returned in the vector dist. Note that X and Y should be cells of
%   matrices, and J should be a positive integer. For more info on the
%   estimated leverage score sampling approach, see [Ch16].
%
%   REFERENCES:
%   
%   [Ch16]  D Cheng, R Peng, I Perros, Y Liu. SPALS: Fast Alternating Least 
%           Squares via Implicit Leverage Scores Sampling. NeurIPS, 2016.

% Get degree, size and number of trials
degree          = length(X);
[sz, no_trials] = size(X{1});

% Construct empty sketches
X_sketched  = ones(J, no_trials)/sqrt(J);
Y_sketched  = ones(J, no_trials)/sqrt(J);

% Compute sketches
for tr = 1:no_trials
    for d = 1:degree
        [Q, ~]            = qr([X{d}(:, tr) Y{d}(:, tr)], 0);
        q                 = sum(Q.^2, 2);
        q                 = q/sum(q);
        S                 = randsample(sz, J, true, q);
        X_sketched(:, tr) = X_sketched(:, tr) .* X{d}(S, tr) ./ sqrt(q(S));
        Y_sketched(:, tr) = Y_sketched(:, tr) .* Y{d}(S, tr) ./ sqrt(q(S));
    end
end

% Compute distances
dist    = sqrt(sum( (X_sketched-Y_sketched).^2, 1 ));

end