function dist = KFJLT(X, Y, J, repl)
%KFJLT Estimate distance between pairs of Kronecker vectors using KFJLT
%   
%   dist = KFJLT(X, Y, J, repl) computes Kronecker fast
%   Johnson-Lindenstrauss transform (KFJLT) sketches of the column vectors
%   of khatrirao(X) and khatrirao(Y) with a target sketch dimension of J,
%   and then computes the distance between each corresponding vector in
%   khatrirao(X) and khatrirao(Y). These estimated distances are then
%   returned in the vector dist. Note that X and Y should be cells of
%   matrices, and J should be a positive integer. The argument repl is set
%   to true for sampling with replacement, and to false for sampling
%   without replacement. For more information on KFJLT, see [Ji19].
%
%   REFERENCES:
%   
%   [Ji19]  R Jin, TG Kolda, R Ward. Faster Johnson-Lindenstrauss 
%           Transforms via Kronecker Products. arXiv:1909.04801, 2019.

% Get degree, size and number of trials
degree          = length(X);
[sz, no_trials] = size(X{1});

% Construct Hadamard matrix and empty sketches
H           = hadamard(sz)/sqrt(sz);
X_sketched  = sqrt(sz^degree/J)*ones(J, no_trials);
Y_sketched  = sqrt(sz^degree/J)*ones(J, no_trials);

% If sampling is done without replacement, samples are computed here
if ~repl
    S = cell(degree,1);
    for d = 1:degree
        S{d} = zeros(J, no_trials);
    end

    for tr = 1:no_trials
        large_sample = randsample(sz^degree, J, false);
        for d = 1:degree
            S{d}(:, tr) = floor(mod(large_sample/sz^(degree+1-d), 1)*sz)+1;
        end
    end
end

% Compute sketches
for d = 1:degree
    % Mix
    D           = round(rand(sz, no_trials))*2-1;
    Xd_mixed    = H*(D.*X{d});
    Yd_mixed    = H*(D.*Y{d});
    
    % Sample
    for tr = 1:no_trials
        if repl
            S                   = randsample(sz, J, true);
            X_sketched(:, tr)   = X_sketched(:, tr).*Xd_mixed(S, tr);
            Y_sketched(:, tr)   = Y_sketched(:, tr).*Yd_mixed(S, tr);
        else
            X_sketched(:, tr)   = X_sketched(:, tr).*Xd_mixed(S{d}(:, tr), tr);
            Y_sketched(:, tr)   = Y_sketched(:, tr).*Yd_mixed(S{d}(:, tr), tr);
        end
    end
end

% Compute distances
dist    = sqrt(sum((X_sketched-Y_sketched).^2, 1));

end

