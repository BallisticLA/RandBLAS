%EXPERIMENT1 Compare different sketch techniques on synthetic data
%
%   In this script we compare the performance of the following sketch
%   techniques applied to vectors with Kronecker structure:
%       - Standard Gaussian sketch (see e.g. [Wo14])
%       - Kronecker fast Johnson-Lindenstrauss transform (KFJLT) [Ji19]
%       - Tensor Randomized Projection (TRP) [Su18]
%       - TensorSketch [Di18]
%       - Estimated leverage score sampling [Ch16]
%
%   This script uses the khatrirao function from Tensor Toolbox [Ba15].
%
%   REFERENCES:
%
%   [Ba15]  BW Bader, TG Kolda and others. MATLAB Tensor Toolbox 
%           Version 2.6, Available online, February 2015. 
%           URL: http://www.sandia.gov/~tgkolda/TensorToolbox/.
%
%   [Ch16]  D Cheng, R Peng, I Perros, Y Liu. SPALS: Fast Alternating Least 
%           Squares via Implicit Leverage Scores Sampling. NeurIPS, 2016.
%
%   [Di18]  H Diao, Z Song, W Sun, DP Woodruff. Sketching for Kronecker 
%           Product Regression and P-splines. AISTATS, 2018.
%
%   [Ji19]  R Jin, TG Kolda, R Ward. Faster Johnson-Lindenstrauss 
%           Transforms via Kronecker Products. arXiv:1909.04801, 2019.
%
%   [Su18]  Y Sun, Y Guo, JA Tropp, M Udell. Tensor Random Projection for 
%           Low Memory Dimension Reduction. NeurIPS Workshop on Relational 
%           Representation Learning, 2019.
%
%   [Wo14]  DP Woodruff. Sketching as a Tool for Numerical Linear Algebra.
%           Foundations and Trends in Theoretical Computer Science 10(1-2),
%           pp. 1-157, 2014.

%% Settings

no_trials       = 1000;
degree          = 3;
sz              = 16;
embedding_dim   = 100:100:1000;
rand_vec_type   = 'normal'; 
no_sketches     = 5;
KFJLT_repl      = false;
sparse_nnz      = 3;

%% Create Kronecker vectors and compute true distances

X   = cell(degree,1);
Y   = cell(degree,1);
for d = 1:degree
    switch rand_vec_type
        case 'normal'
            X{d}    = randn(sz, no_trials);
            Y{d}    = randn(sz, no_trials);
        case 'sparse'
            % Sparse is somewhat adversarial to LS, TRP and TS
            X{d}    = zeros(sz, no_trials);
            Y{d}    = zeros(sz, no_trials);
            rnd_idx = @() randsample(sz, sparse_nnz, false);
            for tr = 1:no_trials
                X{d}(rnd_idx(), tr) = 1e+2*randn(sparse_nnz, 1);
                Y{d}(rnd_idx(), tr) = 1e+2*randn(sparse_nnz, 1);
            end
        case 'large-single'
            % This adversarial to LS, TRP and TS
            rids1   = randsample(sz, no_trials, true);
            rids2   = randsample(sz, no_trials, true);
            X{d}    = full(sparse(rids1, [1:no_trials], 1e+2*ones(no_trials,1)));
            Y{d}    = full(sparse(rids2, [1:no_trials], 1e+2*ones(no_trials,1)));            
        otherwise
            error('Invalid vec_type')
    end
end

X_full  = khatrirao(X);
Y_full  = khatrirao(Y);
dist    = sqrt(sum( (X_full-Y_full).^2, 1 ));
idx     = dist~=0; 

%% Compute and evaluate sketches

mean_dist   = zeros(no_sketches, length(embedding_dim));
std_dist    = zeros(no_sketches, length(embedding_dim));
max_dist    = zeros(no_sketches, length(embedding_dim));

for e_dim = 1:length(embedding_dim)
    
    J   = embedding_dim(e_dim);
    fprintf('Running experiments for J = %d\n', J);

    % Standard Gaussian sketch
    fprintf('\t Running Gaussian sketch...')
    gs_dist             = GS(X, Y, J);
    gs_dev              = abs(gs_dist(idx)./dist(idx) - 1);
    mean_dist(1, e_dim) = mean(gs_dev);
    std_dist(1, e_dim)  = std(gs_dev);
    max_dist(1, e_dim)  = max(gs_dev);
    fprintf(' Done!\n')
    
    % KFJLT sketch
    fprintf('\t Running KFJLT sketch...')
    kfjlt_dist          = KFJLT(X, Y, J, KFJLT_repl);
    kfjlt_dev           = abs(kfjlt_dist(idx)./dist(idx) - 1);
    mean_dist(2, e_dim) = mean(kfjlt_dev);
    std_dist(2, e_dim)  = std(kfjlt_dev);
    max_dist(2, e_dim)  = max(kfjlt_dev);
    fprintf(' Done!\n')
    
    % TRP sketch
    fprintf('\t Running TRP sketch...')
    trp_dist            = TRP(X, Y, J);
    trp_dev             = abs(trp_dist(idx)./dist(idx) - 1);
    mean_dist(3, e_dim) = mean(trp_dev);
    std_dist(3, e_dim)  = std(trp_dev);
    max_dist(3, e_dim)  = max(trp_dev);
    fprintf(' Done!\n')
    
    % TensorSketch
    fprintf('\t Running TensorSketch...')
    ts_dist             = TS(X, Y, J);
    ts_dev              = abs(ts_dist(idx)./dist(idx) - 1);
    mean_dist(4, e_dim) = mean(ts_dev);
    std_dist(4, e_dim)  = std(ts_dev);
    max_dist(4, e_dim)  = max(ts_dev);
    fprintf(' Done!\n')
    
    % Estimated leverage score sampling
    fprintf('\t Running leverage score sampling...')
    ls_dist             = LS(X, Y, J);
    ls_dev              = abs(ls_dist(idx)./dist(idx) - 1);
    mean_dist(5, e_dim) = mean(ls_dev);
    std_dist(5, e_dim)  = std(ls_dev);
    max_dist(5, e_dim)  = max(ls_dev);
    fprintf(' Done!\n')

end

%% Print and save results

fprintf('\nSaving results...')
save(['results_experiment1_', rand_vec_type], 'mean_dist', 'std_dist', 'max_dist');
fprintf(' Done!\n\n')

disp(mean_dist)
disp(' ')
disp(std_dist)
disp(' ')
disp(max_dist)