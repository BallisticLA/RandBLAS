%EXPERIMENT2 Compare different sketch techniques on MNIST
%
%In this script, we compare how well the following four sketch techniques
%work when it comes to estimating the distance between two tensors in CP
%format:
%   - Kronecker fast Johnson-Lindenstrauss transform (KFJLT) [Ji19]
%   - Tensor Randomized Projection (TRP) [Su18]
%   - TensorSketch [Di18]
%   - Estimated leverage score sampling [Ch16]
%
%This script requires Tensor Toolbox [Ba15].
%
%The data in mnist.mat is downloaded via the scripts provided at
%https://github.com/sunsided/mnist-matlab.
%
%REFERENCES:
%
%[Ba15]  BW Bader, TG Kolda and others. MATLAB Tensor Toolbox 
%        Version 2.6, Available online, February 2015. 
%        URL: http://www.sandia.gov/~tgkolda/TensorToolbox/.
%
%[Ch16]  D Cheng, R Peng, I Perros, Y Liu. SPALS: Fast Alternating Least 
%        Squares via Implicit Leverage Scores Sampling. NeurIPS, 2016.
%
%[Di18]  H Diao, Z Song, W Sun, DP Woodruff. Sketching for Kronecker 
%        Product Regression and P-splines. AISTATS, 2018.
%
%[Ji19]  R Jin, TG Kolda, R Ward. Faster Johnson-Lindenstrauss 
%        Transforms via Kronecker Products. arXiv:1909.04801, 2019.
%
%[Su18]  Y Sun, Y Guo, JA Tropp, M Udell. Tensor Random Projection for 
%        Low Memory Dimension Reduction. NeurIPS Workshop on Relational 
%        Representation Learning, 2019.

%% Settings

data_loc        = '../mnist-matlab/mnist.mat';
no_slices       = 128;
pad_size        = 2;
R               = 10;
embedding_dim   = 100:100:5e+3; %round(linspace(1e+2, 3e+4, 20));
no_sketches     = 4;
no_trials       = 1000;

%% Load and pad data

load(data_loc);

data_4 = training.images(:,:,training.labels==4);
data_4 = data_4(:,:,1:no_slices); 
data_4 = padarray(data_4, [pad_size pad_size], 0, 'both');

data_9 = training.images(:,:,training.labels==9);
data_9 = data_9(:,:,1:no_slices); 
data_9 = padarray(data_9, [pad_size pad_size], 0, 'both');

%% Compute CP decompositions and true distance

cp_4 = cp_als(tensor(data_4), R);
cp_9 = cp_als(tensor(data_9), R);
true_dist = norm(tensor(cp_4) - tensor(cp_9));

%% Compute and evaluate sketches

dist = zeros(no_sketches, length(embedding_dim), no_trials);

for e_dim = 1:length(embedding_dim)
    
    J = embedding_dim(e_dim);
    fprintf('Running experiments for J = %d\n', J);
    
    for tr = 1:no_trials
        % KFJLT sketch
        dist(1, e_dim, tr) = KFJLT_tensor(cp_4, cp_9, J);

        % TRP sketch
        dist(2, e_dim, tr) = TRP_tensor(cp_4, cp_9, J);

        % TensorSketch
        dist(3, e_dim, tr) = TS_tensor(cp_4, cp_9, J);

        % Estimated leverage score sampling
        dist(4, e_dim, tr) = LS_tensor(cp_4, cp_9, J);
    end
    
end

%% Print and save results

fprintf('\nSaving results...')
save('results_experiment2', 'dist', 'true_dist');
fprintf(' Done!\n\n')

disp(dist)