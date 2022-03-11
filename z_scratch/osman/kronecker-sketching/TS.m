function dist = TS(X, Y, J)
%TS Estimate distance between pairs of Kronecker vectors using TensorSketch
%   
%   dist = TS(X, Y, J) computes TensorSketches of the column vectors of
%   khatrirao(X) and khatrirao(Y) with a target sketch dimension of J, and
%   then computes the distance between each corresponding vector in
%   khatrirao(X) and khatrirao(Y). These estimated distances are then
%   returned in the vector dist. Note that X and Y should be cells of
%   matrices, and J should be a positive integer. For more info on
%   TensorSketch, see e.g. [Di18].
%
%   REFERENCES:
%   
%   [Di18]  H Diao, Z Song, W Sun, DP Woodruff. Sketching for Kronecker 
%           Product Regression and P-splines. AISTATS, 2018.

% Get degree, size and number of trials
degree          = length(X);
[~, no_trials]  = size(X{1});

% Construct empty sketches
X_sketched  = zeros(J, no_trials);
Y_sketched  = zeros(J, no_trials);

% Compute sketches
for tr = 1:no_trials
    % Set input
    input_mat = cell(degree,1);
    for d = 1:degree
        input_mat{d}    = [X{d}(:, tr) Y{d}(:, tr)];
    end
    
    % Do sketching
    SA                  = TensorSketch(input_mat, J);
    X_sketched(:, tr)   = SA(:, 1);
    Y_sketched(:, tr)   = SA(:, 2);
end

% Compute distances
dist    = sqrt(sum((X_sketched-Y_sketched).^2, 1));

end