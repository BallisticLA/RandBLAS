% TEST_CS_FUNCTIONS Script used to test the different CountSketch functions
%
%   TEST_CS_FUNCTIONS is a script used to test the correctness of 
%   countSketch.c and countSketch_sparse.c. It generates a random sparse
%   matrix A and then applies the same CountSketch computation in Matlab,
%   in countSketch.c and both as a sparse and dense matrix in
%   countSketch_sparse.c.

% Author:   Osman Asif Malik
% Email:    osman.malik@colorado.edu
% Date:     January 29, 2019

% Settings
m = 100; % number of rows of A
n = 20; % number of columns of A
l = 40; % target sketch dimension
density = .1; % target density of A

% Generate test matrix and hash functions
A = sprand(m, n, density);
h = randi(l, m, 1);
s = randi(2, m, 1)*2-3;
tol = 1e-12;

% Compute CountSketch in Matlab
B_matlab = zeros(l, n);
Atemp = diag(s) * A;
for k = 1:m
    B_matlab(h(k), :) = B_matlab(h(k), :) + Atemp(k, :);
end

% Compute using countSketch.c
B_CS = countSketch(full(A'), int64(h), l, s, 1)';

% Compute using countSketch_sparse.c using sparse input
B_CS_sparse = countSketch_sparse(A', int64(h), l, s)';

% Compute using countSketch_sparse.c using dense input
B_CS_dense = countSketch_sparse(full(A'), int64(h), l, s)';

% Compare all computations
error_cs = norm(B_matlab - B_CS, 'fro');
error_cs_sparse = norm(B_matlab - B_CS_sparse, 'fro');
error_cs_dense = norm(B_matlab - B_CS_dense, 'fro');
fprintf('Frob. norm of B_matlab - B_CS: %.8e\n', error_cs);
fprintf('Frob. norm of B_matlab - B_CS_sparse: %.8e\n', error_cs_sparse);
fprintf('Frob. norm of B_matlab - B_CS_dense: %.8e\n', error_cs_dense);
if error_cs < tol && error_cs_sparse < tol && error_cs_dense < tol
    disp('Tests all OK!')
else
    disp('WARNING: Issue with with one of the tests!')
end
