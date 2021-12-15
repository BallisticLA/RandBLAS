% Compile countSketch.c without or with blas
mex countSketch.c
%mex countSketch.c -DUSE_BLAS

% Compile countSketch_sparse.c
mex -largeArrayDims -O countSketch_sparse.c