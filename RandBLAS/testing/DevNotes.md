# Notes for RandBLAS/testing

**None of the files in this directory are part of RandBLAS' public API.**

comparison.hh.

  This currently holds a single utility function for testing approximate-equality of floating point numbers.
  The function constructs an error message if the comparison fails.

lapack_like.hh.

  This has basic implementations of Cholesky, Householder transformations, and some eigenvalue methods.

linops.hh.

  This defines the basic ingredients in RandBLAS' correctness tests for linear operators.
  A given datatype needs to implement to_explicit_buffer, left_apply, and right_apply.
  This file also defines functions reference_left_apply and reference_right_apply,
  which compute an expected answer and a componentwise error tolerance of a given matrix-matrix product.

sparse_data.hh.

  Functions for generating (random) sparse matrices with various structures and formats.
  Some functions are used in correctness tests, while others are useful for benchmarking kernel performance.

stats.hh.

  Infrastructure for Kolmogrov-Smirnov statistical testing.
