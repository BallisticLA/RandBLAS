# Notes on functionality tested in test_matmul_wrappers


## Tests for sketch_sparse

Our tests for [L/R]SKSP3 are adaptations of the tests in ``test_matmul_cores/test_lskge3.cc``
and ``test_matmul_cores/test_rskge3.cc`` where the data matrix was the identity.

 * We only test COOMatrix for the sparse matrix datatype, but that's reasonable since the implementations
   of [L/R]SKSP3 are fully templated for the sparse matrix datatype.
 * These tests don't consider operating on submatrices of the data matrix.
   It's possible to do that in principle (at least for COOMatrix) but that's not necessary since logic of handling
   submatrices of the sparse data matrix is handled in left_spmm and right_spmm.

## Tests for sketch_symmetric
sketch_symmetric currently falls back on sketch_general, so it suffices to test with only DenseSkOp.

## Tests for sketch_vector
sketch_vector currently falls back on sketch_general, so it suffices to test with only DenseSkOp.

There's an argument to be made for sketch_vector to directly handle
dispatching of GEMV (for DenseSkOp) and an SPMV kernel (for SparseSkOp).
Additional tests would be warranted if we made that change.

*Note.* We have some infrastructure in place for SPMV,
in the forms of
    apply_csc_to_vector_from_left_ki
and
    apply_csr_to_vector_from_left_ik.
Those functions basically assume alpha = beta = 1.
