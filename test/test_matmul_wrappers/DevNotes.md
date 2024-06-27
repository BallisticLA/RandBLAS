# Notes on functionality tested in test_matmul_wrappers

## Tests for sketch_sparse


## Tests for sketch_symmetric


## Tests for sketch_vector
Right now sketch_vector falls back on sketch_general.
There's an argument to be made for it to directly handle
dispatching of GEMV (for DenseSkOp) and SPMV (for SparseSkOp).
Additional tests would be warranted if we made that change.

*Note.* We have some infrastructure in place for SPMV,
in the forms of
    apply_csc_to_vector_from_left_ki
and
    apply_csr_to_vector_from_left_ik.
Those functions basically assume alpha = beta = 1.
