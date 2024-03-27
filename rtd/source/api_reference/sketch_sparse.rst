******************************
Sketching sparse data
******************************




Representing sparse matrices
============================


Dense sketches of sparse data
=============================


Sketching with a submatrix from the left
----------------------------------------------------------------------

.. doxygenfunction:: RandBLAS::sketch_sparse(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m, T alpha, DenseSkOp<T,RNG> &S, int64_t S_ro, int64_t S_co, SpMatrix &A, int64_t A_ro, int64_t A_co, T beta, T *B, int64_t ldb) 
  :project: RandBLAS

Sketching with a submatrix from the right
----------------------------------------------------------------------
