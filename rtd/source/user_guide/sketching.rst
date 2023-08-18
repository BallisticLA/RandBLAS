Applying a sketching operator : a GEMM-like interface
=====================================================

RandBLAS currently only supports sketching matrices that would be called "general" in BLAS.
This is done with an overloaded and templated function called :math:`\texttt{sketch_general}`.
The order of arguments provided to :math:`\texttt{sketch_general}` is very important; it
determines whether the sketching operator is applied from the left or from the right.

Sketching from the left
-----------------------

.. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m, T alpha, SKOP &S, int64_t i_off, int64_t j_off, const T *A, int64_t lda, T beta, T *B, int64_t ldb)
   :project: RandBLAS

Sketching from the right
------------------------

.. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, SKOP &S, int64_t i_off, int64_t j_off, T beta, T *B, int64_t ldb)
   :project: RandBLAS