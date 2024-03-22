*****************************
Applying a sketching operator
*****************************


Basic sketching from the left
=============================

.. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m, T alpha, SKOP &S, const T *A, int64_t lda, T beta, T *B, int64_t ldb)
   :project: RandBLAS

Basic sketching from the right
==============================

.. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, SKOP &S, T beta, T *B, int64_t ldb)
   :project: RandBLAS


Advanced material
=================


Applying a submatrix of a sketching operator from the left
----------------------------------------------------------

.. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m, T alpha, SKOP &S, int64_t S_ro, int64_t S_co, const T *A, int64_t lda, T beta, T *B, int64_t ldb)
   :project: RandBLAS


Applying a submatrix of a sketching operator from the right
-----------------------------------------------------------

.. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, SKOP &S, int64_t S_ro, int64_t S_co, T beta, T *B, int64_t ldb)
   :project: RandBLAS
