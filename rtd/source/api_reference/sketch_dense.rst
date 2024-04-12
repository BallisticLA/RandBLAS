
 .. |op| mathmacro:: \operatorname{op}
 .. |mat| mathmacro:: \operatorname{mat}
 .. |submat| mathmacro:: \operatorname{submat}
 .. |lda| mathmacro:: \texttt{lda}
 .. |ldb| mathmacro:: \texttt{ldb}
 .. |opA| mathmacro:: \texttt{opA}
 .. |opS| mathmacro:: \texttt{opS}

******************************************
Computing a sketch: dense data
******************************************

Full matrix-matrix operations
=============================

.. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m, T alpha, SKOP &S, const T *A, int64_t lda, T beta, T *B, int64_t ldb)
   :project: RandBLAS

.. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, SKOP &S, T beta, T *B, int64_t ldb)
   :project: RandBLAS


Full matrix-vector operations
=============================

.. doxygenfunction:: sketch_vector(blas::Op opS, T alpha, SKOP &S, const T *x, int64_t incx, T beta, T *y, int64_t incy)
   :project: RandBLAS


Submatrix operations
====================

.. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m, T alpha, SKOP &S, int64_t S_ro, int64_t S_co, const T *A, int64_t lda, T beta, T *B, int64_t ldb)
  :project: RandBLAS

.. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, SKOP &S, int64_t S_ro, int64_t S_co, T beta, T *B, int64_t ldb)
  :project: RandBLAS

.. doxygenfunction:: sketch_vector(blas::Op opS, int64_t d, int64_t m, T alpha, SKOP &S, int64_t ro_s, int64_t co_s, const T *x, int64_t incx, T beta, T *y, int64_t incy)
  :project: RandBLAS