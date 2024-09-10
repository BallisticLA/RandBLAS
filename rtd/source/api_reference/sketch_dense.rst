   .. |op| mathmacro:: \operatorname{op}
   .. |mat| mathmacro:: \operatorname{mat}
   .. |submat| mathmacro:: \operatorname{submat}
   .. |lda| mathmacro:: \texttt{lda}
   .. |ldb| mathmacro:: \texttt{ldb}
   .. |ldc| mathmacro:: \texttt{ldc}
   .. |opA| mathmacro:: \texttt{opA}
   .. |opB| mathmacro:: \texttt{opB}
   .. |opS| mathmacro:: \texttt{opS}
   .. |mtxA| mathmacro:: \mathbf{A}
   .. |mtxB| mathmacro:: \mathbf{B}
   .. |mtxC| mathmacro:: \mathbf{C}
   .. |mtxS| mathmacro:: \mathbf{S}
   .. |mtxX| mathmacro:: \mathbf{X}
   .. |mtxx| mathmacro:: \mathbf{x}
   .. |mtxy| mathmacro:: \mathbf{y}
   .. |ttt| mathmacro:: \texttt

******************************************
Working with dense data in RandBLAS
******************************************

.. TODO: add a few words about the data model.


Sketching dense matrices and vectors 
====================================

RandBLAS has adaptions of GEMM, GEMV, and SYMM when one of their matrix operands is a sketching operator.
These adaptations are provided through overloaded functions named sketch_general, sketch_vector, and sketch_symmetric.

Out of the functions presented here, only sketch_general has low-level implementations;
sketch_vector and sketch_symmetric are basic wrappers around sketch_general, and are provided
to make implementations less error-prone when porting code that currently uses BLAS
or a BLAS-like interface.



Analogs to GEMM
---------------

.. dropdown:: :math:`\mtxB = \alpha \cdot \op(\mtxS)\cdot \op(\mtxA) + \beta \cdot  \mtxB`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m, T alpha, SKOP &S, const T *A, int64_t lda, T beta, T *B, int64_t ldb)
      :project: RandBLAS

.. dropdown:: :math:`\mtxB = \alpha \cdot \op(\mtxA)\cdot \op(\mtxS) + \beta \cdot \mtxB`
  :animate: fade-in-slide-down
  :color: light

    .. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, SKOP &S, T beta, T *B, int64_t ldb)
      :project: RandBLAS

.. dropdown:: Variants using :math:`\op(\submat(\mtxS))`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m, T alpha, SKOP &S, int64_t S_ro, int64_t S_co, const T *A, int64_t lda, T beta, T *B, int64_t ldb)
      :project: RandBLAS

    .. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, SKOP &S, int64_t S_ro, int64_t S_co, T beta, T *B, int64_t ldb)
      :project: RandBLAS


Analogs to SYMM
---------------

.. dropdown:: :math:`\mtxB = \alpha \cdot \mtxS \cdot \mtxA + \beta \cdot \mtxB`
  :animate: fade-in-slide-down
  :color: light

    .. doxygenfunction:: RandBLAS::sketch_symmetric(blas::Layout layout, T alpha, SKOP &S, const T *A, int64_t lda, T beta, T *B, int64_t ldb, T sym_check_tol = 0)
      :project: RandBLAS

.. dropdown:: :math:`\mtxB = \alpha \cdot \mtxA \cdot \mtxS + \beta \cdot \mtxB`
  :animate: fade-in-slide-down
  :color: light

    .. doxygenfunction:: RandBLAS::sketch_symmetric(blas::Layout layout, T alpha, const T *A, int64_t lda, SKOP &S, T beta, T *B, int64_t ldb, T sym_check_tol = 0)
      :project: RandBLAS


.. dropdown:: Variants using  :math:`\submat(\mtxS)`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: RandBLAS::sketch_symmetric(blas::Layout layout, int64_t d, int64_t n, T alpha, SKOP &S, int64_t ro_s, int64_t co_s, const T *A, int64_t lda, T beta, T *B, int64_t ldb, T sym_check_tol = 0)
      :project: RandBLAS

    .. doxygenfunction:: RandBLAS::sketch_symmetric(blas::Layout layout, int64_t n, int64_t d, T alpha, const T *A, int64_t lda, SKOP &S, int64_t ro_s, int64_t co_s, T beta, T *B, int64_t ldb, T sym_check_tol = 0)
      :project: RandBLAS



Analogs to GEMV
---------------

.. dropdown:: :math:`\mtxy = \alpha \cdot \op(\mtxS) \cdot \mtxx + \beta \cdot \mtxy`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: sketch_vector(blas::Op opS, T alpha, SKOP &S, const T *x, int64_t incx, T beta, T *y, int64_t incy)
      :project: RandBLAS

.. dropdown:: Variants using :math:`\op(\submat(\mtxS))`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: sketch_vector(blas::Op opS, int64_t d, int64_t m, T alpha, SKOP &S, int64_t ro_s, int64_t co_s, const T *x, int64_t incx, T beta, T *y, int64_t incy)
      :project: RandBLAS


Matrix format utility functions
===============================

.. doxygenfunction:: RandBLAS::symmetrize(blas::Layout layout, blas::Uplo uplo, int64_t n, T* A, int64_t lda)
   :project: RandBLAS

.. doxygenfunction:: RandBLAS::transpose_square(T* A, int64_t n, int64_t lda)
   :project: RandBLAS

.. doxygenfunction:: RandBLAS::overwrite_triangle(blas::Layout layout, blas::Uplo to_overwrite, int64_t n, int64_t strict_offset,  T* A, int64_t lda)
   :project: RandBLAS
