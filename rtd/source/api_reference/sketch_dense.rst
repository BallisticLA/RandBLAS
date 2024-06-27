
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

RandBLAS has adaptions of GEMM, GEMV, and SYMM when one of their matrix operands is a sketching operator.
These adaptations are provided through overloaded functions named sketch_general, sketch_vector, and sketch_symmetric.

Out of the functions presented here, only sketch_general has low-level implementations;
sketch_vector and sketch_symmetric are basic wrappers around sketch_general, and are provided to make
to make implementations less error-prone for when porting code that currently uses BLAS
or a BLAS-like interface.

Analogs to GEMM
===============

.. dropdown:: :math:`B = \alpha \cdot \op(S)\cdot \op(A) + \beta \cdot  B`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m, T alpha, SKOP &S, const T *A, int64_t lda, T beta, T *B, int64_t ldb)
      :project: RandBLAS

.. dropdown:: :math:`B = \alpha \cdot \op(A)\cdot \op(S) + \beta \cdot B`
  :animate: fade-in-slide-down
  :color: light

    .. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, SKOP &S, T beta, T *B, int64_t ldb)
      :project: RandBLAS

.. dropdown:: Variants using :math:`\op(\submat(S))`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m, T alpha, SKOP &S, int64_t S_ro, int64_t S_co, const T *A, int64_t lda, T beta, T *B, int64_t ldb)
      :project: RandBLAS

    .. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, SKOP &S, int64_t S_ro, int64_t S_co, T beta, T *B, int64_t ldb)
      :project: RandBLAS


Analogs to SYMM
===============

.. dropdown:: :math:`B = \alpha \cdot S \cdot A + \beta \cdot B`
  :animate: fade-in-slide-down
  :color: light

    .. doxygenfunction:: RandBLAS::sketch_symmetric(blas::Layout layout, T alpha, SKOP &S, const T *A, int64_t lda, T beta, T *B, int64_t ldb, T sym_check_tol = 0)
      :project: RandBLAS

.. dropdown:: :math:`B = \alpha \cdot A \cdot S + \beta \cdot B`
  :animate: fade-in-slide-down
  :color: light

    .. doxygenfunction:: RandBLAS::sketch_symmetric(blas::Layout layout, T alpha, const T *A, int64_t lda, SKOP &S, T beta, T *B, int64_t ldb, T sym_check_tol = 0)
      :project: RandBLAS


.. dropdown:: Variants using  :math:`\submat(S)`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: RandBLAS::sketch_symmetric(blas::Layout layout, int64_t d, int64_t n, T alpha, SKOP &S, int64_t ro_s, int64_t co_s, const T *A, int64_t lda, T beta, T *B, int64_t ldb, T sym_check_tol = 0)
      :project: RandBLAS

    .. doxygenfunction:: RandBLAS::sketch_symmetric(blas::Layout layout, int64_t n, int64_t d, T alpha, const T *A, int64_t lda, SKOP &S, int64_t ro_s, int64_t co_s, T beta, T *B, int64_t ldb, T sym_check_tol = 0)
      :project: RandBLAS



Analogs to GEMV
===============

.. dropdown:: :math:`y = \alpha \cdot \op(S) \cdot x + \beta \cdot y`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: sketch_vector(blas::Op opS, T alpha, SKOP &S, const T *x, int64_t incx, T beta, T *y, int64_t incy)
      :project: RandBLAS

.. dropdown:: Variants using :math:`\op(\submat(S))`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: sketch_vector(blas::Op opS, int64_t d, int64_t m, T alpha, SKOP &S, int64_t ro_s, int64_t co_s, const T *x, int64_t incx, T beta, T *y, int64_t incy)
      :project: RandBLAS

