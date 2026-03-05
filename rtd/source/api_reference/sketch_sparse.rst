   .. |op| mathmacro:: \operatorname{op}
   .. |mat| mathmacro:: \operatorname{mat}
   .. |submat| mathmacro:: \operatorname{submat}
   .. |nrows| mathmacro:: \texttt{n_rows}
   .. |ncols| mathmacro:: \texttt{n_cols}
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
   .. |mtxP| mathmacro:: \mathbf{P}
   .. |mtxI| mathmacro:: \mathbf{I}
   .. |ttt| mathmacro:: \texttt

************************************
Working with sparse data in RandBLAS
************************************

Sparse matrix data structures
==============================


.. dropdown:: The common interface for our sparse matrix types
    :animate: fade-in-slide-down
    :color: light

    .. doxygenenum:: RandBLAS::sparse_data::IndexBase
        :project: RandBLAS

    .. doxygenconcept:: RandBLAS::sparse_data::SparseMatrix
        :project: RandBLAS

.. dropdown:: COOMatrix
    :animate: fade-in-slide-down
    :color: light

    .. doxygenenum:: RandBLAS::sparse_data::NonzeroSort
        :project: RandBLAS

    .. doxygenstruct:: RandBLAS::sparse_data::COOMatrix
        :project: RandBLAS
        :members:

.. dropdown:: CSRMatrix
    :animate: fade-in-slide-down
    :color: light

    .. doxygenstruct:: RandBLAS::sparse_data::CSRMatrix
        :project: RandBLAS
        :members:

.. dropdown:: CSCMatrix
    :animate: fade-in-slide-down
    :color: light

    .. doxygenstruct:: RandBLAS::sparse_data::CSCMatrix
        :project: RandBLAS
        :members:


Operations with sparse matrices
===============================

Sketching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: :math:`\mtxB = \alpha \cdot \op(\submat(\mtxS))\cdot \op(\mtxA) + \beta \cdot \mtxB`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: RandBLAS::sketch_sparse(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m, T alpha, const DenseSkOp &S, int64_t S_ro, int64_t S_co, const SpMat &A, T beta, T *B, int64_t ldb) 
      :project: RandBLAS

.. dropdown:: :math:`\mtxB = \alpha \cdot \op(\mtxA)\cdot \op(\submat(\mtxS)) + \beta \cdot \mtxB`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: RandBLAS::sketch_sparse(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n, T alpha, const SpMat &A, const DenseSkOp &S, int64_t S_ro, int64_t S_co, T beta, T *B, int64_t ldb) 
      :project: RandBLAS


Deterministic operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: :math:`\mtxC = \alpha \cdot \op(\mtxA)\cdot \op(\mtxB) + \beta \cdot  \mtxC,` with sparse :math:`\mtxA`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: RandBLAS::spmm(blas::Layout layout, blas::Op opA, blas::Op opB, int64_t m, int64_t n, int64_t k, T alpha, const SpMat &A, const T *B, int64_t ldb, T beta, T *C, int64_t ldc)  
      :project: RandBLAS

.. dropdown:: :math:`\mtxC = \alpha \cdot \op(\mtxA)\cdot \op(\mtxB) + \beta \cdot  \mtxC,` with sparse :math:`\mtxB`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: RandBLAS::spmm(blas::Layout layout, blas::Op opA, blas::Op opB, int64_t m, int64_t n, int64_t k, T alpha, const T* A, int64_t lda, const SpMat &B, T beta, T *C, int64_t ldc) 
      :project: RandBLAS

.. dropdown:: :math:`\mtxC = \alpha \cdot \op(\mtxA)\cdot \mtxB + \beta \cdot  \mtxC,` with sparse :math:`\mtxA` and sparse :math:`\mtxB` (SpGEMM)
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: RandBLAS::spgemm(blas::Layout layout, blas::Op opA, T alpha, const SpMat1 &A, const SpMat2 &B, T beta, T *C, int64_t ldc)
      :project: RandBLAS

    .. note::

       This function requires Intel MKL and only supports single and double precision (``float`` and ``double``),
       in contrast to other RandBLAS kernels that work with any scalar type.

.. dropdown:: :math:`\mtxB = \alpha \cdot \op(\mtxA)^{-1} \cdot \mtxB,` with sparse triangular :math:`\mtxA`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: RandBLAS::sparse_data::trsm(blas::Layout layout, blas::Op opA, T alpha, const SpMat &A, blas::Uplo uplo, blas::Diag diag, int64_t n, T *B, int64_t ldb, int validation_mode = 1)
      :project: RandBLAS
