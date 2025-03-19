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

    .. doxygenstruct:: RandBLAS::sparse_data::COOMatrix
        :project: RandBLAS
        :members:

    .. doxygenfunction:: RandBLAS::sparse_data::reserve_coo
        :project: RandBLAS

    .. doxygenenum:: RandBLAS::sparse_data::NonzeroSort
        :project: RandBLAS

.. dropdown:: CSRMatrix
    :animate: fade-in-slide-down
    :color: light

    .. doxygenstruct:: RandBLAS::sparse_data::CSRMatrix
        :project: RandBLAS
        :members:

    .. doxygenfunction:: RandBLAS::sparse_data::reserve_csr
        :project: RandBLAS

.. dropdown:: CSCMatrix
    :animate: fade-in-slide-down
    :color: light

    .. doxygenstruct:: RandBLAS::sparse_data::CSCMatrix
        :project: RandBLAS
        :members:

    .. doxygenfunction:: RandBLAS::sparse_data::reserve_csc
        :project: RandBLAS


Operations with sparse matrices
===============================

Sketching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: :math:`\mtxB = \alpha \cdot \op(\submat(\mtxS))\cdot \op(\mtxA) + \beta \cdot \mtxB`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: RandBLAS::sketch_sparse(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m, T alpha, DenseSkOp &S, int64_t S_ro, int64_t S_co, SpMat &A, T beta, T *B, int64_t ldb) 
      :project: RandBLAS

.. dropdown:: :math:`\mtxB = \alpha \cdot \op(\mtxA)\cdot \op(\submat(\mtxS)) + \beta \cdot \mtxB`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: RandBLAS::sketch_sparse(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n, T alpha, SpMat &A, DenseSkOp &S, int64_t S_ro, int64_t S_co, T beta, T *B, int64_t ldb) 
      :project: RandBLAS


Deterministic operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: :math:`\mtxC = \alpha \cdot \op(\mtxA)\cdot \op(\mtxB) + \beta \cdot  \mtxC,` with sparse :math:`\mtxA`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: RandBLAS::spmm(blas::Layout layout, blas::Op opA, blas::Op opB, int64_t m, int64_t n, int64_t k, T alpha, SpMat &A, const T *B, int64_t ldb, T beta, T *C, int64_t ldc)  
      :project: RandBLAS

.. dropdown:: :math:`\mtxC = \alpha \cdot \op(\mtxA)\cdot \op(\mtxB) + \beta \cdot  \mtxC,` with sparse :math:`\mtxB`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: RandBLAS::spmm(blas::Layout layout, blas::Op opA, blas::Op opB, int64_t m, int64_t n, int64_t k, T alpha, const T* A, int64_t lda, SpMat &B, T beta, T *C, int64_t ldc) 
      :project: RandBLAS

.. dropdown:: :math:`\mtxB = \alpha \cdot \op(\mtxA)^{-1} \cdot \mtxB,` with sparse triangular :math:`\mtxA`
    :animate: fade-in-slide-down
    :color: light

    .. doxygenfunction:: RandBLAS::sparse_data::trsm(blas::Layout layout, blas::Op opA, T alpha, const SpMat &A, blas::Uplo uplo, blas::Diag diag, int64_t n, T *B, int64_t ldb, int validation_mode = 1)
      :project: RandBLAS
