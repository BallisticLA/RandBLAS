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

    .. doxygenconcept:: RandBLAS::sparse_data::SparseMatrix
        :project: RandBLAS


.. dropdown:: Specific types: CSCMatrix, CSRMatrix, and COOMatrix
    :animate: fade-in-slide-down
    :color: light

    .. doxygenstruct:: RandBLAS::sparse_data::COOMatrix
        :project: RandBLAS
        :members:

    .. doxygenstruct:: RandBLAS::sparse_data::CSRMatrix
        :project: RandBLAS
        :members:

    .. doxygenstruct:: RandBLAS::sparse_data::CSCMatrix
        :project: RandBLAS
        :members:


Operations on sparse matrices
==============================

Sketching
~~~~~~~~~

.. doxygenfunction:: RandBLAS::sketch_sparse(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m, T alpha, DenseSkOp<T,RNG> &S, int64_t S_ro, int64_t S_co, SpMat &A, int64_t A_ro, int64_t A_co, T beta, T *B, int64_t ldb) 
  :project: RandBLAS

.. doxygenfunction:: RandBLAS::sketch_sparse(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n, T alpha, SpMat &A, int64_t A_ro, int64_t A_co, DenseSkOp<T,RNG> &S, int64_t S_ro, int64_t S_co, T beta, T *B, int64_t ldb) 
  :project: RandBLAS


Deterministic operations
~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: RandBLAS::spmm(blas::Layout layout, blas::Op opA, blas::Op opB, int64_t m, int64_t n, int64_t k, T alpha, SpMat &A, int64_t ro_a, int64_t co_a, const T *B, int64_t ldb, T beta, T *C, int64_t ldc)  
  :project: RandBLAS

.. doxygenfunction:: RandBLAS::spmm(blas::Layout layout, blas::Op opA, blas::Op opB, int64_t m, int64_t n, int64_t k, T alpha, const T* A, int64_t lda, SpMat &B, int64_t ro_b, int64_t co_b, T beta, T *C, int64_t ldc) 
  :project: RandBLAS



