   .. |op| mathmacro:: \operatorname{op}
   .. |mat| mathmacro:: \operatorname{mat}
   .. |submat| mathmacro:: \operatorname{submat}
   .. |lda| mathmacro:: \texttt{lda}
   .. |ldb| mathmacro:: \texttt{ldb}
   .. |ldc| mathmacro:: \texttt{ldc}
   .. |opA| mathmacro:: \texttt{opA}
   .. |opB| mathmacro:: \texttt{opB}

############################################################
Deterministic operations with sparse matrices
############################################################


.. doxygenfunction:: RandBLAS::spmm(blas::Layout layout, blas::Op opA, blas::Op opB, int64_t m, int64_t n, int64_t k, T alpha, SpMat &A, int64_t ro_a, int64_t co_a, const T *B, int64_t ldb, T beta, T *C, int64_t ldc)  
  :project: RandBLAS


.. doxygenfunction:: RandBLAS::spmm(blas::Layout layout, blas::Op opA, blas::Op opB, int64_t m, int64_t n, int64_t k, T alpha, const T* A, int64_t lda, SpMat &B, int64_t ro_b, int64_t co_b, T beta, T *C, int64_t ldc) 
  :project: RandBLAS

