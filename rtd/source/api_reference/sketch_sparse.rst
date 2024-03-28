   .. |op| mathmacro:: \operatorname{op}
   .. |mat| mathmacro:: \operatorname{mat}
   .. |submat| mathmacro:: \operatorname{submat}
   .. |ldb| mathmacro:: \texttt{ldb}
   .. |opA| mathmacro:: \texttt{opA}
   .. |opS| mathmacro:: \texttt{opS}

********************************
Computing a sketch : sparse data
********************************


Representing sparse matrices
============================


Matrix-matrix operations
========================


.. doxygenfunction:: RandBLAS::sketch_sparse(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m, T alpha, DenseSkOp<T,RNG> &S, int64_t S_ro, int64_t S_co, SpMatrix &A, int64_t A_ro, int64_t A_co, T beta, T *B, int64_t ldb) 
  :project: RandBLAS


.. doxygenfunction:: RandBLAS::sketch_sparse(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n, T alpha, SpMatrix &A, int64_t A_ro, int64_t A_co, DenseSkOp<T,RNG> &S, int64_t S_ro, int64_t S_co, T beta, T *B, int64_t ldb) 
  :project: RandBLAS

