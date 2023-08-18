Sketching distributions and sketching operators
===============================================


Essentials of dense sketching
-----------------------------
.. doxygenstruct:: RandBLAS::DenseDist
   :project: RandBLAS
   :members:

.. doxygenstruct:: RandBLAS::DenseSkOp
   :project: RandBLAS
   :members: 

Essentials of sparse sketching 
------------------------------
.. doxygenstruct:: RandBLAS::SparseDist
   :project: RandBLAS
   :members:

.. doxygenstruct:: RandBLAS::SparseSkOp
   :project: RandBLAS
   :members: 

Advanced: sketching operator data structures
--------------------------------------------

.. doxygenfunction:: RandBLAS::fill_sparse(SparseSkOp<T, RNG> &S)
   :project: RandBLAS

.. doxygenfunction:: RandBLAS::fill_dense(DenseSkOp<T, RNG> &S)
   :project: RandBLAS

.. doxygenfunction:: RandBLAS::fill_dense(const DenseDist &D, T *buff, const RNGState<RNG> &seed)
   :project: RandBLAS

.. doxygenfunction:: RandBLAS::fill_dense(const DenseDist &D, int64_t n_rows, int64_t n_cols, int64_t i_off, int64_t j_off, T *buff, const RNGState<RNG> &seed)
   :project: RandBLAS

