***********************************************
Sketching distributions and sketching operators
***********************************************

The most important thing about sketching operators in RandBLAS
is simply their mathematical meaning. Once you understand that
meaning it's extremely easy to construct a sketching
operator and use it to compute a sketch.
Therefore this page starts with the mathematical meanings of the
dense and sparse sketching operators in RandBLAS.
Following that, we cover some advanced topics like the data structures that 
underlie these operators.


.. note::
    Sketching operators in RandBLAS have a "MajorAxis" member.
    The semantics of this member can be complicated.
    We only expect advanced users to benefit from chosing this member
    differently from the defaults we set.
    We discuss the deeper meaning of and motivation for this member
    later on this page.



Core API for dense sketching
============================

.. doxygenenum:: RandBLAS::DenseDistName
   :project: RandBLAS

.. doxygenstruct:: RandBLAS::DenseDist
   :project: RandBLAS
   :members:

.. doxygenstruct:: RandBLAS::DenseSkOp
   :project: RandBLAS
   :members: 

Core API for sparse sketching 
=============================
.. doxygenstruct:: RandBLAS::SparseDist
   :project: RandBLAS
   :members:

.. doxygenstruct:: RandBLAS::SparseSkOp
   :project: RandBLAS
   :members: 


Advanced material
=================


What's this about a "major axis"?
---------------------------------

.. doxygenenum:: RandBLAS::MajorAxis
   :project: RandBLAS


Data structures and memory management
-------------------------------------

We don't prominently expose the data structures used for
the sketching operators.
There are two reasons for this. 
First, dense and sparse sketching operators need very different
data structures, and yet it would be disruptive to users to
constantly be cognizant of these differences.
Second, the particular data structures are ultimately implementation
details, and they may change in the future.
Still, we think it's good practice to provide some information here.
This information might be useful to you if you wanted to apply 
a RandBLAS sketching operator to a data matrix using a custom function
(i.e., a function other than RandBLAS' ``sketch_general``).

Dense sketching operators
^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: RandBLAS::fill_dense(DenseSkOp<T, RNG> &S)
   :project: RandBLAS

.. doxygenfunction:: RandBLAS::fill_dense(const DenseDist &D, T *buff, const RNGState<RNG> &seed)
   :project: RandBLAS

.. doxygenfunction:: RandBLAS::fill_dense(const DenseDist &D, int64_t n_rows, int64_t n_cols, int64_t S_ro, int64_t S_co, T *buff, const RNGState<RNG> &seed)
   :project: RandBLAS

Sparse sketching operators
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: RandBLAS::fill_sparse(SparseSkOp<T, RNG, sint_t> &S)
   :project: RandBLAS
