***************************************************
Distributions and sketching operators
***************************************************

.. _rngstate_api:


Sketching distributions
=======================

.. doxygenconcept:: RandBLAS::SketchingDistribution
    :project: RandBLAS

.. doxygenfunction:: RandBLAS::isometry_scale_factor(SkDist D)
    :project: RandBLAS


Sketching operators and random states
=====================================

.. dropdown:: Sketching operators
    :animate: fade-in-slide-down
    :color: light
    
    .. doxygenconcept:: RandBLAS::SketchingOperator
        :project: RandBLAS


.. dropdown:: The state of a random number generator
    :animate: fade-in-slide-down
    :color: light

    .. doxygenstruct:: RandBLAS::RNGState
      :project: RandBLAS
      :members:



.. _densedist_and_denseskop_api:

DenseDist and DenseSkOp
============================================

.. doxygenenum:: RandBLAS::DenseDistName
    :project: RandBLAS

.. doxygenstruct:: RandBLAS::DenseDist
   :project: RandBLAS
   :members:

.. doxygenstruct:: RandBLAS::DenseSkOp
   :project: RandBLAS
   :members: 

.. doxygenfunction:: RandBLAS::fill_dense(DenseSkOp &S)
   :project: RandBLAS


.. _sparsedist_and_sparseskop_api:

SparseDist and SparseSkOp
==============================

.. doxygenstruct:: RandBLAS::SparseDist
   :project: RandBLAS
   :members:

.. doxygenstruct:: RandBLAS::SparseSkOp
   :project: RandBLAS
   :members: 

.. doxygenfunction:: RandBLAS::fill_sparse(SparseSkOp &S)
   :project: RandBLAS


Advanced material
=================

.. doxygenfunction:: RandBLAS::fill_dense(blas::Layout layout, const DenseDist &D, int64_t n_rows, int64_t n_cols, int64_t S_ro, int64_t S_co, T *buff, const RNGState<RNG> &seed)
   :project: RandBLAS

