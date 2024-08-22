***************************************************
Distributions and sketching operators
***************************************************

.. _rngstate_api:


Core concepts and design patterns
=================================

.. dropdown:: Distributions over random matrices
    :animate: fade-in-slide-down
    :color: light

    .. doxygenconcept:: RandBLAS::SketchingDistribution
        :project: RandBLAS

    .. doxygenfunction:: RandBLAS::isometry_scale_factor(SkDist D)
        :project: RandBLAS

.. dropdown:: States of random number generators
    :animate: fade-in-slide-down
    :color: light

    .. doxygenstruct:: RandBLAS::RNGState
      :project: RandBLAS
      :members:

.. dropdown:: Samples from distributions over random matrices
    :animate: fade-in-slide-down
    :color: light
    
    .. doxygenconcept:: RandBLAS::SketchingOperator
        :project: RandBLAS




.. _densedist_and_denseskop_api:

Dense sketching: Gaussians and the like
=======================================

.. dropdown:: DenseDist structs
   :animate: fade-in-slide-down
   :color: light

    .. doxygenstruct:: RandBLAS::DenseDist
        :project: RandBLAS
        :members:

    .. doxygenenum:: RandBLAS::DenseDistName
       :project: RandBLAS


.. dropdown:: DenseSkOp structs
   :animate: fade-in-slide-down
   :color: light

    .. doxygenstruct:: RandBLAS::DenseSkOp
      :project: RandBLAS
      :members: 

    *Memory management*

    .. doxygenfunction:: RandBLAS::fill_dense(DenseSkOp &S)
        :project: RandBLAS

    .. doxygenfunction:: RandBLAS::fill_dense(blas::Layout layout, const DenseDist &D, int64_t n_rows, int64_t n_cols, int64_t S_ro, int64_t S_co, T *buff, const RNGState<RNG> &seed)
        :project: RandBLAS

.. _sparsedist_and_sparseskop_api:

Sparse sketching: CountSketch et al.
====================================

.. dropdown:: SparseDist structs
   :animate: fade-in-slide-down
   :color: light

    .. doxygenstruct:: RandBLAS::SparseDist
      :project: RandBLAS
      :members:

.. dropdown:: SparseSkOp structs
   :animate: fade-in-slide-down
   :color: light

    .. doxygenstruct:: RandBLAS::SparseSkOp
      :project: RandBLAS
      :members: 

    .. doxygenfunction:: RandBLAS::fill_sparse(SparseSkOp &S)
      :project: RandBLAS


