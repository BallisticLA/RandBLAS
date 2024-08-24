   .. |op| mathmacro:: \operatorname{op}
   .. |mat| mathmacro:: \operatorname{mat}
   .. |submat| mathmacro:: \operatorname{submat}
   .. |D| mathmacro:: \mathcal{D}
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
   .. |mtxx| mathmacro:: \mathbf{x}
   .. |mtxy| mathmacro:: \mathbf{y}
   .. |ttt| mathmacro:: \texttt

******************************************************
Fundamentals of sketching and random number generation
******************************************************

.. _rngstate_api:


Sketching in RandBLAS concerns linear maps (sketching operators) that take vectors in high-dimensional
coordinate representations to vectors in low-dimensional coordinate representations.

We assume that the initial vectors are given as the rows or columns of some matrix.
This lets us describe sketching operators as matrices that are wide (having more columns than rows)
or tall (having more rows than columns).
Square sketching operators are not forbidden, but they are certainly not our focus.

.. dropdown:: Distributions over random matrices
    :animate: fade-in-slide-down
    :color: light

    .. doxygenconcept:: RandBLAS::SketchingDistribution
      :project: RandBLAS

    .. dropdown:: Scaling to obtain partial isometries
      :animate: fade-in-slide-down
      :color: light
      
      .. doxygenfunction:: RandBLAS::isometry_scale_factor(SkDist D)
        :project: RandBLAS

    .. dropdown:: Details on MajorAxis
      :animate: fade-in-slide-down 
      :color: light
        
      .. doxygenenum:: RandBLAS::MajorAxis
          :project: RandBLAS

.. dropdown:: Sketching operators
    :animate: fade-in-slide-down
    :color: light
  
    .. doxygenconcept:: RandBLAS::SketchingOperator
      :project: RandBLAS

.. dropdown:: States of random number generators
    :animate: fade-in-slide-down
    :color: light

    .. doxygenstruct:: RandBLAS::RNGState
      :project: RandBLAS
      :members:


.. _densedist_and_denseskop_api:

Dense sketching: Gaussians et al.
=================================

.. dropdown:: DenseDist : a distribution over matrices with i.i.d., mean-zero, variance-one entries
  :animate: fade-in-slide-down
  :color: light

  .. doxygenstruct:: RandBLAS::DenseDist
      :project: RandBLAS
      :members:

  .. doxygenenum:: RandBLAS::DenseDistName
      :project: RandBLAS

.. dropdown:: DenseSkOp : a sample from a DenseDist
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

.. dropdown:: SparseDist : a distribution over structured sparse matrices
  :animate: fade-in-slide-down
  :color: light

  .. doxygenstruct:: RandBLAS::SparseDist
    :project: RandBLAS
    :members:

.. dropdown:: SparseSkOp : a sample from a SparseDist
  :animate: fade-in-slide-down
  :color: light

  .. doxygenstruct:: RandBLAS::SparseSkOp
    :project: RandBLAS
    :members: 

  .. doxygenfunction:: RandBLAS::fill_sparse(SparseSkOp &S)
    :project: RandBLAS


