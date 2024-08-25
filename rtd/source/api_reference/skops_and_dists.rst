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

********************************************************************
Fundamentals
********************************************************************

  ..  very similar effects can be acheived in C, Fortran, or Julia. While there are 
  ..  certainly some popular programming languages that don't support this kind of API
  ..  (e.g., MATLAB, Python, and R), accessing RandBLAS from these languages should
  ..  be mediated operator-overloaded objects in a way that's analogous to how one 
  ..  would access BLAS.

RandBLAS has a polymorphic free-function API. We have spent a significant amount of 
effort on minimizing the number of RandBLAS-specific datastructures needed in order
to acheive the polymorphic API.

RandBLAS is very light on C++ idioms. The main C++ idioms we use are
templating and function overloading, plus some mild memory management
with destructor methods for structs. 

:ref:`Test of sketch updates <sketch_updates>`.


Abstractions
============

.. dropdown:: Distributions over random matrices
    :animate: fade-in-slide-down
    :color: light
    :open:

    .. doxygenconcept:: RandBLAS::SketchingDistribution
      :project: RandBLAS

.. dropdown:: Details on MajorAxis
  :animate: fade-in-slide-down 
  :color: light
    
  .. doxygenenum:: RandBLAS::MajorAxis
      :project: RandBLAS

.. dropdown:: Shared aspects of the sketching operator interface
    :animate: fade-in-slide-down
    :color: light
  
    .. doxygenconcept:: RandBLAS::SketchingOperator
      :project: RandBLAS


Distributions
=============

.. dropdown:: DenseDist : a distribution over matrices with i.i.d., mean-zero, variance-one entries
  :animate: fade-in-slide-down
  :color: light

  .. doxygenstruct:: RandBLAS::DenseDist
      :project: RandBLAS
      :members:

  .. doxygenenum:: RandBLAS::DenseDistName
      :project: RandBLAS

.. dropdown:: SparseDist : a distribution over structured sparse matrices
  :animate: fade-in-slide-down
  :color: light

  .. doxygenstruct:: RandBLAS::SparseDist
    :project: RandBLAS
    :members:


Random states and sketching operators
=====================================

.. dropdown:: RNGState 
    :animate: fade-in-slide-down
    :color: light

    .. doxygenstruct:: RandBLAS::RNGState
      :project: RandBLAS
      :members:

.. dropdown:: DenseSkOp : a sample from a DenseDist
  :animate: fade-in-slide-down
  :color: light

  .. doxygenstruct:: RandBLAS::DenseSkOp
    :project: RandBLAS
    :members: 

  .. doxygenfunction:: RandBLAS::fill_dense(DenseSkOp &S)
      :project: RandBLAS

  .. doxygenfunction:: RandBLAS::fill_dense(blas::Layout layout, const DenseDist &D, int64_t n_rows, int64_t n_cols, int64_t S_ro, int64_t S_co, T *buff, const RNGState<RNG> &seed)
      :project: RandBLAS

.. dropdown:: SparseSkOp : a sample from a SparseDist
  :animate: fade-in-slide-down
  :color: light

  .. doxygenstruct:: RandBLAS::SparseSkOp
    :project: RandBLAS
    :members: 

  .. doxygenfunction:: RandBLAS::fill_sparse(SparseSkOp &S)
    :project: RandBLAS
