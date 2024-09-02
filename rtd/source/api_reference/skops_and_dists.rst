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
   .. |vecnnz| mathmacro:: \texttt{vec_nnz}
   .. |ttt| mathmacro:: \texttt

********************************************************************
Fundamentals
********************************************************************

  ..  very similar effects can be achieved in C, Fortran, or Julia. While there are 
  ..  certainly some popular programming languages that don't support this kind of API
  ..  (e.g., MATLAB, Python, and R), accessing RandBLAS from these languages should
  ..  be mediated operator-overloaded objects in a way that's analogous to how one 
  ..  would access BLAS.

RandBLAS has a polymorphic free-function API. We have spent a significant amount of 
effort on minimizing the number of RandBLAS-specific datastructures needed in order
to achieve that polymorphism.

RandBLAS is very light on C++ idioms. The main idioms we use are
templating and function overloading, plus some mild memory management
with destructor methods for structs. The only place we even use inheritance is
in our test code!

We have a bunch of functions that aren't documented on this website.
If such a function looks useful, you should feel free to use it. If you
end up doing that and you care about your code's compatibility with future
versions of RandBLAS, then please let us know by filing a quick GitHub issue.


Preliminaries
=============

.. dropdown:: The Axis enum
  :animate: fade-in-slide-down 
  :color: light
    
  .. doxygenenum:: RandBLAS::Axis
      :project: RandBLAS

.. dropdown:: RNGState 
    :animate: fade-in-slide-down
    :color: light

    .. doxygenstruct:: RandBLAS::RNGState
        :project: RandBLAS
        :members:

Gaussians et al.
================

.. dropdown:: DenseDist : a distribution over matrices with i.i.d., mean-zero, variance-one entries
  :animate: fade-in-slide-down
  :color: light

  .. doxygenenum:: RandBLAS::ScalarDist
      :project: RandBLAS

  .. doxygenstruct:: RandBLAS::DenseDist
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

  .. doxygenfunction:: RandBLAS::fill_dense_unpacked(blas::Layout layout, const DenseDist &D, int64_t n_rows, int64_t n_cols, int64_t S_ro, int64_t S_co, T *buff, const RNGState<RNG> &seed)
      :project: RandBLAS


CountSketch et al.
==================

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

  .. doxygenfunction:: RandBLAS::fill_sparse_unpacked_nosub(const SparseDist &D, int64_t &nnz, T* vals, sint_t* rows, sint_t* cols, const state_t &seed_state)
      :project: RandBLAS



The unifying (C++20) concepts
=============================

.. dropdown:: SketchingDistribution
    :animate: fade-in-slide-down
    :color: light

    .. doxygenconcept:: RandBLAS::SketchingDistribution
        :project: RandBLAS


.. dropdown:: SketchingOperator
    :animate: fade-in-slide-down
    :color: light
  
    .. doxygenconcept:: RandBLAS::SketchingOperator
        :project: RandBLAS

