.. :sd_hide_title:

.. toctree::
  :maxdepth: 3

****************************************************************
Defining a sketching distribution
****************************************************************

RandBLAS' sketching operators can be divided into two categories.

  *Dense* sketching operators have entries that are sampled iid from
  a mean-zero distribution over the reals.
  Distributions over these operators are represented with the
  :ref:`DenseDist <densedist_and_denseskop_api>` class.

  *Sparse* sketching operators have random (but highly structured) sparsity patterns.
  Their nonzero entries are sampled iid and uniformly from :math:`\{-1,1\}.`
  Distributions over these operators are represented with the 
  :ref:`SparseDist <sparsedist_and_sparseskop_api>` class.

The first order of business in correctly using RandBLAS is to decide which type of sketching
operator is appropriate in your situation. 
From there, you need to instantiate a specific distribution by setting some parameters.
This part of the tutorial gives tips on both of these points.


How to choose between dense and sparse sketching
=====================================================================

Let's say you have an :math:`m \times n` matrix :math:`A` and an integer :math:`d,`
and you want to compute a sketch of :math:`A` that has rank :math:`\min\{d, \operatorname{rank}(A)\}.`
Here's a chart to help decide whether to use a dense or a sparse sketching operator.

  .. raw:: html
      :file: ../assets/sparse_vs_dense_diagram_no_header.html

Discussion of the chart's first yes/no branch.
  RandBLAS doesn't allow applying sparse sketching operators to sparse data.
  This is because RandBLAS is only intended to produce sketches that are dense.

  While 
  algorithms exist to multiply two sparse matrices and store the
  result as a dense matrix, we don't know of practical RandNLA
  algorithms that would benefit from this functionality.


Discussion of the chart's second yes/no branch.
  This gets at whether adding :math:`O(dmn)` flops to a randomized algorithm
  can decisively impact that algorithm's performance.
  Some randomized algorithms for dense matrix computations make it easy to answer this question.
  Consider, for example ...

    *Subspace iteration methods for low-rank approximation.* These methods have complexity :math:`\Omega(dmn)`
    regardless of whether the complexity of computing the initial sketch is :math:`o(dmn)`.

    *Sketch-and-precondition methods for least squares.* These methods need to set :math:`d \geq \min\{m,n\}`.
    As a result, they can't tolerate :math:`O(dmn)` operations for sketching while still providing
    asymptotically faster runtime than a direct least squares solver.

  With this in mind, notice that the chart indicates a preference dense sketching over
  sparse sketching when dense sketching can be afforded.
  This preference stems from how if the sketching dimension is fixed, then the statistical properties of dense sketching
  operators will generally be preferable to those of sparse
  sketching operators.

  .. note::
    See Wikipedia for the meanings of
    `big-omega notation <https://en.wikipedia.org/wiki/Big_O_notation#Big_Omega_notation>`_ and 
    `little-o notation <https://en.wikipedia.org/wiki/Big_O_notation#Little-o_notation>`_.


Distribution parameters
=======================

This part of the web docs is coming soon!



The semantics of Axis
==========================

Sketching operators in RandBLAS have a "Axis" member.
The semantics of this member can be complicated.
We only expect advanced users to benefit from chosing this member
differently from the defaults we set.

A proper explanation of Axis' semantics is coming soon!
Bear with us until then.
