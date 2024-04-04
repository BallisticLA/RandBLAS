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
Here's a visual aid to help decide whether to use a dense or a sparse sketching operator.

  .. raw:: html
      :file: ../assets/sparse_vs_dense_diagram_no_header.html

The chart's first question asks about the representation of :math:`A,`
and has a plain yes-no answer. 
It reflects a limitation of RandBLAS that sparse data can only be sketched with dense sketching operators.

The chart's second question is more subjective.
It's getting at whether adding :math:`O(dmn)` flops to a larger randomized algorithm
has potential to decisively impact that algorithm's performance.
Some randomized algorithms for dense matrix computations make it easy to answer this question.

  Subspace iteration methods for low-rank approximation have complexity :math:`\Omega(dmn)`
  regardless of whether the complexity of computing the initial sketch is :math:`o(dmn)`.

  Sketch-and-precondition methods for least squares need to set :math:`d \geq \min\{m,n\}`.
  As a result, these methods can't tolerate :math:`O(dmn)` operations for sketching while still providing
  asymptotically faster runtime than a direct least squares solver.

In other situations, it's not as clear cut. 


Distribution parameters
=======================

DenseDist
-----------------


SparseDist
------------------



The semantics of MajorAxis
==========================



Sketching operators in RandBLAS have a "MajorAxis" member.
The semantics of this member can be complicated.
We only expect advanced users to benefit from chosing this member
differently from the defaults we set.
We discuss the deeper meaning of and motivation for this member
later on this page.
