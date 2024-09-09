   .. |op| mathmacro:: \operatorname{op}
   .. |mat| mathmacro:: \operatorname{mat}
   .. |submat| mathmacro:: \operatorname{submat}
   .. |D| mathmacro:: \mathcal{D}
   .. |mtxA| mathmacro:: \mathbf{A}
   .. |mtxB| mathmacro:: \mathbf{B}
   .. |mtxS| mathmacro:: \mathbf{S}
   .. |mtxx| mathmacro:: \mathbf{x}
   .. |ttt| mathmacro:: \texttt
   .. |vecnnz| mathmacro:: \texttt{vec_nnz}
   .. |majoraxis| mathmacro:: \texttt{major_axis}
   .. |nrows| mathmacro:: \texttt{n_rows}
   .. |ncols| mathmacro:: \texttt{n_cols}
   .. |vals| mathmacro:: \texttt{vals}

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

Let's say you have an :math:`m \times n` matrix :math:`\mtxA` and an integer :math:`d,`
and you want to compute a sketch of :math:`\mtxA` that has rank :math:`\min\{d, \operatorname{rank}(\mtxA)\}.`
Here's a two-question heuristic for choosing the best sketching operator RandBLAS can offer.

  Q1. Is :math:`\mtxA` sparse?
    If so, use a dense operator.

  Q2. Supposing that :math:`\mtxA` is dense -- can you afford :math:`\Theta(dmn)` flops to compute the sketch?
    If so, use a dense operator. If not, use a sparse operator.


Discussion of Q1.
  RandBLAS doesn't allow applying sparse sketching operators to sparse data.
  This is because RandBLAS is only intended to produce sketches that are dense.

  While 
  algorithms exist to multiply two sparse matrices and store the
  result as a dense matrix, we don't know of practical RandNLA
  algorithms that would benefit from this functionality.


Discussion of Q2.
  This gets at whether adding :math:`O(dmn)` flops to a randomized algorithm
  can decisively impact that algorithm's performance.
  Some randomized algorithms for dense matrix computations make it easy to answer this question.
  Consider, for example ...

    *Subspace iteration methods for low-rank approximation.* These methods have complexity :math:`\Omega(dmn)`
    regardless of whether the complexity of computing the initial sketch is :math:`o(dmn).`

    *Sketch-and-precondition methods for least squares.* These methods need to set :math:`d \geq \min\{m,n\}.`
    As a result, they can't tolerate :math:`O(dmn)` operations for sketching while still providing
    asymptotically faster runtime than a direct least squares solver.

With this in mind, note how our hueristic indicates a preference dense sketching over
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

Here are example constructor invocations for DenseDist and SparseDist if we include all optional arguments.

.. code:: c++

    int64_t d = 500; int64_t m = 10000; // We don't require that d < m. This is just for concreteness.

    ScalarDist family = ScalarDist::Gaussian;
    DenseDist  D_dense( d, m, family,  Axis::Long);

    int64_t vec_nnz = 4;
    SparseDist D_sparse(d, m, vec_nnz, Axis::Short);

The first two arguments have *identical* meanings for DenseDist and
SparseDist; they give the number of rows and columns in the operator.
The meanings of the remaining arguments differ
*significantly* depending on which case you're in.


DenseDist: family and major_axis
--------------------------------
A DenseDist represents a distribution over matrices with fixed dimensions, where 
the entries are i.i.d. mean-zero variance-one random variables. 
Its trailing constructor arguments are called  :math:`\ttt{family}` and :math:`\majoraxis.`

The family argument indicates whether the entries follow the standard normal distribution
(:math:`\ttt{ScalarDist::Gaussian}`) or the uniform distribution over :math:`[-\sqrt{3},\sqrt{3}]`
(:math:`\ttt{ScalarDist::Uniform}`).
These "Gaussian operators" and "uniform operators" (as we'll call them) are *very similar* in theory
and *extremely similar* in practice.
The DenseDist constructor uses Gaussian by default since this is only marginally more expensive
than uniform in most cases, and Gaussian operators are far more common in theoretical analysis
of randomized algorithms.

Then there's :math:`\majoraxis.` From a statistical perspective there is absolutely
no difference between :math:`\ttt{major_axis = Short}` or :math:`\ttt{major_axis = Long}.`
Although, there are 
narrow circumstances where one of these might be preferred in practice. We'll explain with an example.
    
.. code:: c++

    // Assume previous code defined integers (d1, d2, n) where 0 < d1 < d2 < n,
    // and "family" variable equal to ScalarDist::Gaussian or ScalarDist::Uniform,
    // and a "state" variable of type RNGState.
    DenseDist D1(d1, n, family, Axis::Long);
    DenseDist D2(d2, n, family, Axis::Long);
    DenseSkOp S1(D1, state);
    DenseSkOp S2(D2, state);
    // If S1 and S2 are represented explicitly as dense matrices, then S1 is the 
    // n-by-d1 submatrix of S2 obtained by selecting its first d1 columns.

In this example, long-axis-major DenseDists provide for a reproducible stream of random column vectors
for tall sketching operators. If the row and column dimensions were swapped, then we'd have a mechanism
for reproducibly sampling from streams of random row vectors for wide sketching operators.
See :ref:`this page <sketch_updates>` of the tutorial for more information on the role of 
:math:`\majoraxis` for dense sketching operators.

.. _sparsedist_params:

SparseDist: vec_nnz and major_axis
----------------------------------

A SparseDist represents a distribution over sparse matrices with fixed dimensions, where 
either the rows or the columns are sampled independently from certains distribution over
sparse vectors.
The distribution is determined by the trailing constructor arguments:  :math:`\vecnnz` and :math:`\majoraxis.`

Let :math:`k = \ttt{dim_major}.`
If :math:`\majoraxis = \ttt{Short},` this is :math:`\min\{\nrows,\ncols\}.`
If :math:`\majoraxis = \ttt{Long},` this is :math:`\max\{\nrows,\ncols\}.`
The major-axis vectors of a SparseSkOp follow a distribution :math:`\mathcal{V}` over :math:`\mathbb{R}^k.`
The number of nonzeros in each major-axis vector is bounded by  :math:`1 \leq \vecnnz \leq k.`

All else equal, larger values of :math:`\vecnnz` result in distributions
that are "better" at preserving Euclidean geometry when sketching.
The value of :math:`\vecnnz` that suffices for a given context will 
also depend on the sketch size, :math:`d := \min\{\nrows,\ncols\}.`
Larger sketch sizes make it possible to "get away with" smaller values of
:math:`\vecnnz.`

SparseDist when :math:`\majoraxis = \ttt{Short}.`

  A sample from :math:`\mathcal{V}` has exactly :math:`\vecnnz` nonzeros.
  The locations of those nonzeros are chosen uniformly
  without replacement from :math:`\{0,\ldots,k-1\}.` The values of the nonzeros are
  sampled independently and uniformly from :math:`\pm 1.`

  Many sketching distributions from the literature fall into this category.
  :math:`\vecnnz = 1` corresponds to the distribution over CountSketch operators.
  :math:`\vecnnz > 1` corresponds to distributions which have been studied under
  many names, including OSNAPs, SJLTs, and hashing embeddings.

  The community has come to a consensus that very small values of :math:`\vecnnz` can suffice for good performance.
  For example, suppose we seek a constant-distortion embedding
  of an unknown subspace of dimension :math:`n,` where :math:`1{,}000 \leq n \leq 10{,}000.`
  If :math:`d = 2n`, then many practitioners
  would restrict their attention to :math:`\vecnnz \leq 8.`
  There are no special performance benefits in RandBLAS to setting :math:`\vecnnz = 1.`
  Additionally, using :math:`\vecnnz > 1` makes it far more likely for a sketch to retain
  useful geometric information from the data.
  Therefore, we recommend using :math:`\vecnnz \geq 2` in practice.

SparseDist when :math:`\majoraxis = \ttt{Long}.` 

  A sample :math:`\mtxx` from :math:`\mathcal{V}` has *at most* :math:`\vecnnz` nonzero
  entries. The locations of the nonzeros are determined by sampling uniformly
  with replacement from :math:`\{0,\ldots,k-1\}.`
  If index :math:`j` occurs in the sample :math:`\ell` times, then 
  :math:`\mtxx_j` will equal :math:`\sqrt{\ell}` with probability 1/2 and
  :math:`-\sqrt{\ell}` with probability 1/2.

  In the literature,
  :math:`\vecnnz = 1` corresponds to operators for sampling uniformly with replacement
  from the rows or columns of a data matrix (although the signs on the rows or
  columns may be flipped). :math:`\vecnnz > 1` is a special case of LESS-uniform
  distributions, where the underlying scalar sub-gaussian distribution is the uniform
  distribution over :math:`\pm 1.`

  It is important to use (much) larger values of :math:`\vecnnz` here compared to the 
  short-axis-major case.
  There is less consensus in the community for what constitutes "big enough in practice," 
  therefore we make no prescriptions on this front.
