
Changes to RandBLAS
===================

This page details changes made to RandBLAS over time, in reverse chronological order.
We have a tentative policy of providing bugfix support for any release of 
RandBLAS upon request, no matter how old. With any luck, this project will grow enough
that we'll have to change this policy.

RandBLAS follows `Semantic Versioning <https://semver.org>`_.


RandBLAS 1.0
------------
*Release date: September 12, 2024. Release manager: Riley Murray.*

Today marks RandBLAS' second-ever release, its first *stable* release,
and its first release featuring the contributions of someone who showed
up entirely out of the blue (shoutout to Rylie Weaver)!

Overview of changes
~~~~~~~~~~~~~~~~~~~

**New features for core functionality**

The semantics of :cpp:any:`RandBLAS::SparseDist::major_axis` have changed in RandBLAS 1.0.
As a result of this change, SparseSkOps can represent 
LESS-Uniform operators and operators for plain row or column sampling with replacement.
(This is in addition to hashing-style operators like CountSketch, which we've supported since version 0.2.)

We have four new functions for sampling from index sets.

  * :cpp:any:`RandBLAS::weights_to_cdf`
  * :cpp:any:`RandBLAS::sample_indices_iid`
  * :cpp:any:`RandBLAS::sample_indices_iid_uniform`
  * :cpp:any:`RandBLAS::repeated_fisher_yates`

We have two new functions for getting low-level data for a sketching operator's explicit representation:
:cpp:any:`RandBLAS::fill_dense_unpacked` and :cpp:any:`RandBLAS::fill_sparse_unpacked_nosub`. 
These are useful if you want to incorporate RandBLAS' sketching functionality into other frameworks,
like Kokkos, cuBLAS, or MKL.

Finally, there's :cpp:any:`RandBLAS::sketch_symmetric`, overloaded for sketching from the left or right.

**Quality-of-life improvements**

 * We've significantly expanded the tutorial part of our web docs. It now has details on updating sketches and 
   some advice on choosing parameters for sketching distributions.
 * :cpp:any:`RandBLAS::Error` is now in the public API.
 * :cpp:any:`RandBLAS::print_buff_to_stream` is for writing MATLAB-style or NumPy-style string representations of matrices to a provided stream, like std::cout.
 * We settled on a unified memory-management / memory-ownership policy. There's no difference between DenseSkOp, SparseSkOp, or any of the sparse matrix types. The abstract policy is described in our web documentation. The consequences of the policy for each of the aforementioned types is documented in source code and on our website. 
 * We added a few utility functions for working with dense matrices: symmetrize, overwrite_triangle, and transpose_square.

**Significantly revised APIs for sketching distributions and operators**

 * Added new :cpp:any:`RandBLAS::SketchingDistribution` and :cpp:any:`RandBLAS::SketchingOperator` C++20 concepts.
 * API revisions to DenseDist/DenseSkOp and SparseDist/SparseSkOp were mostly about taking quantities which we would compute from an object's const members with free-functions,
   and instead making those quantities const members themselves. Good examples of this are :cpp:any:`RandBLAS::DenseDist::isometry_scale`
   and :cpp:any:`RandBLAS::SparseDist::isometry_scale`, whose meanings are explained in the SketchingDistribution docs.
 * :cpp:any:`RandBLAS::DenseSkOp::next_state` and :cpp:any:`RandBLAS::SparseSkOp::next_state` are computed at construction time,
   without actually performing any random sampling. This means that one can define a sequence of independent sketching without
   changing an RNGState's "key" and without realizing any of them explicitly.	

**New statistical tests**

 * Kolmogorov–Smirnov tests for distributional correctness of sample_indices_iid, sample_indices_iid_uniform, repeated_fisher_yates, and the scalar distributions that can be used with DenseSkOp (standard-normal and uniform over [-1,1]).
 * Tests for subspace embedding properties of DenseSkOp. A forthcoming paper will describe how these tests cover a wide range of relevant parameter values at very mild computational cost.
 * We've incorporated select tests from Random123 into our testing framework.


Contributors
~~~~~~~~~~~~

I'd like to start by acknowledging the contributions of `Parth Nobel <https://ptnobel.github.io/>`_ to RandBLAS' development.
Parth and I have worked on-and-off on several projects involving RandNLA algorithms.
None of these projects has been published yet, but they've had a significant role in uncovering
bugs and setting development priorities for RandBLAS. (As a recent example in the latter category,
I probably wouldn't have added the "sample_indices_iid" function were it not for its relevance to
one of our projects.) This led me to be quite surprised when I noticed that Parth technically hasn't
made a commit to the RandBLAS repository! Let this statement set the record straight: Parth has
made very real contributions to RandBLAS, even if the commit history doesn't currently reflect that.

Rylie Weaver, the aforementioned out-of-the-blue contributor, helped write our Kolmogorov–Smirnov tests for repeated Fisher–Yates. 

I wrote a lot of code (as one might imagine).

Funding acknowledgements
~~~~~~~~~~~~~~~~~~~~~~~~

This work was wholly supported by LDRD funding from Sandia National Laboratories.

Sandia National Laboratories is a multi-mission laboratory managed and operated by
National Technology and Engineering Solutions of Sandia, LLC., a wholly owned subsidiary
of Honeywell International, Inc., for the U.S. Department of Energy’s National Nuclear
Security Administration under contract DE-NA-0003525.



RandBLAS 0.2
------------

*Released June 5, 2024.*

Today marks the first formal release of RandBLAS. We've been working on it for over three years, so 
we couldn't possibly describe all of its capabilities in just this changelog. Instead, we'll repurpose some
text that's featured prominently in our documentation at the time of this release.

A quote from the README, describing the aims of this project:

    RandBLAS supports high-level randomized linear algebra algorithms (like randomized low-rank SVD) that might be implemented in other libraries.
    Our goal is for RandBLAS to become a standard like the BLAS, in that hardware vendors might release their own optimized implementations of algorithms which confirm to the RandBLAS API.

A quote from the website, describing our current capabilities:

    RandBLAS is efficient, flexible, and reliable.
    It uses CPU-based OpenMP acceleration to apply its sketching operators to dense or sparse data matrices stored in main memory.
    All sketches produced by RandBLAS are dense.
    As such, dense data matrices can be sketched with dense or sparse operators, while sparse data matrices can only be sketched with dense operators.
    RandBLAS can be used in distributed environments through its ability to (reproducibly) compute products with *submatrices* of sketching operators.

There's a *ton* of documentation besides those snippets. In fact, we have three separate categories of documentation!

 1. Traditional source code comments.
 2. Web documentation (i.e., this entire website)
 3. Developer notes; `one <https://github.com/BallisticLA/RandBLAS/blob/a66751ced6a0b44667e21bc4cb6fe59b5785c7fb/RandBLAS/DevNotes.md>`_ for RandBLAS as a whole,
    `another <https://github.com/BallisticLA/RandBLAS/blob/a66751ced6a0b44667e21bc4cb6fe59b5785c7fb/RandBLAS/sparse_data/DevNotes.md>`_ for our sparse matrix functionality,
    and `a third <https://github.com/BallisticLA/RandBLAS/blob/a66751ced6a0b44667e21bc4cb6fe59b5785c7fb/rtd/DevNotes.md>`_ for this website.

Contributors and Acknowledgements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Since this is our first release, many acknowledgements in order.
We'll start with contributors to the RandBLAS codebase as indicated by the 
repository commit history.

    Riley Murray, Burlen Loring, Kaiwen He, Maksim Melnichenko, Tianyu Liang, and Vivek Bharadwaj.

In addition to code contributors, we had the benefit of supervision and input
from the following established principal investigators

    James Demmel, Michael Mahoney, Jack Dongarra, Piotr Luszczek, Mark Gates, and Julien Langou.

We would also like to thank Weslley da Silva Pereira, who gave valuable feedback at
the earliest stages of this project, and all of the individuals who gave feedback on 
our `RandNLA monograph <https://arxiv.org/abs/2302.11474>`_. 

The work that lead to this release of RandBLAS was funded by the
U.S. National Science Foundation and the U.S. Department of Energy, and was
conducted at the International Computer Science Institute,
the University of California at Berkeley, the University of Tennessee at Knoxville, 
Lawrence Berkeley National Laboratory, and Sandia National Laboratories. 

What happened to RandBLAS 0.1?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We tagged a commit on the RandBLAS repository with version 0.1.0 almost two years ago.
However, we hadn't maintained version numbers or a dedicated changelog since then. RandBLAS 0.2.0 is
our *first* formal release. We opted not to release under version 0.1.0 since that could
ambiguously refer to anything from the now-very-old 0.1.0 tag up to the present.
