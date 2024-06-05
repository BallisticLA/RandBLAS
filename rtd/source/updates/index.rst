
Changes to RandBLAS
===================

This page details changes made to RandBLAS over time, in reverse chronological order.
We have a tentative policy of providing bugfix support for any release of 
RandBLAS upon request, no matter how old. With any luck, RandBLAS will grow enough
in the future that we will change this policy to support a handful of versions
at a time.

RandBLAS follows `Semantic Versioning <semver.org>`_.


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
