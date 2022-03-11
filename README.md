# RandBLAS

RandBLAS is a **pre-alpha release** C++ library for sketching in randomized linear algebra.
Our goal is for RandBLAS to become a standard library that supports randomized linear algebra algorithms
in other libraries.
In fact, we want RandBLAS to eventually become a *standard* like the BLAS, in that hardware vendors might release their own optimized implementations of algorithms which confirm to the RandBLAS API.

This library will surely undergo several major revisions before its API stabilizes.
Therefore we insist that no one use RandBLAS as a dependency in a larger project for now.
While RandBLAS is dependency for a library called ["RandLAPACK"](https://github.com/BallisticLA/proto_randlapack) which we are also developing, we are preparred to make major changes to RandLAPACK whenever a major change is made to RandBLAS.

Refer to ``INSTALL.md`` for directions on how to install RandBLAS' dependencies, install
RandBLAS itself, and use RandBLAS in other projects.


## Notes on scratch code workspaces and basic benchmarking 

We'll need to conduct experiments for proofs-of-concept and benchmarking while developing RandBLAS.
Those experiments should be kept under version control.
If you want to make such an experiment, create a branch like
```
git checkout -b experiments/riley-sjltidea-220311
```
The branch name should always have the prefix "experiments/[your name]".
The example name above includes keywords on the nature of the branch and date in YYMMDD format.
If you want to share that example with others then you can push the branch to the ``BallisticLA/RandBLAS``
repository.

Using branches is important because the RandBLAS API is nowhere near settled;
an example or benchmark written to work for the state of ``main`` today might break
badly on the state of ``main`` tomorrow.
If you get to the point of a clean example which you would like to refer to in the past,
we recommend that you use [git tags](https://en.wikibooks.org/wiki/Git/Advanced#Tags) for important
commits on your experiment branch.

