### RandBLAS : A header-only C++ library for sketching in randomized linear algebra

RandBLAS supports high-level randomized linear algebra algorithms (like randomized low-rank SVD) that might be implemented in other libraries.

Our goal is for RandBLAS to become a standard like the BLAS, in that hardware vendors might
release their own optimized implementations of algorithms which confirm to the RandBLAS API.

For those who are new to randomized linear algebra, we recommend you check out [this 35-minute YouTube video](https://www.youtube.com/watch?v=6htbyY3rH1w) on the subject.

### Documentation

We have three types of documentation.
 1. Traditional source code comments.
 2. Web documentation, split into a [tutorial](https://randblas.readthedocs.io/en/latest/tutorial/index.html) and an [API reference](https://randblas.readthedocs.io/en/latest/api_reference/index.html).
 3. Developer notes; [one](RandBLAS/DevNotes.md) for RandBLAS as a whole and [another](RandBLAS/sparse_data/DevNotes.md) for our sparse matrix functionality.

Detailed installation instructions are in [INSTALL.md](INSTALL.md).

### Continuous integration builds

![Latest Ubuntu (OpenMP)](https://github.com/BallisticLA/RandBLAS/actions/workflows/core-linux.yaml/badge.svg)
![Latest macOS (serial)](https://github.com/BallisticLA/RandBLAS/actions/workflows/core-macos.yml/badge.svg)
![Latest macOS (OpenMP)](https://github.com/BallisticLA/RandBLAS/actions/workflows/openmp-macos.yaml/badge.svg)
![Old macOS (OpenMP)](https://github.com/BallisticLA/RandBLAS/actions/workflows/openmp-macos-13.yaml/badge.svg)

### Copyright and license

RandBLAS is licensed under the BSD 3-Clause License.
See [LICENSE](LICENSE) for information and copyright assertions.

### Source code

The source code can be found at the [RandBLAS github repository](https://github.com/BallisticLA/RandBLAS).
