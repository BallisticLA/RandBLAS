### RandBLAS : A header-only C++ library for sketching in randomized linear algebra

RandBLAS facilitates implementation of high-level randomized linear algebra algorithms, like randomized low-rank SVD.
It does this by providing basic functionality for sketching with a BLAS-like interface.

Right now RandBLAS' main use is to provide the sketching backend for [RandLAPACK](https://github.com/BallisticLA/RandLAPACK).
Our goal is for RandBLAS to become a standard like the BLAS, in that hardware vendors might
release their own optimized implementations of algorithms which conform to the RandBLAS API.

Please swing by the [**RandLAPACK Discord server**](https://discord.gg/R4qj8Er9YW) if you have questions about RandBLAS or would like to get involved with RandBLAS' development.

For those who are new to randomized linear algebra, we recommend you check out [this 35-minute YouTube video](https://www.youtube.com/watch?v=6htbyY3rH1w) on the subject.

### Documentation

We have three types of documentation.
 1. Traditional source code comments.
 2. Web documentation, split into a [tutorial](https://randblas.readthedocs.io/en/latest/tutorial/index.html) and an [API reference](https://randblas.readthedocs.io/en/latest/api_reference/index.html).
 3. Developer notes; [one](RandBLAS/DevNotes.md) for RandBLAS as a whole and [another](RandBLAS/sparse_data/DevNotes.md) for our sparse matrix functionality.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) before preparing a change.
Source and documentation conventions are in
[`STYLE_GUIDE.md`](STYLE_GUIDE.md).

Detailed installation instructions are in [INSTALL.md](INSTALL.md).

### Continuous integration builds

![core](https://github.com/BallisticLA/RandBLAS/actions/workflows/core.yml/badge.svg)
![thread-sanitizer](https://github.com/BallisticLA/RandBLAS/actions/workflows/thread-sanitizer.yml/badge.svg)
![examples](https://github.com/BallisticLA/RandBLAS/actions/workflows/examples.yml/badge.svg)
![downstream-consumer](https://github.com/BallisticLA/RandBLAS/actions/workflows/downstream-consumer.yml/badge.svg)
![docs](https://github.com/BallisticLA/RandBLAS/actions/workflows/docs.yml/badge.svg)

### Copyright and license

RandBLAS is licensed under the BSD 3-Clause License.
See [LICENSE](LICENSE) for information and copyright assertions.

### Source code

The source code can be found at the [RandBLAS github repository](https://github.com/BallisticLA/RandBLAS).
