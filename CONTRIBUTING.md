# Contributing to RandBLAS

RandBLAS is a header-only C++ library with performance-sensitive kernels and
reproducible random-number generation. This guide explains the project
constraints and development workflow that matter when changing it.

Follow the [`STYLE_GUIDE.md`](STYLE_GUIDE.md) for source and documentation
conventions. Improvements to the style guide are always in scope for any pull
request, even when they are not required by its primary purpose.

## Before you start

Open an issue or talk with the maintainers on the
[RandLAPACK Discord server](https://discord.gg/R4qj8Er9YW) before making a
large public-API change or undertaking a performance optimization. A small
bug fix does not need a design discussion first.

Read the documentation closest to your change:

- The [FAQ](https://randblas.readthedocs.io/en/stable/FAQ.html),
  [tutorial](https://randblas.readthedocs.io/en/stable/tutorial/index.html),
  and [API reference](https://randblas.readthedocs.io/en/stable/api_reference/index.html)
  describe the user-facing library.
- [`RandBLAS/DevNotes.md`](RandBLAS/DevNotes.md) describes the main
  implementation, and
  [`RandBLAS/sparse_data/DevNotes.md`](RandBLAS/sparse_data/DevNotes.md)
  describes the sparse kernels and dispatchers.
- [`test/DevNotes.md`](test/DevNotes.md) and the other `DevNotes.md` files
  describe specialized tests and implementation details.
- Pages 17--28 of the
  [2024 LDRD final report](https://www.osti.gov/servlets/purl/2462906) provide
  broader project background.

Developer notes describe the implementation at the time they were written.
Check the code and tests when a note appears stale, and update the note if your
change makes it inaccurate.

## Project constraints

Preserve these properties unless a change explicitly sets out to revise one:

- RandBLAS requires C++20 and exposes BLAS-like operations.
- Random generation goes through Random123 and RandBLAS's `RNGState` type.
  Do not use `std::random` outside a test with a specific reason for it.
- A sampled random matrix must be identical for every OpenMP thread count.
  Partition random-number streams by logical data position, not by thread
  arrival order.
- Do not expose standard-library data structures in the public API. For this
  rule, the public API is the set of declarations selected under
  [`rtd/source/api_reference/`](rtd/source/api_reference/).
- Preserve memory ownership, thread safety, numerical behavior, and public API
  compatibility unless the proposal and review call for a deliberate change.

Keep patches focused. Do not mix a functional change with a repository-wide
formatting pass.

## Set up a development environment

For a standalone clone, follow [`INSTALL.md`](INSTALL.md). Its installer and
plain-CMake instructions are the supported starting points for Linux, macOS,
and Windows.

The maintainers validate changes in the Spack-backed RandNLA workspace.
Homebrew may provide tools for a standalone macOS build, but it is not the
reference development environment. If you are using the RandNLA workspace and
its build directory is configured, run:

```bash
cd /path/to/randnla/dev
source sourceme.sh
make -C build-randblas -j
ctest --test-dir build-randblas --output-on-failure
```

To build the examples after a library change, install the headers and rebuild
the separate examples tree:

```bash
cd /path/to/randnla/dev
source sourceme.sh
make -C build-randblas install
make -C build-randblas-examples -j
```

## Test the change

Put a regression test at the same abstraction level as the behavior:

- random-number tests under `test/basic_rng/`;
- data-structure tests under `test/datastructures/`;
- linear-operation and sparse-dispatch tests under `test/linops/`.

Run focused tests while developing. Before requesting review for a code
change, run the full suite for your configured build. If you use the RandNLA
workspace, use the Spack-backed commands shown above; maintainers use that
environment for final validation.

Sampling changes need tests with more than one OpenMP thread count. Sparse
dispatch changes need coverage for every affected combination of storage
format, transpose flags, and dense layout. `left_spmm` has twelve principal
paths; `right_spmm` transforms its inputs and delegates to that dispatcher.

## Update the documentation

Update documentation in the same patch when behavior, a public declaration,
or an implementation rationale changes.

- Public declarations need comments that render through Sphinx and Breathe.
  The style guide explains the supported comment forms.
- API additions or removals may require a corresponding directive under
  `rtd/source/api_reference/`.
- Tutorials, FAQ material, and other public prose belong under `rtd/source/`.
- Implementation rationale belongs in the nearest `DevNotes.md`.

[`rtd/DevNotes.md`](rtd/DevNotes.md) describes the website build. Check the
rendered result when changing public comments or reStructuredText, rather than
judging only the Doxygen XML.

## Measure performance changes

Discuss a performance optimization with the maintainers before undertaking
it. Benchmark the old and new implementations under comparable conditions,
then report the matrix sizes, compiler, BLAS backend, OpenMP configuration,
thread count, and results.

A faster result in one configuration does not establish a general
improvement. Keep correctness tests separate from benchmark evidence.

## Before requesting review

- [ ] The patch has one clear purpose and no unrelated formatting sweep.
- [ ] Public API names and argument order match neighboring BLAS-like operations.
- [ ] Ownership, RNG-state, and behavioral changes are documented.
- [ ] Focused tests pass; code changes also pass the full test suite.
- [ ] Random output is independent of the OpenMP thread count.
- [ ] Sparse changes cover every affected dispatch path.
- [ ] Public comments render through the Sphinx--Breathe pipeline.
- [ ] Performance claims include comparable before-and-after measurements.
