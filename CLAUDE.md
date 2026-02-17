# RandBLAS Project Guide for Claude

## Project Overview

RandBLAS is a header-only C++ library for sketching in randomized linear algebra. It provides BLAS-like functionality for applying random sketching operators to dense and sparse matrices, enabling efficient implementation of randomized algorithms like low-rank SVD, least squares, and other dimension reduction techniques.

**Core Purpose**: Enable efficient, flexible, and reliable sketching operations for randomized numerical linear algebra on CPUs using OpenMP parallelization.

## Key Resources

- **Web Documentation**: https://randblas.readthedocs.io/en/1.1.0/
- **Main Repository**: https://github.com/BallisticLA/RandBLAS
- **DevNotes**: Critical implementation details are in `RandBLAS/DevNotes.md`, `RandBLAS/sparse_data/DevNotes.md`, and `test/DevNotes.md`

## Architecture and Code Organization

### Directory Structure

```
RandBLAS/               # Core library headers (header-only)
├── base.hh            # Basic types, Random123 wrappers
├── random_gen.hh      # RNG infrastructure
├── dense_skops.hh     # Dense sketching operators (DenseSkOp)
├── sparse_skops.hh    # Sparse sketching operators (SparseSkOp)
├── skge.hh            # sketch_general() - main entry point for sketching
├── sksy.hh            # Symmetric sketching
├── skve.hh            # Vector sketching
├── util.hh            # Utility functions
└── sparse_data/       # Sparse matrix abstractions and operations
    ├── base.hh        # Sparse matrix types (COO, CSR, CSC)
    ├── coo_matrix.hh  # Coordinate format
    ├── csr_matrix.hh  # Compressed sparse row
    ├── csc_matrix.hh  # Compressed sparse column
    ├── conversions.hh # Format conversions
    ├── spmm_dispatch.hh    # Sparse matrix-matrix multiply dispatch
    ├── coo_spmm_impl.hh    # COO kernel implementations
    ├── csr_spmm_impl.hh    # CSR kernel implementations
    ├── csc_spmm_impl.hh    # CSC kernel implementations
    ├── trsm_dispatch.hh    # Sparse triangular solve dispatch
    ├── csr_trsm_impl.hh    # CSR triangular solve implementations
    ├── csc_trsm_impl.hh    # CSC triangular solve implementations
    └── sksp.hh        # Sketching sparse data with dense operators

test/                  # GoogleTest-based test suite
├── test_basic_rng/    # RNG tests (deterministic and statistical)
├── test_datastructures/ # Tests for DenseSkOp, SparseSkOp, sparse matrices
├── test_matmul_cores/ # Low-level kernel tests (lskge3, rskge3, left_spmm, right_spmm)
├── test_matmul_wrappers/ # High-level API tests (sketch_general, sketch_sparse, etc.)
└── test_sparse_trsm/  # Sparse triangular solve tests

examples/              # Example programs demonstrating usage
├── sparse-data-matrices/  # Sparse matrix performance benchmarks
├── sparse-low-rank-approx/ # Sparse SVD/QRCP examples
└── total-least-squares/   # TLS with dense and sparse operators

rtd/                   # ReadTheDocs source files
```

### Key Components

1. **Random Number Generation**: Uses [Random123](https://github.com/DEShawResearch/random123) for counter-based PRNGs with thread-safe, reproducible parallel generation.

2. **BLAS Portability**: Uses [BLAS++](https://github.com/icl-utk-edu/blaspp) as the portability layer. Main BLAS functions used: GEMM, GEMV, SCAL, COPY, AXPY.

3. **Sketching Operators**:
   - `DenseSkOp`: Gaussian, uniform, sparse (CountSketch variants)
   - `SparseSkOp`: Sparse random matrices in COO format internally

4. **Main API Functions**:
   - `sketch_general()`: Entry point for sketching dense data (routes to `lskge3`, `rskge3`, `lskges`, `rskges`)
   - `left_spmm()`, `right_spmm()`: Sparse matrix × dense matrix multiplication
   - `sketch_sparse()`: Sketching sparse data with dense operators
   - `trsm()`: Sparse triangular solve (B ← αA⁻¹B for triangular sparse matrix A)

## Critical Implementation Details

### Multi-Threading and Reproducibility

**IMPORTANT**: RandBLAS guarantees that randomly generated matrices are **identical regardless of the number of threads** used. This is achieved through careful management of RNG state in `dense_skops.hh` and `sparse_skops.hh`.

- When modifying sampling code, preserve this thread-independence property
- Tests verify this property; always run tests after RNG-related changes

### Sparse Matrix Dispatch Logic

The sparse matrix multiplication functions (`left_spmm`, `right_spmm`) use a **12-codepath dispatch system** based on:
- Matrix format (COO, CSR, CSC)
- Transposition flags (`opA`, `opB`)
- Memory layout (RowMajor, ColMajor)

See `RandBLAS/sparse_data/DevNotes.md` for the full dispatch flow. Key points:

- `right_spmm` reduces to `left_spmm` by flipping flags
- Transposition of sparse matrices creates lightweight views (CSR ↔ CSC)
- All 12 codepaths should be covered by tests

The sparse triangular solve function (`trsm`) handles transposition similarly:
- Transposition creates lightweight CSR ↔ CSC views and flips uplo (upper ↔ lower)
- Supports both unit and non-unit diagonal matrices
- Includes validation modes for checking structural properties

### Function Naming Conventions

- `lskgeX`: **Left** sketching, **ge**neral (dense) data, variant X
- `rskgeX`: **Right** sketching, **ge**neral (dense) data, variant X
- `lskges`: Left sketching with **s**parse operator
- `lskge3`: Left sketching with dense operator (calls GEMM, "3" for 3-argument GEMM-like)
- `lsksp3`: Left sketching **sp**arse data (where "left" refers to operator position)

**Counterintuitive detail**: In `lsksp3` and `rsksp3`, the "left/right" refers to the operator's position. But these functions call `right_spmm`/`left_spmm` respectively, where "left/right" refers to the sparse data matrix position. See `sparse_data/DevNotes.md` lines 59-74.

## Coding Standards and Conventions

### Language and Compiler Requirements

- **C++20** required (uses concepts for type constraints)
- Some compilers (e.g., gcc 8.5) may need `-fconcepts` flag
- macOS may need `-D __APPLE__` for `sincosf`/`sincos` functions

### Style Guidelines

- Header-only library: all code in `.hh` files
- Use BLAS++ enumerations (`blas::Layout`, `blas::Op`, etc.) extensively
- Follow existing patterns for GEMM-like APIs (side flags, transposition, layouts)
- Prefer templates with C++20 concepts over traditional template metaprogramming

### Performance Considerations

- **OpenMP is critical** for performance (both dense operator sampling and sparse operations)
- Fast GEMM is essential for dense sketching operations
- Sparse matrix kernels are hand-tuned; changes should be benchmarked
- BLAS++ configuration significantly affects performance

**Before making performance optimizations**:
1. Ask for confirmation on approach
2. Benchmark before and after
3. Document performance implications in PR/commit message

## Testing Requirements

### When to Run Tests

**Always run `ctest` after making code changes**, especially:
- Any modifications to sampling logic (dense or sparse operators)
- Changes to sparse matrix kernels or dispatch logic
- RNG-related changes
- New feature implementations

### Test Organization

Tests are organized by abstraction level:

- `test_basic_rng/`: RNG correctness (deterministic and statistical)
- `test_datastructures/`: Data structure constructors, accessors, format conversions
- `test_matmul_cores/`: Low-level kernels (`lskge3`, `left_spmm`, etc.)
- `test_matmul_wrappers/`: High-level API (`sketch_general`, `sketch_sparse`, `sketch_symmetric`)
- `test_sparse_trsm/`: Sparse triangular solve

### Running Tests

```bash
cd RandBLAS-build
ctest                    # Run all tests
ctest -R test_name       # Run specific test
ctest -V                 # Verbose output
```

### Adding New Tests

- Use GoogleTest framework
- Follow patterns in existing test files
- For new sketching operators: test both left and right application, various transposition flags
- For sparse operations: ensure all relevant codepaths are covered

## Build System

### CMake Configuration

Key CMake variables:
- `blaspp_DIR`: Path to BLAS++ installation (containing `blasppConfig.cmake`)
- `Random123_DIR`: Path to Random123 headers
- `CMAKE_BUILD_TYPE`: Release or Debug
- `CMAKE_CXX_FLAGS`: May need `-D __APPLE__` on macOS

### Installation

```bash
mkdir RandBLAS-build && cd RandBLAS-build
cmake -DCMAKE_BUILD_TYPE=Release \
      -Dblaspp_DIR=/path/to/blaspp-install/lib/cmake/blaspp/ \
      -DRandom123_DIR=/path/to/random123-install/include/ \
      ../RandBLAS/
make -j install
ctest
```

See `INSTALL.md` for full details.

## Documentation

### Types of Documentation

1. **Source code comments**: Inline documentation in headers
2. **Web documentation**: Tutorial and API reference at readthedocs.io
3. **DevNotes**: Implementation details not suitable for user guide

### Updating Documentation

- **Code comments**: Update when changing function signatures or behavior
- **DevNotes**: Update when implementation approach changes significantly
- **Web docs**: Source in `rtd/` directory, built with Sphinx, deployed to ReadTheDocs

### Style for Documentation

- **Concise explanations**: Focus on what changed and why
- Reference line numbers when discussing specific code locations (e.g., `file.hh:42`)
- Use mathematical notation when helpful but don't over-explain standard linear algebra concepts
- Link to relevant sections of web docs or DevNotes for deeper context

## Git and Release Procedures

### Commit Messages

- Follow conventional commit style where appropriate
- For complex changes, explain the "why" not just the "what"
- Reference issue numbers when applicable

### Branches

- `main`: Primary development branch
- Version tags: `X.Y.Z` format (e.g., `1.1.0`)

### Release Process

See `PROCEDURES.md` for full release procedures. Key steps:
1. Create git tag in `X.Y.Z` format
2. Write release notes
3. Update ReadTheDocs default version
4. Create GitHub release

### CI/CD

GitHub Actions workflows test:
- Ubuntu (OpenMP)
- macOS (serial and OpenMP, current and older versions)

All CI tests must pass before merging.

## Working with Claude on RandBLAS

### Preferred Workflow

1. **Always run tests** after making code changes to verify correctness
2. **Ask before performance optimizations** - confirm approach and document changes
3. **Concise explanations** - briefly explain what changed and why, without excessive detail
4. **Reference files with line numbers** using format `[file.hh:42](RandBLAS/file.hh#L42)`

### Common Tasks

**Adding a new sketching operator**:
1. Define operator struct in `dense_skops.hh` or `sparse_skops.hh`
2. Implement sampling logic (preserve thread-independence!)
3. Add `sketch_general` support in `skge.hh`
4. Add tests in `test_datastructures` and `test_matmul_wrappers`
5. Update web documentation in `rtd/`
6. Run full test suite

**Modifying sparse matrix kernels**:
1. Review dispatch logic in `sparse_data/DevNotes.md`
2. Make changes to specific kernel in `coo_spmm_impl.hh`, `csr_spmm_impl.hh`, or `csc_spmm_impl.hh`
3. Verify all affected codepaths have test coverage
4. Run tests and benchmarks
5. Document performance implications

**Improving code quality**:
1. Identify area for refactoring
2. Review existing tests to understand expected behavior
3. Make incremental changes, running tests after each step
4. Consider performance implications
5. Update DevNotes if implementation approach changes

### What to Avoid

- **Don't** change RNG behavior without carefully preserving thread-independence
- **Don't** optimize sparse kernels without benchmarking
- **Don't** add BLAS++ dependencies beyond current subset without discussion
- **Don't** break CMake configuration for downstream projects
- **Don't** assume all 12 sparse matrix codepaths are tested (verify coverage)

## External Dependencies

### Required

- **BLAS++** (blaspp): BLAS portability layer - must be built with CMake
- **Random123**: Header-only RNG library
- **C++20 compiler**: gcc ≥9, clang ≥10, or equivalent

### Optional

- **GoogleTest**: Required for testing (`ctest`)
- **OpenMP**: Required for performance (parallel sampling and sparse operations)
- **LAPACK++** (lapackpp): Often used in projects that depend on RandBLAS

### Dependency Notes

- BLAS++ configuration heavily affects performance - users should inspect CMake output
- Random123 headers must be in include path for downstream projects
- OpenMP detection can fail on macOS with default system compilers (use homebrew gcc/clang)

## Security and Correctness

- No known security vulnerabilities
- Primary correctness concerns: RNG reproducibility, numerical accuracy, thread safety
- Statistical tests verify RNG quality (Kolmogorov-Smirnov tests for distribution correctness)
- Deterministic tests compare against reference values for Random123 generators

## Getting Help

- GitHub Issues: https://github.com/BallisticLA/RandBLAS/issues
- Documentation: https://randblas.readthedocs.io/
- Contact: Project maintainers listed in repository

---

*This CLAUDE.md file was created to help Claude Code understand the RandBLAS project structure, conventions, and workflows. Update it as the project evolves.*
