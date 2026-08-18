# Developer Notes for RandBLAS

This file reviews aspects of RandBLAS' implementation that aren't (currently) suitable 
for our user guide.


 * Our basic random number generation is handled by [Random123](https://github.com/DEShawResearch/random123).
   We have small wrappers around Random123 code in ``RandBLAS/base.hh`` and ``RandBLAS/random_gen.hh``.
  
 * ``RandBLAS/dense_skops.hh`` has code for representing and sampling dense sketching operators.
   The sampling code is complicated because it supports multi-threaded random (sub)matrix generation, and yet the generated (sub)matrices are the same no matter how many threads
   you're using.

 * ``RandBLAS/sparse_skops.hh`` has code for representing and sampling sparse sketching operators.
   The sampling code has a customized method for repeatedly sampling from an index set without
   replacement, which is needed to quickly generate the structures used in statistically reliable
   sparse sketching operators.

## Sparse sampling and OpenMP

Sparse sampling assigns randomness to logical major-axis vectors rather than to physical
threads. If the first requested vector starts at `initial_counter`, then vector `i` starts at
`initial_counter + i * vec_nnz`. The state returned by the sampling routine is computed outside
the OpenMP region by adding `num_major_axis_vectors * vec_nnz` to the initial counter. Thread
scheduling therefore cannot change either the sampled operator or the returned state.

Short-axis-sparse operators (SASOs) sample without replacement. Each active thread owns a
restored permutation of length `dim_major` and a pivot array of length `vec_nnz`; no thread
shares mutable sampling workspace. This gives an `O(T * dim_major)` permutation-workspace cost
for `T` active threads. An internal policy limits `T` by the available OpenMP threads, the number
of major-axis vectors, the amount of sampling work, and the work available to amortize each
permutation. The specialized `vec_nnz == 1` path does not allocate permutation workspace.

Long-axis-sparse operators (LASOs) sample with replacement. Vector `i` first occupies a lane of
length `vec_nnz` at offset `i * vec_nnz`. A thread-private pair of hash maps merges duplicate
locations, the surviving entries are sorted by major coordinate, and a per-vector count records
the live prefix of each lane. A serial pass then packs lanes in increasing vector order. Every
packed destination precedes or equals its source, so this pass is safe in place and preserves the
canonical COO ordering without a second `O(nnz)` buffer.

 * [BLAS++ (aka blaspp)](https://github.com/icl-utk-edu/blaspp) is our portability layer for BLAS.
   We actually use very few functions in BLAS at time of writing (GEMM, SCAL, COPY, and
   AXPY) but we use its enumerations _everywhere_. Fast GEMM is important for sketching dense
   data with dense operators.

 * The ``sketch_general`` functions in ``RandBLAS/skge.hh`` are the main entry point for sketching dense data.
   These functions are small wrappers around functions with more BLAS-like names:
      * ``lskge3`` and ``rskge3`` are basically wrappers around GEMM.
      * ``lskges`` and ``rskges`` trigger an opaque call sequence that uses sparse matrix operations.

 * There is no widely accepted standard for sparse BLAS operations. This is a bummer because
   sparse matrices are super important in data science and scientific computing. In view of this,
   RandBLAS provides its own abstractions for sparse matrices (CSC, CSR, and COO formats).
   The abstractions can either own their associated data or just wrap existing data (say, data
   attached to a sparse matrix in Eigen). RandBLAS has reasonably flexible and high-performance code
   for multiplying a sparse matrix and a dense matrix. All code related to sparse matrices is in
   ``RandBLAS/sparse_data``. See that folder's [``DevNotes.md``](sparse_data/DevNotes.md) file for details.
