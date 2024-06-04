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

 * [BLAS++ (aka blaspp)](https://github.com/icl-utk-edu/blaspp) is our portability layer for BLAS.
   We actually use very few functions in BLAS at time of writing (GEMM, GEMV, SCAL, COPY, and
   AXPY) but we use its enumerations _everywhere_. Fast GEMM is important for sketching dense
   data with dense operators.

 * The ``sketch_general`` functions in ``RandBLAS/skge.hh`` are the main entry point for sketching dense data.
   These functions are small wrappers around functions with more BLAS-like names:
      * ``lskge3`` and ``rskge3`` in ``RandBLAS/skge3_to_gemm.hh``.
      * ``lskges`` and ``rskges`` in ``RandBLAS/skges_to_spmm.hh``.
   The former pair of functions are just fancy wrappers around GEMM.
   The latter pair of functions trigger a far more opaque call sequence, since they rely on sparse
   matrix operations.

 * There is no widely accepted standard for sparse BLAS operations. This is a bummer because
   sparse matrices are super important in data science and scientific computing. In view of this,
   RandBLAS provides its own abstractions for sparse matrices (CSC, CSR, and COO formats).
   The abstractions can either own their associated data or just wrap existing data (say, data
   attached to a sparse matrix in Eigen). RandBLAS has reasonably flexible and high-performance code
   for multiplying a sparse matrix and a dense matrix. All code related to sparse matrices is in
   ``RandBLAS/sparse_data``. See that folder's [``DevNotes.md``](sparse_data/DevNotes.md) file for details.
