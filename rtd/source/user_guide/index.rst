###################
RandBLAS User Guide
###################


It's useful to think of RandBLAS' sketching workflow in three steps.

  1. Get your hands on a random state.
  2. Define a sketching distribution, and use the random state to sample a sketching operator from that distribution.
  3. Apply the sketching operator with a function that's *almost* identical to GEMM.

To illustrate this workflow, suppose we have a 20,000-by-10,000 double-precision matrix :math:`A`  stored in column-major
layout. Suppose also that we want to compute a sketch of the form :math:`B = AS`, where :math:`S` is a Gaussian matrix of size 10,000-by-50.
This can be done as follows.

   .. code:: c++

      // step 1
      RandBLAS::RNGState state();
      // step 2
      RandBLAS::DenseDist D(10000, 50);
      RandBLAS::DenseSkOp<double> S(D, state);
      // step 3
      double B* = new double[20000 * 50];
      RandBLAS::sketch_general(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            20000, 50,  10000,
            1.0, A, 20000, S, 0.0, B, 20000
      ); // B = AS

RandBLAS has a wealth of capabilities that are not reflected in that code sippet.
For example, it lets you set an integer-valued the seed when defining :math:`\texttt{state}`, and it provides a wide range of both dense and sparse sketching operators.
It even lets you compute products against *submatrices* of sketching operators without ever forming the full operator in memory.

The full range of possibilities with RandBLAS are described in the pages linked below.


.. toctree::
    :maxdepth: 3

    Applying a sketching operator <sketching>
    Sketching distributions and sketching operators <operators>
    Random states <rng_details>
    Sparse data <sparse_data>
