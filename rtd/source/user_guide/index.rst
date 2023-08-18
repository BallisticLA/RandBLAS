###################
RandBLAS User Guide
###################


Computing a sketch with RandBLAS can be done in three easy steps.

  1. Get your hands on an RNGState variable.
     This can be as simple as the following.

      .. code:: c++

            RandBLAS::RNGState state();

    Alternatively, you can pass an unsigned integer to an RNGState constructor as a seed for the random state.
    If an integer isn't provided (like above), then the seed defaults to zero.

  2. Define a distribution :math:`\mathcal{D}` over random matrices with desired properties, then use :math:`\texttt{state}`
     to sample a sketching operator :math:`S` from :math:`\mathcal{D}`.
     
     For example, here is how we could define a 1000-by-50 sketching operator with iid Gaussian entries stored in double precision.

      .. code:: c++

            RandBLAS::DenseDist D(1000, 50);
            RandBLAS::DenseSkOp<double> S(D, state);

     RandBLAS provides a wide range of possibilities for the choice of sketching distribution. In particular, it supports
     both dense and sparse sketching operators. More on this later!

  3. Use :math:`S` with a function that is *almost* identical to GEMM. For example, here is how we could compute
     a sketch :math:`A S` where :math:`A` is a 2000-by-1000 double-precision matrix stored in column-major order.

      .. code:: c++

            double B* = new double[2000 * 50];
            RandBLAS::sketch_general(
                blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                2000, 50,  1000,
                1.0, A, 2000, S, 0.0, B, 2000
            ); // B = A S.

     RandBLAS has been designed with efficiency and performance in mind. So it's also possible to apply a 
     *submatrix* of a sketching operator without so much as even generating the full operator to begin with.

We elaborate on each of these steps in the pages linked below.

.. toctree::
    :maxdepth: 3

    Applying a sketching operator <sketching>
    Sketching distributions and sketching operators <operators>
    Random states <rng_details>
