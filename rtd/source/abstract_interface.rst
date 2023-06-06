RandBLAS User Guide
===================


The basic workflow for sketching
--------------------------------

Computing a sketch with RandBLAS has four steps. 
  0. Get your hands on an RNGState struct variable; let's call it :math:`\texttt{state}`. This can be as simple as 

      .. code:: c++

            RandBLAS::RNGState state(); // could pass an unsigned integer.

  1. Define a distribution :math:`\mathcal{D}` over random matrices with desired dimensions. For example,

      .. code:: c++

            RandBLAS::dense::DenseDist D{.n_rows = 1000, .n_cols = 100};
   
  2. Using :math:`\texttt{state}`, sample a sketching operator :math:`S` from :math:`\mathcal{D}`. For example,

      .. code:: c++

            RandBLAS::dense::DenseSkOp S(D, state);

  3. Use :math:`S` with a function that is *almost* identical to GEMM.

We elaborate on each of these steps below.
Of particular note is our coverage of the sketching operator distributions that RandBLAS supports.

Applying a sketching operator : a GEMM-like interface
-----------------------------------------------------

RandBLAS currently only supports sketching matrices that would be called "general" in BLAS.
This is done with an overloaded and templated function called :math:`\texttt{sketch_general}`.
The order of arguments provided to :math:`\texttt{sketch_general}` is very important; it
determines whether the sketching operator is applied from the left or from the right.

Left-sketching
^^^^^^^^^^^^^^

.. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m, T alpha, SKOP &S, int64_t i_off, int64_t j_off, const T *A, int64_t lda, T beta, T *B, int64_t ldb)
   :project: RandBLAS

Right-sketching
^^^^^^^^^^^^^^^

.. doxygenfunction:: RandBLAS::sketch_general(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, SKOP &S, int64_t i_off, int64_t j_off, T beta, T *B, int64_t ldb)
   :project: RandBLAS

Choosing a sketching distribution
---------------------------------

Dense sketching operators
^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenstruct:: RandBLAS::dense::DenseDist
   :project: RandBLAS
   :members:

.. doxygenstruct:: RandBLAS::dense::DenseSkOp
   :project: RandBLAS
   :members: 

.. doxygenfunction:: RandBLAS::dense::fill_dense(const DenseDist &D, T *buff, RNGState<RNG> const& state)
   :project: RandBLAS

.. doxygenfunction:: RandBLAS::dense::fill_dense(SKOP &S)
   :project: RandBLAS

Sparse sketching operators
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenstruct:: RandBLAS::sparse::SparseDist
   :project: RandBLAS
   :members:

.. doxygenstruct:: RandBLAS::sparse::SparseSkOp
   :project: RandBLAS
   :members: 

.. doxygenfunction:: RandBLAS::sparse::fill_sparse(SKOP &S)
   :project: RandBLAS

Details on Random number generation
-----------------------------------

RandBLAS relies on counter-based random number generators (CBRNGs).
The mathematical state of a CBRNG is specified by two integers: a *counter* and a *key*.
We use the following class to represent a CBRNG and its underlying state.

.. doxygenstruct:: RandBLAS::base::RNGState
   :project: RandBLAS

.. important::

   Every RandBLAS function that involves random sampling needs an RNGState as input!

There are two ways to construct an RNGState from scratch:

.. code:: c++

   RandBLAS::RNGState s1();     // key and counter are initialized to 0.
   RandBLAS::RNGState s2(42);   // key set to 42, counter set to 0.

Note that in both cases the counter is initialized to zero.
This is important: you should never set the counter yourself!
If you want statistically independent runs of the same program, then you can start with different values for the key.


Advanced material on random number generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An RNGState has :math:`\texttt{ctr}` and :math:`\texttt{key}` members.
These members are in fact arrays of integers, rather than integers themselves.
Users should not manipulate these values directly.
It is reasonable to access them for debugging purposes and for creating copies, as below:

.. code:: c++

   RandBLAS::RNGState s3(s2.ctr, s2.key); // s3 is a copy of s2

Every RNGState has an associated template parameter, RNG.
The default value of the RNG template parameter is :math:`\texttt{Philox4x32}`.
An RNG template parameter with name :math:`\texttt{GeneratorNxW}` will represent
the counter and key by an array of (at most) :math:`\texttt{N}` unsiged :math:`\texttt{W}`-bit integers.