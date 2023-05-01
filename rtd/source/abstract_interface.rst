RandBLAS Overview
=================



Random number generation
------------------------

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

The general workflow for sketching
----------------------------------

Computing a sketch with RandBLAS has four steps. 
  0. Get your hands on an RNGState :math:`\texttt{state}`.
  1. Define a distribution :math:`\mathcal{D}` over matrices.
  2. Using :math:`\texttt{state}`, sample a sketching operator :math:`S` from :math:`\mathcal{D}`.
  3. Use :math:`S` with a function that is *almost* identical to GEMM.

Each of these things can be done in one line of code.
The relevant structs and functions needed for the three later steps are described below.

Applying a sketching operator : a GEMM-like interface
-----------------------------------------------------

Left-sketching
^^^^^^^^^^^^^^

.. doxygenfunction:: RandBLAS::ramm::lskgex(blas::Layout layout, blas::Op transS, blas::Op transA, int64_t d, int64_t n, int64_t m, T alpha, SKOP &S, int64_t row_offset, int64_t col_offset, const T *A, int64_t lda, T beta, T *B, int64_t ldb)
   :project: RandBLAS

Right-sketching
^^^^^^^^^^^^^^^

.. doxygenfunction:: RandBLAS::ramm::rskgex(blas::Layout layout, blas::Op transA, blas::Op transS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, SKOP &S, int64_t i_os, int64_t j_os, T beta, T *B, int64_t ldb)
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

Sparse sketching operators
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenstruct:: RandBLAS::sparse::SparseDist
   :project: RandBLAS
   :members:

.. doxygenstruct:: RandBLAS::sparse::SparseSkOp
   :project: RandBLAS
   :members: 
