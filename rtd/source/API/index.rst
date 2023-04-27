RandBLAS API Documentation
==========================



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



Dense sketching
---------------

Do you want to use a Gaussian sketching operator?
What about a sketching operator whose entries are an drawn i.i.d. uniformly from :math:`[-1, 1]`?

In either case, you'll need to do four things.
  0. Get your hands on an RNGState :math:`\texttt{state}`.
  1. Define a distribution :math:`\mathcal{D}` over dense matrices.
  2. Using :math:`\texttt{state}`, sample a sketching operator :math:`S` from :math:`\mathcal{D}`.
  3. Use :math:`S` with a function that is *almost* identical to GEMM.

Each of these things can be done in one line of code.
The relevant structs and functions needed for the three later steps are described under "Essentials of dense sketching," below.

Essentials of dense sketching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: RandBLAS::dense::DenseDist
   :project: RandBLAS
   :members:

.. doxygenstruct:: RandBLAS::dense::DenseSkOp
   :project: RandBLAS
   :members: 

.. doxygenfunction:: RandBLAS::dense::lskge3
   :project: RandBLAS

Advanced aspects of dense sketching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: RandBLAS::dense::DenseDistName
   :project: RandBLAS


Helper functions for dense sketching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: RandBLAS::dense::fill_buff

.. doxygenfunction:: RandBLAS::dense::realize_full


Sparse sketching
----------------

So, you want to sketch a data matrix with a random sparse operator, do you?

Well, you'll need to do four things.
  0. Get your hands on an RNGState :math:`\texttt{state}`.
  1. Define a distribution :math:`\mathcal{D}` over sparse matrices.
  2. Using :math:`\texttt{state}`, sample a sketching operator :math:`S` from :math:`\mathcal{D}`.
  3. Use :math:`S` with a function that is *almost* identical to GEMM.

Each of these things can be done in one line of code.
The relevant structs and functions needed for the three later steps are described under "Essentials of sparse sketching," below.


Essentials of sparse sketching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: RandBLAS::sparse::SparseDist
   :project: RandBLAS
   :members:

.. doxygenstruct:: RandBLAS::sparse::SparseSkOp
   :project: RandBLAS
   :members: 

.. doxygenfunction:: RandBLAS::sparse::lskges
   :project: RandBLAS

Advanced aspects of sparse sketching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: RandBLAS::sparse::SparsityPattern
   :project: RandBLAS


Helper functions for sparse sketching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: RandBLAS::sparse::transpose

.. doxygenfunction:: RandBLAS::sparse::fill_sparse
