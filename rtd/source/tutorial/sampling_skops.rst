.. toctree::
  :maxdepth: 3

.. Note to self: I can first describe CBRNGs mathematically. Then I get to implementation details.

******************************************************************************
Sampling a sketching operator
******************************************************************************

RandBLAS relies on counter-based random number generators (CBRNGs).
The mathematical state of a CBRNG is specified by two integers: a *counter* and a *key*.
We use the following class to represent a CBRNG and its underlying state.

TODO: link to RNGState documentation.

Constructing RNGStates
======================

There are two ways to construct an RNGState from scratch:

.. code:: c++

   RandBLAS::RNGState s1();     // key and counter are initialized to 0.
   RandBLAS::RNGState s2(42);   // key set to 42, counter set to 0.

Note that in both cases the counter is initialized to zero.
This is important: you should never set the counter yourself!
If you want statistically independent runs of the same program, then you can start with different values for the key.

You can also construct an RNGState with a copy operation:

.. code:: c++

  RandBLAS::RNGState s3(s1);   // s3 is a copy of s1.


RNGStates associated with sketching operators
=============================================

Every RandBLAS function that involves random sampling needs access to an RNGState.
However, most of those functions are members of sketching operator objects
(like DenseSkOp and SparseSkOp), which accept the RNGState at construction time
and access it when needed. 

TODO: explain the ``.seed_state`` and ``.next_state`` pattern.

