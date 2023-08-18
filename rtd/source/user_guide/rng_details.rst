*************
Random states
*************

RandBLAS relies on counter-based random number generators (CBRNGs).
The mathematical state of a CBRNG is specified by two integers: a *counter* and a *key*.
We use the following class to represent a CBRNG and its underlying state.

.. doxygenstruct:: RandBLAS::RNGState
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


Advanced material
=================

Counters and keys in CBRNGs
---------------------------

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
