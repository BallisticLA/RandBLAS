.. toctree::
  :maxdepth: 3

.. Note to self: I can first describe CBRNGs mathematically. Then I get to implementation details.

******************************************************************************
Sampling a sketching operator
******************************************************************************

RandBLAS relies on counter-based random number generators (CBRNGs) from Random123.
A CBRNG returns a random number upon being called with two integer parameters: the *counter* and the *key*.
The time required for the CBRNG to return does not depend on either of these parameters.
A serial application can set the key at the outset of the program and never change it, while
parallel applications should use different keys across different threads.
Sequential calls to the CBRNG with a fixed key should use different values for the counter. 


RandBLAS doesn't expose CBRNGs directly. Instead, it exposes an abstraction of
a CBRNG's state as defined in the :ref:`RNGState <rngstate_api>` class.
RNGState objects are needed to construct sketching operators.

.. _constructing_rng_states_tut:

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


Constructing your first sketching operator
==========================================

RandBLAS provides several constructors for the DenseSkOp and SparseSkOp classes.
However, the *recommended* constructors for these classes just accept two parameters:
a representation of a distribution (i.e., a DenseDist or a SparseDist) and an RNGState.

For example, the following code produces a :math:`10000 \times 50` dense sketching operator 
whose entries are iid samples from the uniform distribution over :math:`[-1, 1]`.

   .. code:: c++

      RandBLAS::RNGState my_state();
      RandBLAS::DenseDist my_dist(10000, 50);
      RandBLAS::DenseSkOp<double> S(my_dist, my_state);
      // my_state is stored as a constant value S.seed_state.
      // S.seed_state will be accessed by RandBLAS' random sampling
      // functions behind the scenes, only when needed.
  
We note that the numerical precision of the sketching operator must be specified with a template parameter;
the entries of the sketching operator are defined by sampling in single precision and then
casting the sample to double if needed.

Formal API docs for the recommended constructors can be found :ref:`here <densedist_and_denseskop_api>` and :ref:`here <sparsedist_and_sparseskop_api>`.


Constructing your :math:`N^{\text{th}}` sketching operator, for :math:`N > 1`
==============================================================================

Suppose you have an application that requires two statistically independent dense
sketching operators, :math:`\texttt{S1}` and :math:`\texttt{S2}`, each of size
:math:`10000 \times 50`.  How should you get your hands on these objects?

.. warning::
    If you try to construct those sketching operators as follows ...

    .. code:: c++

      RandBLAS::RNGState my_state();
      RandBLAS::DenseDist my_dist(10000, 50);
      RandBLAS::DenseSkOp<double> S1(my_dist, my_state);
      RandBLAS::DenseSkOp<double> S2(my_dist, my_state);

    *then your results would be invalid! Far from being independent,* :math:`\texttt{S1}`
    *and* :math:`\texttt{S2}` *would be equal from a mathematical perspective.*

One correct approach is to then call the constructor for :math:`\texttt{S2}`
using :math:`\texttt{S1.next_state}` as its RNGState argument:

  .. code:: c++

    RandBLAS::RNGState my_state();
    RandBLAS::DenseDist my_dist(10000, 50);
    RandBLAS::DenseSkOp<double> S1(my_dist, my_state);
    // ^ Defines S1 from a mathematical perspective. Computes S1.next_state,
    //   but otherwise performs no work.
    RandBLAS::DenseSkOp<double> S2(my_dist, S1.next_state);

Another valid approach is to declare two RNGState objects from the beginning using
different keys, as in the following code:

  .. code:: c++

    RandBLAS::RNGState my_state1(19);
    // ^ An RNGState with zero'd counter and key initialized to 19.
    RandBLAS::RNGState my_state2(93);
    // ^ An RNGState with zero'd counter and key initialized to 93.
    RandBLAS::DenseDist my_dist(10000, 50);
    RandBLAS::DenseSkOp<double> S1(my_dist, my_state1);
    RandBLAS::DenseSkOp<double> S2(my_dist, my_state2);
    // ^ S1 and S2 are defined only from a mathematical perspective.
    //   No real work is performed here.


