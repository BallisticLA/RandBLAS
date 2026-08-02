.. toctree::
  :maxdepth: 3


******************************************************************************
Sampling a sketching operator
******************************************************************************

RandBLAS includes native, header-only counter-based random-number generators
(CBRNGs). A CBRNG is a stateless block function of a counter and a key. RandBLAS
sampling functions consume a state abstraction that binds such an engine to one
counter and one key.

This organization lets RandBLAS assign counter blocks to matrix coordinates
instead of relying on the order in which threads happen to run. Sampling a full
operator or one of its submatrices is therefore reproducible and independent of
the OpenMP thread count.


.. _constructing_rng_states_tut:

Constructing RNG states
=======================

Most applications should use :cpp:type:`RandBLAS::DefaultRNGState`, whose
engine is ``RandBLAS::rng::Philox<4, 32, 10>``:

.. code:: c++

   RandBLAS::DefaultRNGState s1{};     // zero counter and default key
   RandBLAS::DefaultRNGState s2{42};   // zero counter and key mapped from seed 42

The scalar seed is mapped to a key by the engine's stable ``make_key`` function;
it should be treated as a seed rather than as the key's representation. Using
different seeds is the ordinary way to obtain independent program runs.

An RNG state is copyable:

.. code:: c++

   RandBLAS::DefaultRNGState s3{s1};


Constructing your first sketching operator
==========================================

The recommended DenseSkOp and SparseSkOp constructors accept a distribution and
an RNG state. For example, the following code defines a
:math:`10000 \times 50` dense sketching operator whose entries are independent
standard-normal samples:

.. code:: c++

   RandBLAS::DefaultRNGState state{};
   RandBLAS::DenseDist dist(10000, 50);
   RandBLAS::DenseSkOp<double> S(dist, state);
   // state is copied into S.seed_state. Sampling happens only when needed.

The numerical precision of the sketching operator is a template parameter.
Entries are sampled in single precision and cast to double when needed.

Formal API docs for the recommended constructors can be found
:ref:`here <densedist_and_denseskop_api>` and
:ref:`here <sparsedist_and_sparseskop_api>`.


Constructing your :math:`N^{\text{th}}` sketching operator, for :math:`N > 1`
==============================================================================

Constructing two operators from the same distribution and state defines the
same mathematical operator, not two independent ones:

.. code:: c++

   RandBLAS::DefaultRNGState state{};
   RandBLAS::DenseDist dist(10000, 50);
   RandBLAS::DenseSkOp<double> S1(dist, state);
   RandBLAS::DenseSkOp<double> S2(dist, state); // S2 is the same as S1

Use the first operator's ``next_state`` for the second operator:

.. code:: c++

   RandBLAS::DefaultRNGState state{};
   RandBLAS::DenseDist dist(10000, 50);
   RandBLAS::DenseSkOp<double> S1(dist, state);
   RandBLAS::DenseSkOp<double> S2(dist, S1.next_state);

Alternatively, start with two states constructed from different seeds:

.. code:: c++

   RandBLAS::DefaultRNGState state1{19};
   RandBLAS::DefaultRNGState state2{93};
   RandBLAS::DenseDist dist(10000, 50);
   RandBLAS::DenseSkOp<double> S1(dist, state1);
   RandBLAS::DenseSkOp<double> S2(dist, state2);


Engine and state details
========================

Users who need direct block access can name the engine and state explicitly:

.. code:: c++

   using Engine = RandBLAS::rng::Philox<4, 32, 10>;
   using State = RandBLAS::RNGState<Engine>;

   State state{1234};
   Engine::res_t block{};
   state.generate(block); // writes every element of the output-only array
   state.advance(1);      // add one to the engine's multiword counter

The engine provides ``ctr_t``, ``key_t``, and ``res_t`` aliases. A state exposes
the same aliases and const ``counter()`` and ``key()`` accessors. Generation does
not mutate the state: advancing by one block is always explicit. The default
Philox engine produces exactly the same integer blocks as Philox4x32-10 in
Random123 for the same counter and key. Philox is a statistical generator, not a
cryptographic random-number generator.

``RandBLAS::rng::RepackedOutput`` can expose each result word as narrower chunks
without changing the block boundary. Chunks are least-significant first within
each source word: ``0xAABBCCDD`` becomes ``{0xCCDD, 0xAABB}`` with 16-bit words
and ``{0xDD, 0xCC, 0xBB, 0xAA}`` with 8-bit words, regardless of host byte order.
Current RandBLAS samplers accept native 32- or 64-bit result words; repacked
8- and 16-bit results are provided for direct use and future sampler work.

The dense Gaussian transform calls the host ``sin``, ``cos``, ``log``, and
``sqrt`` functions. Its final bits can therefore vary across math libraries,
compilers, or architectures even though the underlying integer stream and block
assignment are fixed.
