.. toctree::
   :hidden:
   :maxdepth: 3

   Installation <installation/index>
   Tutorial <tutorial/index>
   API Reference <api_reference/index>
   Changelog <updates/index>
   FAQ and Limitations <FAQ>

.. default-domain:: cpp


RandBLAS: sketching for randomized numerical linear algebra
===========================================================

RandBLAS is a C++ library for randomized linear dimension reduction — an operation commonly known as *sketching*.
We built RandBLAS to make it easier to write and debug high-performance implementations of sketching-based algorithms.

RandBLAS is efficient, flexible, and reliable.
It uses CPU-based OpenMP acceleration to apply its sketching operators to matrices stored in main memory.
It includes dense and sparse sketching operators (e.g., Gaussian operators, CountSketch, OSNAPs, etc …), which can 
be applied to dense or sparse data in any combination that leads to a dense sketch.

With RandBLAS and an LAPACK-like library at your disposal, you can implement
a huge range of shared-memory randomized algorithms for matrix computations.
RandBLAS can be used in distributed environments through its ability to compute products with *submatrices* of sketching operators, 
without ever realizing the entire sketching operator in memory.

Learn more by reading our `tutorial <tutorial/index.html>`_ or our `API reference <api_reference/index.html>`_.
If we've piqued your interest, try RandBLAS yourself!
We've got a handy `installation guide <installation/index.html>`_  on this website
and `examples <https://github.com/BallisticLA/RandBLAS/tree/main/examples>`_ in
our `GitHub repository <https://github.com/BallisticLA/RandBLAS>`_.
