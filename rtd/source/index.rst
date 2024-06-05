.. toctree::
   :hidden:
   :maxdepth: 3

   Installation <installation/index>
   Tutorial <tutorial/index>
   API Reference <api_reference/index>
   Changelog <updates/index>

.. default-domain:: cpp


RandBLAS: sketching for randomized numerical linear algebra
===========================================================

RandBLAS is a C++ library for dimension reduction via random linear transformations.
These random linear transformations are called *sketching operators*.
The act of applying a sketching operator -- that is, the act of *sketching* -- is of fundamental importance to randomized numerical linear algebra.

RandBLAS is efficient, flexible, and reliable.
It uses CPU-based OpenMP acceleration to apply its sketching operators to dense or sparse data matrices stored in main memory.
All sketches produced by RandBLAS are dense.
As such, dense data matrices can be sketched with dense or sparse operators, while sparse data matrices can only be sketched with dense operators.
RandBLAS can be used in distributed environments through its ability to (reproducibly) compute products with *submatrices* of sketching operators.

Learn more by reading our `tutorial <tutorial/index.html>`_ or our `API reference <api_reference/index.html>`_.
If we've piqued your interest, try RandBLAS yourself!
We've got a handy `installation guide <installation/index.html>`_  on this website.


Source Code
-----------
Source code can be obtained at our `github repository <https://github.com/BallisticLA/RandBLAS>`_.
