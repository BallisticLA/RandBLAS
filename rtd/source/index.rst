RandBLAS: sketching for randomized numerical linear algebra
===========================================================

RandBLAS is a C++ library for dimension reduction via random linear transformations.
These random linear transformations are called *sketching operators*.
The act of applying a sketching operator -- that is, the act of *sketching* -- is of fundamental importance to randomized numerical linear algebra.

RandBLAS is efficient, flexible, and reliable.
It can sample sketching operators from a wide range of dense and sparse distributions.
It uses CPU-based OpenMP acceleration to apply its sketching operators to dense data matrices stored in main memory.
It can be used in distributed environments through its ability to (reproducibly) compute products with *submatrices* of sketching operators.

Learn more by reading the `RandBLAS User Guide <user_guide/index.html>`_.

Source Code
-----------
Source code can be obtained at the `RandBLAS github repository <https://github.com/BallisticLA/RandBLAS>`_.


Build and Install
-----------------
RandBLAS is configured with CMake. The following CMake variables influence the build.

+-------------------------+----------------------------------------------------+
| CMake Variable          | Description                                        |
+-------------------------+----------------------------------------------------+
| CMAKE_BUILD_TYPE        | Release or Debug. The default is Release.          |
+-------------------------+----------------------------------------------------+
| blaspp_DIR              | The path to your local BLAS++ install              |
+-------------------------+----------------------------------------------------+
| Random123_DIR           | The path to your local random123 install           |
+-------------------------+----------------------------------------------------+


.. toctree::
   :hidden:

   User Guide <user_guide/index>
