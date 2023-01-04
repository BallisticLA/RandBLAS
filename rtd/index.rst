The RandBLAS  User’s Guide
==========================

Source Code
-----------
Source code can be obtained at the `RandBLAS github repository <https://github.com/BallisticLA/RandBLAS>`_.

Documentation
-------------


Base
^^^^

.. doxygenstruct:: RandBLAS::base::RNGState
   :project: RandBLAS

Dense
^^^^^

.. doxygenstruct:: RandBLAS::dense::DenseSkOp
   :project: RandBLAS


.. doxygenfunction:: RandBLAS::dense::lskge3
   :project: RandBLAS

Sparse
^^^^^^

.. doxygenstruct:: RandBLAS::sparse::SparseSkOp
   :project: RandBLAS

.. doxygenfunction:: RandBLAS::sparse::lskges
   :project: RandBLAS




Doxygen
--------
RandBLAS's C++ sources are documented via Doxygen at the `RandBLAS Doxygen site <doxygen/index.html>`_.


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



