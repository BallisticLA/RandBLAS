RandBLAS: sketching for randomized numerical linear algebra
===========================================================

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

   User Guide <guide/index>

.. toctree::
   :hidden:

   API Documentation <API/index>
