Installation
============


RandBLAS is a header-only C++20 library with two required dependencies. One of these
dependencies (`Random123 <https://github.com/DEShawResearch/random123>`_) is header-only,
while the other (`BLAS++ <https://github.com/icl-utk-edu/blaspp>`_) needs to be compiled.

Having a compiled dependency makes setting up RandBLAS a little more complicated than
setting up other header-only libraries. RandBLAS also has OpenMP and GoogleTest as 
optional dependencies. Access to OpenMP is essential for RandBLAS to achieve its 
best possible performance.

RandBLAS is most useful when called from programs that can access LAPACK,
or an equivalent library for dense matrix computations. However, we don't
require that such a library is available.


CMake users
-----------
RandBLAS offers a CMake build system.
This system manages a simple configuration step (populating ``RandBLAS/config.h``),
connecting to BLAS++ (assuming it was built with CMake), and building unit tests.

You can opt to *install* RandBLAS if you want other CMake projects to 
include it as a dependency. Formal installation just consists 
of copying the header files and CMake metadata into a directory of your choosing.

See 
`INSTALL.md <https://github.com/BallisticLA/RandBLAS/blob/main/INSTALL.md>`_
for detailed build and installation instructions.
Check out our `examples <https://github.com/BallisticLA/RandBLAS/tree/main/examples>`_
for CMake projects that use RandBLAS and `LAPACK++ <https://github.com/icl-utk-edu/lapackpp>`_
to implement high-level randomized algorithms.

.. warning::

  Make sure to use the flag ``-Dblas_int=int64`` in the CMake configuration line for BLAS++
  If you don't do that then you might get int32, which can lead to issues for large matrices.

Everyone else
-------------
Strictly speaking, we only need three things to use RandBLAS in other projects.

1. ``RandBLAS/config.h``, filled according to the instructions in ``RandBLAS/config.h.in``.

2. The locations of Random123 header files.

3. The locations of the header files and compiled binary for BLAS++ (which will
   referred to as "blaspp" when installed on your system).

If you have these things at hand, then compiling a RandBLAS-dependent
program is just a matter of specifying standard compiler flags. 

We recommend that you take a look at 
`INSTALL.md <https://github.com/BallisticLA/RandBLAS/blob/main/INSTALL.md>`_.
even if you aren't using CMake, since it has additional 
advice about selecting an acceptable compiler or getting RandBLAS
to see OpenMP.
