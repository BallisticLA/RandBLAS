
# Installing and using RandBLAS

Sections 1 through 3 of this file describe how to perform a *basic* installation
of RandBLAS and its dependencies.

Section 4 explains how RandBLAS can be used in other CMake projects.

Section 5 gives detailed recommendations on configuring BLAS++ and LAPACK++
for use with RandBLAS.
Its installation instructions for BLAS++ can be used in place
of the BLAS++ instructions in Section 1.

*We recommend that you not bother with Section 5 the first time you build RandBLAS.*


## 1. Required Dependencies: BLAS++ and Random123

BLAS++ is a C++ API for the Basic Linear Algebra Subroutines.
BLAS++ can be installed with GNU make or CMake;
RandBLAS requires the CMake install of BLAS++. 
Random123 is collection of counter-based random number generators.

We give recipies for installing BLAS++ and Random123 below.
Later on, we'll assume these recipes were executed from a directory
that contains (or will contain) the ``proto_rblas`` project directory as a subdirectory.

One can compile and install BLAS++ from
[source](https://bitbucket.org/icl/blaspp/src/master/) using CMake by running
```shell
git clone https://bitbucket.org/icl/blaspp.git
mkdir blaspp-build
cd blaspp-build
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=../../blaspp-install \
    -DCMAKE_BINARY_DIR=`pwd`
    ../blaspp
make -j install
```

One can install Random123 from
[source](https://github.com/DEShawResearch/random123) by running
```shell
git clone git@github.com:DEShawResearch/random123.git
cd random123/
make prefix=`pwd`/../random123-install install-include
```

## 2. Optional dependencies: GTest and OpenMP

GoogleTest is Google’s C++ testing and mocking framework.  GTest is an optional
dependency without which RandBLAS regression tests will not be available. GTest
can be installed with your favorite package manager.

OpemMP is an open standard that enables code to be parallelized as it is
compiled. RandBLAS detects the presence of OpenMP automatically and makes use of
it if it's found.

## 3. Building and installing RandBLAS

RandBLAS is configured with CMake and built with GNU make.
The configuration and build processes are simple once RandBLAS' dependencies are in place. 

Assuming you used the recipies from Section 1 to get RandBLAS' dependencies,
you can build download, build, and install RandBLAS as follows:

```shell
git clone git@github.com:BallisticLA/proto_rblas.git
mkdir RandBLAS-build
cd RandBLAS-build
cmake -DCMAKE_BUILD_TYPE=Release \
    -Dblaspp_DIR=`pwd`/../blaspp-install/lib/blaspp/ \
    -DRandom123_DIR=`pwd`/../random123-install/include/ \
    -DCMAKE_BINARY_DIR=`pwd` \
    -DCMAKE_INSTALL_PREFIX=`pwd`/../RandBLAS-install \
    ../proto_rblas/
make -j install  # might need "sudo make -j install"
ctest  # run unit tests (only if GTest was found by CMake)
```

Here are the conceptual meanings in the recipe's build flags:

* `-Dblaspp_DIR=X` means `X` is the directory containing the file `blasppConfig.cmake`.
   
    If you follow BLAS++ installation instructions from Section 5 instead of
    Section 1, then you'd set ``-Dblaspp_DIR=/opt/mklpp/lib/blaspp``.
    Recall that we do not recommend that you follow Section 5 the first time you
    build RandBLAS.

* `-DRandom123_DIR=Y` means `Y` is the directory containing the Random123
  header files.

* `-DCMAKE_INSTALL_PREFIX=Z` means subdirectories within `Z` will contain
   the RandBLAS binaries, header files, and CMake configuration files needed
   for using RandBLAS in other projects. You should make note of the directory
   that ends up containing the file ``RandBLAS.cmake``.


## 4. Using RandBLAS in other projects

Once RandBLAS has been compiled and installed it can be used like any other CMake project.
For instance, the following CMakeLists.txt demonstrates how an executable can
be linked to the RandBLAS library:

```cmake
cmake_minimum_required(VERSION 3.0)
project(myexec)

find_package(RandBLAS REQUIRED)

add_executable(myexec ...)
target_link_libraries(myexec RandBLAS ...)
```
In order to build that CMake project you'd need to specify a build flag ``-DRandBLAS_DIR=X``, where ``X`` is a directory that contains ``RandBLAS.cmake``.

The vast majority of projects that use RandBLAS will also use BLAS++ and LAPACK++.
Below we give recommendations on how to configure BLAS++ and LAPACK++.
If you follow those recommendations, then you'd use build flags
``-Dblaspp_DIR=/opt/mklpp/lib/blaspp`` and ``-Dlapackpp_DIR=/opt/mklpp/lib/lapackpp``
when running CMake for your project.

## 5. Tips

### Pay attention to the BLAS++ configuration

The performance of RandBLAS depends heavily on how BLAS++ is configured.
If performance matters to you then you should inspect the
information that's printed to screen when you run ``cmake`` for the BLAS++ installation.
Save that information somewhere while you're setting up your RandBLAS
development environment.

### Recommended BLAS++ and LAPACK++ configuration

LAPACK++ is a C++ API for a wide range of possible LAPACK implementations.
While LAPACK++ is not required for RandBLAS, it is required in 
RandLAPACK, and so it's prudent to configure BLAS++ and LAPACK++ at the same time.

We recommend you install BLAS++ and LAPACK++ so they link to Intel MKL
version 2022 or higher.
That version of MKL will come with CMake figuration files.
Those configuration files are extremely useful if
you want to make a project that connects RandBLAS and Intel MKL.
Such a situation might arise if you want to use RandBLAS together with
MKL's sparse linear algebra functionaliy.

One of the RandBLAS developers (Riley) has run into trouble
getting BLAS++ to link to MKL as intended.
Here's how Riley configured his BLAS++ and LAPACK++ installations:

1. Download BLAS++ source, create a new folder called ``build``
   at the top level of the BLAS++ project directory, and ``cd`` into that
   folder.
2. Run ``export CXX=gcc`` so that ``gcc`` is the default compiler for
   the current bash session.
3. Decide a common prefix for where you'll put BLAS++ and LAPACK++
   installation files. We recommend ``/opt/mklpp``.
4. Run the following CMake command 
    ```
    cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/mklpp \
        -DBLAS_LIBRARIES='-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread' \
        -Dbuild_tests=no ..
    ```
    Note how the MKL BLAS and threading libraries are specified explicitly with the ``-DBLAS_LIBRARIES`` flag.
    Using that flag is in contrast with simply setting ``-Dblas=mkl``,
    in which case the BLAS++ CMake recipe tries to configure MKL for you.
5. Run ``cmake --build .``
6. Run ``sudo make -j2 install``
7. Download LAPACK++ source, create a new folder called ``build`` at the top level
   of the LAPACK++ project directory, and ``cd`` into that folder.
8. Run the following CMake command
   ```
    cmake -DCMAKE_BUILD_TYPE=Debug \
       -Dblaspp_DIR=/opt/mklpp/lib/blaspp \
       -DCMAKE_INSTALL_PREFIX=/opt/mklpp \
       -DCMAKE_BINARY_DIR=`pwd` \
       -Dbuild_tests=no ..
    make -j2 install
    ```

You can then link to BLAS++ and LAPACK++ in other CMake projects
just by including ``find_package(blaspp)`` and ``find_package(lapackpp)``
in your ``CMakeLists.txt`` file, and then passing build flags
```
-Dblaspp_DIR=/opt/mklpp/lib/blaspp -Dlapackpp_DIR=/opt/mklpp/lib/lapackpp
```
when running ``cmake``.

### Installation trouble

RandBLAS has a GitHub Actions workflow to install it from scratch and run its suite of unit tests.
If you're having trouble installing RandBLAS, you can always refer to [that workflow file](https://github.com/BallisticLA/proto_rblas/tree/main/.github/workflows).
The workflow includes statements which print the working directory
and list the contents of that directory at various points in the installation.
We do that so that it's easier to infer a valid choice of directory structure for building RandBLAS.
