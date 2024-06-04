
# Installing and using RandBLAS

This guide has five sections.

Sections 1 through 3 describe how to build and install RandBLAS using CMake.

Section 4 explains how to use RandBLAS in other CMake projects.

Section 5 concludes with extra tips.

If you want a TL;DR version of this guide, refer to one of the following.
 * Our GitHub Actions to [workflow files](https://github.com/BallisticLA/RandBLAS/tree/main/.github/workflows).
 * The [examples folder](https://github.com/BallisticLA/RandBLAS/tree/main/examples).


## 1. Required dependencies: a C++20 compatible compiler, BLAS++, and Random123

RandBLAS uses C++20 [concepts](https://en.cppreference.com/w/cpp/language/constraints).
Make sure your compiler supports these. Some compilers (like gcc 8.5) might need to be
invoked with an additional flag (``-fconcepts``) in order to support this aspect of the
C++20 standard. See [this issue](https://github.com/BallisticLA/RandBLAS/issues/90) for more info.

BLAS++ is a C++ API for the Basic Linear Algebra Subroutines.
It can be installed with GNU make or CMake;
RandBLAS requires the CMake install of BLAS++. 

Random123 is a collection of counter-based random number generators.

We give recipes for installing BLAS++ and Random123 below.
Later on, we'll assume these recipes were executed from a directory
that contains (or will contain) the ``RandBLAS`` project directory as a subdirectory.

One can compile and install BLAS++ from
[source](https://bitbucket.org/icl/blaspp/src/master/) using CMake by running the following.
Note that all CMake-related terms for BLAS++ use the name ``blaspp`` instead of ``BLAS++``.
```shell
git clone https://github.com/icl-utk-edu/blaspp.git
mkdir blaspp-build
cd blaspp-build
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=`pwd`/../blaspp-install \
    -DCMAKE_BINARY_DIR=`pwd` \ 
    -Dbuild_tests=OFF \
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

GoogleTest (aka *GTest*) is Google’s C++ testing and mocking framework.  It is an optional
dependency, without which RandBLAS regression tests will not be available. It
can be installed with your favorite package manager.

OpenMP is a standard that enables code to be parallelized as it is compiled.
RandBLAS does not strictly require OpenMP, but it needs OpenMP to quickly
sample dense sketching operators and to quickly perform any sparse matrix computations.

RandBLAS' CMake configuration step should automatically detect if OpenMP is available.
Sometimes the CMake configuration will fail to recognize OpenMP even if it's 
on your system. This is especially common with the default system compilers on macOS
(you can execute ``gcc`` or ``g++`` on macOS, but those are just aliased to 
limited versions of ``clang`` and ``clang++``). So if you're on macOS then we 
recommend you install new versions of clang/clang++ via Homebrew, and call
```
export CC=<path to the clang you got from Homebrew>
export CXX=<path to the clang++ you got from Homebrew>
```

## 3. Building and installing RandBLAS

Assuming you used the recipes from Section 1 to get RandBLAS' dependencies,
you can download, build, and install RandBLAS as follows:

```shell
git clone git@github.com:BallisticLA/RandBLAS.git
mkdir RandBLAS-build
cd RandBLAS-build
cmake -DCMAKE_BUILD_TYPE=Release \
    -Dblaspp_DIR=`pwd`/../blaspp-install/lib/cmake/blaspp/ \
    -DRandom123_DIR=`pwd`/../random123-install/include/ \
    -DCMAKE_BINARY_DIR=`pwd` \
    -DCMAKE_INSTALL_PREFIX=`pwd`/../RandBLAS-install \
    ../RandBLAS/
make -j install  # might need "sudo make -j install"
ctest  # run unit tests (only if GTest was found by CMake)
```

If you're running macOS, then it may be necessary to specify
an additional flag to CMake
```shell
     -DCMAKE_CXX_FLAGS="-D __APPLE__"
```
This flag is needed to avoid compiler errors with the "sincosf" and "sincos"
functions in "random_gen.hh".

Here are the conceptual meanings of the recipe's other build flags:

* `-Dblaspp_DIR=X` means `X` is the directory containing the file `blasppConfig.cmake`.

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
cmake_minimum_required(VERSION 3.11)
find_package(RandBLAS REQUIRED)
add_executable(myexec ...)
target_link_libraries(myexec RandBLAS ...)
```
In order to build that CMake project you'd need to specify a build flag ``-DRandBLAS_DIR=X``, where ``X`` is a directory that contains ``RandBLAS.cmake``.

The vast majority of projects that use RandBLAS will also use BLAS++ and LAPACK++.
Here is example CMake code for such a project. Note that it references BLAS++ in the final line (as ``blaspp``),
but it doesn't have a ``find_package`` command for BLAS++. That's because when CMake is told to find RandBLAS,
the RandBLAS installation will tell CMake where to find blaspp as a dependency.
Note also that LAPACK++ is referenced as ``lapackpp``.
```cmake
cmake_minimum_required(VERSION 3.11)
project(my_randblas_project)
# ^ The project name can be whatever you want.
find_package(RandBLAS REQUIRED)
find_package(lapackpp REQUIRED)

set(myproject_cxx_source my_project.cc)
add_executable(my_project ${myproject_cxx_source})
target_include_directories(myproject PUBLIC ${Random123_DIR})
target_link_libraries(myproject PUBLIC RandBLAS blaspp lapackpp)
```

## 5. Tips

The performance of RandBLAS depends heavily on how BLAS++ is configured.
If performance matters to you then you should inspect the
information that's printed to screen when you run ``cmake`` for the BLAS++ installation.
Save that information somewhere while you're setting up your RandBLAS
development environment.

[An earlier version](https://github.com/BallisticLA/RandBLAS/blob/9d0a03fa41fd7c126b252002a54c2f2562fae31a/INSTALL.md#5-tips)
of this installation guide had specific recommendations for configuring BLAS++ and LAPACK++ on Intel machines.
Those recommendations may be useful to you if you're having a hard time getting these libraries setup correctly.
We removed those recommendations from this guide, since they encouraged a somewhat bad practice of installing BLAS++
and LAPACK++ to a system-wide location (under ``/opt/``) instead of a location that's used for one project.
