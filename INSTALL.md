
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
Make sure your compiler supports these. We test gcc ≥13 on Linux, and both Apple Clang
and Homebrew LLVM on macOS; older toolchains (such as gcc 8.5 with `-fconcepts`) are not
supported.

BLAS++ is a C++ API for the Basic Linear Algebra Subroutines.
It can be installed with GNU make or CMake.
If you want to use RandBLAS' CMake build system,
then it will be necessary to have built and installed BLAS++ via CMake.

Random123 is a header-only library of counter-based random number generators.

We give recipes for installing BLAS++ and Random123 below.
Later on, we'll assume these recipes were executed from a directory
that contains (or will contain) the ``RandBLAS`` project directory as a subdirectory.

One can compile and install BLAS++ from
[source](https://bitbucket.org/icl/blaspp/src/master/) using CMake by running the following.
Note that all CMake and system terms for BLAS++ use the name ``blaspp`` instead of ``BLAS++``.
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
limited versions of ``clang`` and ``clang++``). See [this GitHub issue comment](https://github.com/BallisticLA/RandBLAS/issues/86#issue-2248281376)
for more info.


## 3. Building and installing RandBLAS

The following CMake variables influence the RandBLAS build.

| CMake Variable   | Description                               |
|------------------|-------------------------------------------|
| CMAKE_BUILD_TYPE | Release or Debug. The default is Release. |
| blaspp_DIR       | The path to your local BLAS++ install     |
| Random123_DIR    | The path to your local random123 install  |

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

Here are the conceptual meanings of the recipe's other build flags:

* `-Dblaspp_DIR=X` means `X` is the directory containing the file `blasppConfig.cmake`.

* `-DRandom123_DIR=Y` means `Y` is the directory containing the Random123
  header files.

* `-DCMAKE_INSTALL_PREFIX=Z` means subdirectories within `Z` will contain
   the RandBLAS binaries, header files, and CMake configuration files needed
   for using RandBLAS in other projects. The CMake configuration files are
   installed to `Z/lib/cmake/RandBLAS/`.


## 4. Using RandBLAS in other projects

Once RandBLAS has been compiled and installed it can be used like any other CMake project.
For instance, the following CMake snippet demonstrates how an executable can
be linked to the RandBLAS library:

```cmake
cmake_minimum_required(VERSION 3.12)
find_package(RandBLAS REQUIRED)
add_executable(myexec ...)
target_link_libraries(myexec RandBLAS ...)
```
In order to build that CMake project you'd need to point CMake at the RandBLAS installation. The recommended way is ``-DCMAKE_PREFIX_PATH=Z``, where ``Z`` is the installation prefix from the previous section. Alternatively, set ``-DRandBLAS_DIR=Z/lib/cmake/RandBLAS`` to skip CMake's package-search and name the config directory directly.

Most projects that use RandBLAS will also use LAPACK++.
Here is example CMake code for such a project. Note that it references BLAS++ in the final line (as ``blaspp``),
but it doesn't have a ``find_package`` command for BLAS++. That's because when CMake is told to find RandBLAS,
the RandBLAS installation will tell CMake where to find blaspp as a dependency.
Note also that LAPACK++ is referenced as ``lapackpp``.
```cmake
cmake_minimum_required(VERSION 3.12)
project(my_randblas_project)
# ^ The project name can be whatever you want.
find_package(RandBLAS REQUIRED)
find_package(lapackpp REQUIRED)

set(myproject_cxx_source my_project.cc)
add_executable(my_project ${myproject_cxx_source})
target_include_directories(myproject PUBLIC ${Random123_DIR})
target_link_libraries(myproject PUBLIC RandBLAS blaspp lapackpp)
```
