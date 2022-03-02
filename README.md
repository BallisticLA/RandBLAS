# proto RBLAS
This repository hosts BALLISTIC-internal RBLAS prototyping.
This includes a proper C++ RBLAS prototype and scratch work
for very rough benchmarking.

The directory "z_scratch/" is a scratch workspace for debugging, preparing examples, and informal benchmarking.

## Installing RBLAS
### Dependencies
RBLAS can be compiled with a C++11 compliant compiler and depends on CMake,
BLAS++ and Random123. RBLAS has optional dependencies on GTest and OpenMP.

#### BLAS++
Blas++ is a C++ API for the Basic Linear Algebra Subroutines. BLAS++ can be
installed with GNU make or CMake. RBLAS depends on the CMake install of
BLAS++.  One may compile and install BLAS++ from
[source](https://bitbucket.org/icl/blaspp/src/master/) using CMake with the
following recipe:

```shell
git clone https://bitbucket.org/icl/blaspp.git
mkdir blaspp-build
cd blaspp-build
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=../../blaspp-install \
    ../blaspp
make -j install
```

When configuring RBLAS use the `-Dblaspp_DIR=X` CMake variable to point to the
install of BLAS++ where `X` is the path to the directory containing the file
`blasppConfig.cmake`.

#### Random123
The Random123 library is a collection of counter-based random number generators.
One may install Random123 from
[source](https://github.com/DEShawResearch/random123) using the following
recipe:

```shell
git clone git@github.com:DEShawResearch/random123.git
cd random123/
make prefix=`pwd`/../random123-install install-include
```

When configuring RBLAS use the `-DRandom123_DIR=X` cmake variable to point to
the install of Random123 where `X` is the directory containing the Random123
header files.

#### GTest
GoogleTest is Google’s C++ testing and mocking framework.  GTest is an optional
dependency without which RBLAS regression tests will not be available. GTest
can be installed with your favorite package manager.

#### OpenMP
OpemMP is an open standard that enables code to be parallelized as it is
compiled. RBLAS detects the presence of OpenMP automatically and makes use of
it if it's found.

### RBLAS
RBLAS is configured with CMake and built with GNU make. First, see the section above
on installing [RBLAS dependencies](Dependencies). Once the dependencies are installed
RBLAS may be compiled and installed using the following recipe:

```shell
git clone git@github.com:BallisticLA/rblas.git
mkdir rblas-build
cd rblas-build
cmake -DCMNAKE_BUILD_TYPE=Release \
    -Dblaspp_DIR=`pwd`/../blaspp-install/lib64/blaspp/ \
    -DRandom123_DIR=`pwd`/../random123-install/include/ \
    -DCMAKE_INSTALL_PREFIX=`pwd`/../rblas-install \
    ../rblas/
make -j install
```
Note: this recipe assumes the directory structured used in the [above recipes](Dependencies)
for installing dependencies.

#### Testing
From the RBLAS build directory:

```
ctest
```

Note: tests are only available if GTest was found during the build.

## Using RBLAS
Once RBLAS has been compiled and installed (See [Installing
RBLAS](Installing-RBLAS)) it can be used like any other CMake project.
For instance, the following CMakeLists.txt demonstrates how an executable can
be linked to the RBLAS library:

```cmake
cmake_minimum_required(VERSION 3.0)
project(myexec)

find_package(rblas REQUIRED)

add_executable(myexec ...)
target_link_libraries(myexec rblas ...)
```

## The z_scratch/ directory 

If you work in this repo, create a folder with your name in ``z_scratch/``.
You have total authority over what goes in that folder.
If you have something presentable that you want to reference later, then a copy should be kept in the "sharing" folder.

When you make an entry in the sharing folder you should has the format "year-mm-dd-[your initials]-[letter]".
 * You substitute-in "[your initials]" with whatever short character string is appropriate for you (for Riley John Murray, that's rjm).
The "[letter]" should start every day with "a" and then increment along the alphabet.
 * Hopefully you won't have more than 26 folders you'd want to add to "sharing" on one day.

If you need source code for a third-party library in order to run your experiments, add a git submodule.
