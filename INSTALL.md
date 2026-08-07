
# Installing and using RandBLAS

This guide has four main sections and a native-Windows appendix.

Sections 1 through 3 describe how to build and install RandBLAS using CMake.

Section 4 explains how to use RandBLAS in other CMake projects.

Appendix A follows the same general flow for a native Windows build with MSVC.

If you want a TL;DR version of this guide, refer to one of the following.
 * Our GitHub Actions to [workflow files](https://github.com/BallisticLA/RandBLAS/tree/main/.github/workflows).
 * The [examples folder](https://github.com/BallisticLA/RandBLAS/tree/main/examples).


## 1. Required dependencies: a C++20 compatible compiler and BLAS++

RandBLAS uses C++20 [concepts](https://en.cppreference.com/w/cpp/language/constraints).
Make sure your compiler supports these. We test gcc ≥13 on Linux, and both Apple Clang
and Homebrew LLVM on macOS; older toolchains (such as gcc 8.5 with `-fconcepts`) are not
supported.

BLAS++ is a C++ API for the Basic Linear Algebra Subroutines.
It can be installed with GNU make or CMake.
If you want to use RandBLAS' CMake build system,
then it will be necessary to have built and installed BLAS++ via CMake.

RandBLAS includes its header-only counter-based random-number generators. There
is no separate random-number package to install or configure.

We give a recipe for installing BLAS++ below.
Later on, we'll assume this recipe was executed from a directory
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

**A note if you arrived here from RandLAPACK.** RandLAPACK vendors RandBLAS
as a git submodule pinned to an exact commit, and that pinned copy is the
only RandBLAS configuration RandLAPACK is developed and tested against. Do
not develop RandBLAS inside RandLAPACK's submodule checkout, and do not
point RandLAPACK at a RandBLAS working copy: if you want to work on RandBLAS
itself, clone this repository directly and build it as its own project, as
described below. Installing RandBLAS this way does not affect RandLAPACK in
any way; RandLAPACK keeps using its pinned submodule. (Package maintainers
building RandBLAS and RandLAPACK as separate packages should see
`RandLAPACK_EXTERNAL_RandBLAS` in RandLAPACK's INSTALL.md, which admits an
external RandBLAS only when it was built from exactly the pinned commit.)

The following CMake variables influence the RandBLAS build.

| CMake Variable   | Description                               |
|------------------|-------------------------------------------|
| CMAKE_BUILD_TYPE | Release or Debug. The default is Release. |
| blaspp_DIR       | The path to your local BLAS++ install     |

Assuming you used the recipes from Section 1 to get RandBLAS' dependencies,
you can download, build, and install RandBLAS as follows:

```shell
git clone git@github.com:BallisticLA/RandBLAS.git
mkdir RandBLAS-build
cd RandBLAS-build
cmake -DCMAKE_BUILD_TYPE=Release \
    -Dblaspp_DIR=`pwd`/../blaspp-install/lib/cmake/blaspp/ \
    -DCMAKE_BINARY_DIR=`pwd` \
    -DCMAKE_INSTALL_PREFIX=`pwd`/../RandBLAS-install \
    ../RandBLAS/
make -j install  # might need "sudo make -j install"
ctest  # run unit tests (only if GTest was found by CMake)
```

Here are the conceptual meanings of the recipe's other build flags:

* `-Dblaspp_DIR=X` means `X` is the directory containing the file `blasppConfig.cmake`.

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
target_link_libraries(myproject PUBLIC RandBLAS blaspp lapackpp)
```


## Appendix A. Native Windows installation with MSVC

This appendix gives a native Windows recipe; it does not use WSL. It follows
the same dependency, build, and downstream-use flow as the main guide. The
commands use `C:/randblas-work` as a replaceable workspace root and keep source,
build, and install directories separate.

Run the commands from an **x64 Native Tools Command Prompt for Visual Studio**.
The recipe uses the NMake generator, so `cmake --build` is serial and does not
need `--parallel`.

### A.1. Required dependencies: MSVC, oneMKL, and BLAS++

Install the following tools first:

* Visual Studio 2022 or Build Tools with **Desktop development with C++**;
* Git;
* CMake 3.24 or later;
* vcpkg.

Create the workspace directories:

```bat
mkdir C:\randblas-work
mkdir C:\randblas-work\src
mkdir C:\randblas-work\build
mkdir C:\randblas-work\install
```

The following command assumes vcpkg is installed at `C:\vcpkg`. Adjust that
path if necessary. Install the oneMKL port into a dedicated dependency prefix:

```bat
C:\vcpkg\vcpkg.exe install intel-mkl:x64-windows ^
  --x-install-root=C:\randblas-work\vcpkg-installed

set "MKLROOT=C:\randblas-work\vcpkg-installed\x64-windows"
set "PATH=%MKLROOT%\bin;%PATH%"
```

Build a shared, sequential, ILP64 BLAS++:

```bat
git clone --branch windows-portability ^
  https://github.com/RaphaelArkadyMeyerNYU/blaspp.git ^
  C:\randblas-work\src\blaspp

cmake --fresh ^
  -S C:/randblas-work/src/blaspp ^
  -B C:/randblas-work/build/blaspp ^
  -G "NMake Makefiles" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_INSTALL_PREFIX=C:/randblas-work/install/blaspp ^
  -DBUILD_SHARED_LIBS=ON ^
  -Duse_cmake_find_blas=false ^
  -DBLAS_LIBRARIES="C:/randblas-work/vcpkg-installed/x64-windows/lib/mkl_intel_ilp64_dll.lib;C:/randblas-work/vcpkg-installed/x64-windows/lib/mkl_sequential_dll.lib;C:/randblas-work/vcpkg-installed/x64-windows/lib/mkl_core_dll.lib" ^
  -Dblas_int=ilp64 ^
  -Dblas_threaded=false ^
  -Duse_openmp=false ^
  -Dgpu_backend=none ^
  -Dbuild_tests=OFF

cmake --build C:/randblas-work/build/blaspp --target install
```

### A.2. Optional dependencies: GoogleTest and OpenMP

GoogleTest is needed only to build and run the RandBLAS test suite. A minimal
installation can omit GoogleMock:

```bat
git clone --branch v1.17.0 ^
  https://github.com/google/googletest.git ^
  C:\randblas-work\src\googletest

cmake --fresh ^
  -S C:/randblas-work/src/googletest ^
  -B C:/randblas-work/build/googletest ^
  -G "NMake Makefiles" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_INSTALL_PREFIX=C:/randblas-work/install/googletest ^
  -DBUILD_GMOCK=OFF ^
  -DINSTALL_GTEST=ON

cmake --build C:/randblas-work/build/googletest --target install
```

MSVC supplies OpenMP support. RandBLAS's CMake configuration automatically
selects `/openmp:experimental` under MSVC because its sparse kernels use
`#pragma omp simd`. No OpenMP flag needs to be added manually.

OpenMP is optional. To request a serial build explicitly, add
`-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE` to the RandBLAS configuration
command in the next section.

### A.3. Building, installing, and testing RandBLAS

Clone RandBLAS, configure it against the installed BLAS++, and enable the test
suite:

```bat
git clone https://github.com/BallisticLA/RandBLAS.git ^
  C:\randblas-work\src\RandBLAS

cmake --fresh ^
  -S C:/randblas-work/src/RandBLAS ^
  -B C:/randblas-work/build/RandBLAS ^
  -G "NMake Makefiles" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_INSTALL_PREFIX=C:/randblas-work/install/RandBLAS ^
  -Dblaspp_DIR=C:/randblas-work/install/blaspp/blaspp ^
  -DCMAKE_PREFIX_PATH=C:/randblas-work/install/googletest ^
  -DBUILD_TESTS=ON

cmake --build C:/randblas-work/build/RandBLAS --target install

ctest --test-dir C:/randblas-work/build/RandBLAS --output-on-failure
```

The exact location of `blasppConfig.cmake` can vary between BLAS++ revisions.
The value of `blaspp_DIR` must be the directory containing that file. If the
path above does not exist, locate the installed file with:

```bat
dir C:\randblas-work\install\blaspp\blasppConfig.cmake /s /b
```

Similarly, `CMAKE_PREFIX_PATH` points at the GoogleTest installation prefix,
not directly at `GTestConfig.cmake`.

AddressSanitizer is optional. Add `-DSANITIZE_ADDRESS=ON` to configure an
instrumented build. This requires the optional **C++ AddressSanitizer**
component from the Visual Studio Installer. When tests are enabled, configure
and install a separate ASan-enabled copy of GoogleTest by adding
`-DCMAKE_CXX_FLAGS="/fsanitize=address /Zi"` to the GoogleTest CMake command.
Then set `CMAKE_PREFIX_PATH` to that GoogleTest installation when configuring
the sanitized RandBLAS build.

### A.4. Using the installed package

The CMake target remains the same on every platform:

```cmake
cmake_minimum_required(VERSION 3.24)
project(my_randblas_project LANGUAGES CXX)

find_package(RandBLAS REQUIRED)

add_executable(myexec main.cc)
target_link_libraries(myexec PRIVATE RandBLAS)
```

Configure and build that consumer by pointing `CMAKE_PREFIX_PATH` at the
RandBLAS installation:

```bat
cmake --fresh ^
  -S C:/path/to/my_randblas_project ^
  -B C:/path/to/my_randblas_project-build ^
  -G "NMake Makefiles" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_PREFIX_PATH=C:/randblas-work/install/RandBLAS

cmake --build C:/path/to/my_randblas_project-build
```

The installed `RandBLASConfig.cmake` records the BLAS++ location used to build
RandBLAS, so ordinary consumers should not need to supply that path again.

Repository-owned tests copy imported dependency DLLs beside their executables.
An arbitrary downstream application is responsible for its own deployment.
When running the example consumer above from the build tree, ensure that the
oneMKL and BLAS++ runtime directories are on `PATH`, or copy the required DLLs
beside the executable:

```bat
set "PATH=C:\randblas-work\vcpkg-installed\x64-windows\bin;C:\randblas-work\install\blaspp\bin;%PATH%"

C:\path\to\my_randblas_project-build\myexec.exe
```
