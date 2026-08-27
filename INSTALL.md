
# Installing and using RandBLAS

## Quick start: the installer scripts

If you just want a working RandBLAS, run the installer for your platform. It
builds RandBLAS and every dependency it needs into a self-contained
`RandNLA-project` directory beside your clone, installs nothing system-wide,
and does not touch your shell configuration.

```bash
bash installers/install.sh            # Linux and macOS
```

```powershell
powershell -ExecutionPolicy Bypass -File installers\install.ps1    # Windows
```

**You supply the toolchain; the installer supplies everything above it.** You
need a C++20 compiler, CMake 3.21 or later, and Git — on Windows, in an *x64*
developer shell. The script does not install compilers or package managers.
When something is missing it says so and points at the usual way to get it.

Useful options, common to both scripts:

| Option | Effect |
|---|---|
| `--blas=` / `-Backend` | `auto`, `openblas`, `mkl`, `accelerate`, `custom` |
| `--project-dir=` / `-ProjectDir` | where dependencies, builds and installs go |
| `--prefix=` / `-Prefix` | install RandBLAS itself somewhere else |
| `--examples` / `-Examples` | also build `examples/` (see below) |
| `--fresh`, `--no-tests`, `-j N` | rebuild from scratch, skip GoogleTest, set parallelism |
| `--yes` / `-Yes` | never prompt; also the behavior when stdin is redirected |

Run with `--help` for the full list. Every option has an environment-variable
equivalent, and already-installed dependencies are reused when you point at
them with `BLASPP_INSTALL_DIR`, `RANDOM123_INSTALL_DIR` or `GTEST_ROOT`.

### Sharing dependencies with RandLAPACK

Both installers use the same `RandNLA-project` layout and both honour
`RANDNLA_PROJECT_DIR`. Set it once and whichever installer runs second reuses
the first one's BLAS++ instead of building a second copy:

```bash
export RANDNLA_PROJECT_DIR=$HOME/RandNLA-project    # Linux, macOS
setx RANDNLA_PROJECT_DIR C:\RandNLA-project         # Windows
```

A dependency is reused only when it was built from the same source *and* in a
compatible configuration; a BLAS++ built for a different backend or integer
width is rebuilt rather than silently reused.

### Examples are opt-in

`examples/` is not built by default. It needs two dependencies RandBLAS itself
does not — LAPACK++ and `fast_matrix_market` — and it requires OpenMP, which
stock Apple Clang does not provide. The installer offers to build them when it
finishes, and prints the exact command to do it later.

### Building without the installer

The installer is a convenience, never a requirement. Everything it does is
reproducible with plain CMake and pre-installed dependencies, which is what
sections 1 through 3 describe and what packagers should follow. See
**Appendix B** for the packaging contract.

---

The rest of this guide has four main sections and two appendices.

Sections 1 through 3 describe how to build and install RandBLAS using CMake.

Section 4 explains how to use RandBLAS in other CMake projects.

Appendix A follows the same general flow for a native Windows build with MSVC.

Appendix B lists the configurations we test, and what a conda-forge or Spack
recipe needs to know.

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


## Appendix A. Native Windows installation with MSVC

This appendix gives a native Windows recipe; it does not use WSL. It follows
the same dependency, build, and downstream-use flow as the main guide. The
commands use `C:/randblas-work` as a replaceable workspace root and keep source,
build, and install directories separate.

Run the commands from an **x64 Native Tools Command Prompt for Visual Studio**.
The recipe uses the NMake generator, so `cmake --build` is serial and does not
need `--parallel`.

### A.1. Required dependencies: MSVC, oneMKL, BLAS++, and Random123

Install the following tools first:

* Visual Studio 2022 or Build Tools with **Desktop development with C++**;
* Git;
* CMake 3.24 or later;
* vcpkg (the copy bundled with Visual Studio's C++ workload works; so does a
  standalone clone).

Create the workspace directories:

```bat
mkdir C:\randblas-work
mkdir C:\randblas-work\src
mkdir C:\randblas-work\build
mkdir C:\randblas-work\install
```

Install the oneMKL port into a dedicated dependency prefix. The commands below
use vcpkg's manifest mode, which every vcpkg distribution supports -- the copy
bundled with Visual Studio has no classic-mode instance, so a plain
`vcpkg install intel-mkl:x64-windows` fails there. First write a minimal
manifest (the bundled vcpkg requires the `builtin-baseline` field; the pin
below is vcpkg release 2026.07.29, whose intel-mkl port is 2025.2.0):

```bat
mkdir C:\randblas-work\vcpkg-manifest
(
echo {
echo   "name": "randblas-windows-deps",
echo   "version-string": "1",
echo   "builtin-baseline": "9e593bb18ea69cc5095e012465dcd675a822ed0d",
echo   "dependencies": [ "intel-mkl" ]
echo }
) > C:\randblas-work\vcpkg-manifest\vcpkg.json
```

Then install from the manifest directory. The command below assumes the
Visual Studio bundled vcpkg, which a Native Tools prompt exposes under
`%VSINSTALLDIR%VC\vcpkg`; substitute `C:\vcpkg\vcpkg.exe` (or wherever your
copy lives) if you use a standalone clone. The scratch-directory flags keep
vcpkg's working trees out of the read-only Visual Studio install location:

```bat
cd C:\randblas-work\vcpkg-manifest
"%VSINSTALLDIR%VC\vcpkg\vcpkg.exe" install ^
  --triplet x64-windows ^
  --x-install-root=C:\randblas-work\vcpkg-installed ^
  --downloads-root=C:\randblas-work\vcpkg-scratch\downloads ^
  --x-buildtrees-root=C:\randblas-work\vcpkg-scratch\buildtrees ^
  --x-packages-root=C:\randblas-work\vcpkg-scratch\packages

set "MKLROOT=C:\randblas-work\vcpkg-installed\x64-windows"
set "PATH=%MKLROOT%\bin;%PATH%"
```

After the install succeeds, `C:\randblas-work\vcpkg-scratch` can be deleted;
the installed prefix is self-contained.

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

Random123 is header-only. Copy its public headers into a stable installation
prefix so the installed RandBLAS package does not depend on retaining the
Random123 source checkout:

```bat
git clone https://github.com/DEShawResearch/Random123.git ^
  C:\randblas-work\src\Random123

cmake -E make_directory C:/randblas-work/install/Random123/include
cmake -E copy_directory ^
  C:/randblas-work/src/Random123/include/Random123 ^
  C:/randblas-work/install/Random123/include/Random123
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
selects `/openmp:llvm` under MSVC. The classic `/openmp` mode implements only
OpenMP 2.0, which rejects the `omp simd` directive the sparse kernels use, as
well as 64-bit loop indices and the `collapse` clause downstream consumers
such as RandLAPACK rely on. No OpenMP flag needs to be added manually; to
choose a different mode, set `-DOpenMP_CXX_FLAGS=...` at configure time.

OpenMP is optional. To request a serial build explicitly, add
`-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE` to the RandBLAS configuration
command in the next section.

### A.3. Building, installing, and testing RandBLAS

Clone RandBLAS, configure it against the installed BLAS++ and the Random123
headers, and enable the test suite:

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
  -DRandom123_DIR=C:/randblas-work/install/Random123/include ^
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

The installed `RandBLASConfig.cmake` records the BLAS++ and Random123 locations
used to build RandBLAS, so ordinary consumers should not need to supply those
paths again.

Repository-owned tests copy imported dependency DLLs beside their executables.
An arbitrary downstream application is responsible for its own deployment.
When running the example consumer above from the build tree, ensure that the
oneMKL and BLAS++ runtime directories are on `PATH`, or copy the required DLLs
beside the executable:

```bat
set "PATH=C:\randblas-work\vcpkg-installed\x64-windows\bin;C:\randblas-work\install\blaspp\bin;%PATH%"

C:\path\to\my_randblas_project-build\myexec.exe
```


## Appendix B. Tested configurations, integer width, and packaging

### B.1. What we test

Every row below is a lane in CI, so this table is a statement about what is
actually exercised on every commit rather than what we believe should work.
Anything not listed may well work; it is simply untested.

| OS | Compiler | BLAS backend | Integer width | OpenMP | Notes |
|---|---|---|---|---|---|
| Ubuntu (latest) | gcc | OpenBLAS | LP64 | yes | release, debug+ASan, release+UBSan |
| Ubuntu (latest) | gcc | oneMKL | ILP64 | yes | enables the MKL sparse path |
| Ubuntu (latest) | clang | OpenBLAS | LP64 | yes | release, ASan, TSan |
| macOS 14 | Apple Clang | Accelerate (new interface) | ILP64 | **no** | Apple Clang ships no OpenMP runtime |
| macOS 15 | Homebrew LLVM | Accelerate (new interface) | ILP64 | yes | via Homebrew `libomp` |
| macOS (latest) | Apple Clang | Accelerate (new interface) | LP64 | **no** | installer lane, explicit `--blas-int=lp64` |
| Windows | MSVC | oneMKL | ILP64 | yes (`/openmp:llvm`) | x64 only |

Compiler floor: RandBLAS uses C++20 [concepts](https://en.cppreference.com/w/cpp/language/constraints),
which in practice means **gcc ≥ 13**. Older gcc will not compile it. CMake
3.21 or later is required on every platform.

The installer lanes additionally cover a fresh install, an idempotent re-run,
dependency discovery, and a build performed with plain CMake and no network.

### B.2. Integer width: which BLAS you get, and why

RandBLAS's own API is `int64_t` regardless of the BLAS underneath, because
BLAS++ presents `int64_t` either way. The width of the *underlying* BLAS still
matters in two places: an LP64 BLAS caps each individual matrix dimension at
2³¹, and the MKL sparse path requires `MKL_INT` to match RandBLAS's `int64_t`
sparse indices.

The installer therefore **prefers ILP64 wherever the backend can genuinely
provide it, and falls back to LP64 with a warning where it cannot**:

| Backend | Width | Why |
|---|---|---|
| oneMKL | ILP64 | `mkl_intel_ilp64` is a distinct library, so the choice is real and verifiable |
| OpenBLAS | LP64 | see below |
| Accelerate | ILP64 | Apple's new interface (macOS ≥ 13.3) ships ILP64 inside the framework; BLAS++ supports it as of [blaspp#134](https://github.com/icl-utk-edu/blaspp/pull/134). On older macOS the `int64` probe fails and `auto` falls back to LP64 with a warning |

**OpenBLAS is the subtle one.** BLAS++ probes `int32` before `int64` and uses
`blas_int` only to filter library *names*. For MKL that is enough. For
OpenBLAS there is only ever `-lopenblas`, so an LP64 build passes the `int32`
probe and is accepted — a successful `blas_int=int64` configure proves
nothing. If you have an ILP64 OpenBLAS (on Debian or Ubuntu,
`libopenblas64-dev`), point at it explicitly rather than hoping it is found:

```bash
bash installers/install.sh --blas=custom --blas-int=ilp64 \
  --blas-libraries=/usr/lib/x86_64-linux-gnu/libopenblas64.so
```

The installer reports the width it actually built, read back from BLAS++'s
generated `blas/defines.h` rather than from what was requested, and the CMake
configuration summary reports the same.

### B.3. Packaging with conda-forge or Spack

Neither ecosystem runs install scripts. Both configure with CMake against
dependencies they installed themselves, often with no network available. That
path is tested on every commit by the `linux-plain-cmake-offline` lane, which
configures inside a network namespace with no interfaces.

What a recipe needs to know:

- **Dependencies are `blaspp` and `Random123`.** LAPACK++ is needed only for
  `examples/`, GoogleTest only for the test suite.
- **Nothing is downloaded during configure.** `examples/` is a standalone
  `project()` and is the only place using `FetchContent`, so a packager never
  reaches it.
- **Default to LP64.** conda-forge's `libblas` metapackage — the mechanism
  that lets a user swap BLAS implementations at runtime — is LP64, and this is
  RandBLAS's default for every backend except MKL, so the two agree.
- **RandBLAS never selects a BLAS itself.** It reaches the BLAS only through
  BLAS++ and never calls `find_package(BLAS)`, so `BLA_VENDOR` and the choice
  of implementation stay entirely with the packager.
- **Pass `-DBUILD_TESTS=OFF`** unless you are running the suite; it defaults
  to ON, and without GoogleTest that silently produces a build with zero tests.
  The configuration summary warns when this happens.
- **The installed package is relocatable.** It records the dependency paths
  used at build time, but CMake falls back to a normal `CMAKE_PREFIX_PATH`
  search when those paths do not exist, so a package built in one prefix and
  consumed from another resolves correctly.
