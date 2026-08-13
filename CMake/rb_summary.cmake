# Configuration summary, printed at the end of configure.
#
# RandBLAS has several features that switch themselves off when a dependency
# is missing rather than failing the configure step. That is the right
# behaviour -- a missing GoogleTest or MKL header should not stop a build that
# does not need them -- but it means a degraded configuration and a complete
# one look identical unless someone reads every STATUS line. This module
# collects those decisions in one block and says, in words, what is on, what
# is off, and what to do about anything that is off by accident.
#
# Included last from the top-level CMakeLists.txt so that every decision,
# including the test subdirectory's, has already been made.

# Read a compile definition out of the blaspp package. blasppConfig.cmake sets
# the blaspp_defines variable; older or differently packaged builds only carry
# them on the target. Check both, exactly as CMake/MKL_sparse.cmake does.
function(_rb_blaspp_has define out_var)
    if (DEFINED blaspp_defines AND blaspp_defines MATCHES "${define}")
        set(${out_var} TRUE PARENT_SCOPE)
        return()
    endif()
    if (TARGET blaspp)
        get_target_property(_defs blaspp INTERFACE_COMPILE_DEFINITIONS)
        if (_defs AND _defs MATCHES "${define}")
            set(${out_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()
    set(${out_var} FALSE PARENT_SCOPE)
endfunction()

_rb_blaspp_has("BLAS_ILP64"    _rb_blaspp_ilp64)
_rb_blaspp_has("BLAS_HAVE_MKL" _rb_blaspp_mkl)

# The integer width is a property of the BLAS that blaspp was built against,
# not of RandBLAS -- RandBLAS's own API is int64_t either way. It still belongs
# here, because mixing an ILP64 blaspp with an LP64 BLAS (or the reverse) is
# the single most common way to get a build that links and then misbehaves.
if (_rb_blaspp_ilp64)
    set(_rb_int_width "ILP64 (64-bit BLAS integers)")
else()
    set(_rb_int_width "LP64 (32-bit BLAS integers)")
endif()

if (_rb_blaspp_mkl)
    set(_rb_blaspp_backend "MKL")
else()
    set(_rb_blaspp_backend "non-MKL")
endif()

# Collect anything the user probably did not intend, to repeat under the block.
#
# Each note must be a single quoted string containing NO semicolon: CMake lists
# are semicolon-separated, so both a second list(APPEND) argument and a
# semicolon inside the text become separate elements, and the note is then
# emitted as several truncated warnings instead of one. Use string(CONCAT) to
# build long notes, and comma-splice rather than semicolon-splice the prose.
set(_rb_notes "")

if (RandBLAS_HAS_OpenMP)
    set(_rb_openmp "enabled")
    if (MSVC)
        set(_rb_openmp "${_rb_openmp} (${OpenMP_CXX_FLAGS})")
    endif()
else()
    set(_rb_openmp "disabled")
    if (APPLE)
        # Apple Clang genuinely ships no OpenMP runtime, so this one is
        # expected rather than a misconfiguration.
        set(_rb_openmp "${_rb_openmp} (Apple Clang has no OpenMP runtime)")
    else()
        list(APPEND _rb_notes
            "OpenMP is disabled, so RandBLAS runs single-threaded.")
    endif()
endif()

if (RandBLAS_HAS_MKL)
    set(_rb_mkl_sparse "enabled")
else()
    set(_rb_mkl_sparse "disabled")
    # Only the "requested but broken" case is worth nagging about. Being
    # deliberately off, or having a non-MKL BLAS, is a normal configuration.
    if (RandBLAS_MKL_SPARSE_REASON MATCHES "not found")
        string(CONCAT _rb_note
            "MKL sparse BLAS was requested but mkl_spblas.h was not found, so "
            "sparse kernels fall back to the portable implementations. "
            "Set MKLROOT, or pass -DRandBLAS_USE_MKL_SPARSE=OFF to make this deliberate.")
        list(APPEND _rb_notes "${_rb_note}")
    endif()
endif()
if (RandBLAS_MKL_SPARSE_REASON)
    set(_rb_mkl_sparse "${RandBLAS_MKL_SPARSE_REASON}")
endif()

if (NOT BUILD_TESTS)
    set(_rb_tests "not built (BUILD_TESTS=OFF)")
elseif (RandBLAS_HAS_GTest)
    set(_rb_tests "built")
else()
    set(_rb_tests "requested, but GoogleTest was not found -- none built")
    string(CONCAT _rb_note
        "BUILD_TESTS is ON but GoogleTest was not found, so no tests were built "
        "and ctest will report zero tests. Point CMake at GoogleTest with "
        "-DGTest_ROOT=<prefix>, or pass -DBUILD_TESTS=OFF to make skipping them "
        "explicit.")
    list(APPEND _rb_notes "${_rb_note}")
endif()

message(STATUS "")
message(STATUS "---- RandBLAS configuration summary ----")
message(STATUS "  version            ${RandBLAS_VERSION_MAJOR}.${RandBLAS_VERSION_MINOR}.${RandBLAS_VERSION_PATCH} (${RandBLAS_COMMIT_HASH})")
message(STATUS "  build type         ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard       ${CMAKE_CXX_STANDARD}")
message(STATUS "  install prefix     ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  BLAS++             ${_rb_blaspp_backend}, ${_rb_int_width}")
message(STATUS "  OpenMP             ${_rb_openmp}")
message(STATUS "  MKL sparse BLAS    ${_rb_mkl_sparse}")
message(STATUS "  tests              ${_rb_tests}")
message(STATUS "  address sanitizer  ${SANITIZE_ADDRESS}")
message(STATUS "----------------------------------------")

# RandBLAS is an INTERFACE (header-only) target, so BUILD_SHARED_LIBS has no
# effect on it. It is still an advertised option, and it does affect anything
# built alongside, so say so rather than letting it look ignored.
if (BUILD_SHARED_LIBS)
    message(STATUS
        "  note: RandBLAS is header-only, so BUILD_SHARED_LIBS does not change "
        "what is installed.")
endif()

foreach (_rb_note IN LISTS _rb_notes)
    message(WARNING "${_rb_note}")
endforeach()

message(STATUS "")
