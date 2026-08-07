message(STATUS "Checking for OpenMP ... ")

# MSVC's classic /openmp mode implements OpenMP 2.0 only: it rejects the
# omp simd directive used by the sparse kernels, 64-bit loop indices, and
# the collapse clause used by downstream consumers such as RandLAPACK. The
# /openmp:llvm runtime supports all of these (and subsumes what
# /openmp:experimental offered). Callers can still override the mode
# explicitly with -DOpenMP_CXX_FLAGS=... at configure time.
if (MSVC AND NOT DEFINED OpenMP_CXX_FLAGS)
    set(OpenMP_CXX_FLAGS "/openmp:llvm" CACHE STRING
        "OpenMP compiler flags for C++")
endif()
if (MSVC)
    set(RandBLAS_OpenMP_MSVC_FLAGS "${OpenMP_CXX_FLAGS}")
endif()

find_package(OpenMP COMPONENTS CXX)

# FindOpenMP may replace OpenMP_CXX_FLAGS while probing the compiler. Ensure
# the imported target used by RandBLAS carries the requested MSVC mode.
if (MSVC AND OpenMP_CXX_FOUND AND TARGET OpenMP::OpenMP_CXX)
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY
        INTERFACE_COMPILE_OPTIONS "${RandBLAS_OpenMP_MSVC_FLAGS}")
    set(OpenMP_CXX_FLAGS "${RandBLAS_OpenMP_MSVC_FLAGS}")
endif()

set(tmp FALSE)
if (OpenMP_CXX_FOUND)
    set(tmp TRUE)
endif()

set(RandBLAS_HAS_OpenMP ${tmp} CACHE BOOL "Set if we have a working OpenMP")
message(STATUS "Checking for OpenMP ... ${RandBLAS_HAS_OpenMP}")
