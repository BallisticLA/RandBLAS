message(STATUS "Checking for OpenMP ... ")

# MSVC's ordinary /openmp mode does not support the omp simd directive used by
# the sparse kernels. Recent MSVC versions provide it through
# /openmp:experimental. Allow callers to override this setting explicitly.
if (MSVC AND NOT DEFINED OpenMP_CXX_FLAGS)
    set(OpenMP_CXX_FLAGS "/openmp:experimental" CACHE STRING
        "OpenMP compiler flags for C++")
endif()

find_package(OpenMP COMPONENTS CXX)

# FindOpenMP may replace OpenMP_CXX_FLAGS while probing the compiler. Ensure
# the imported target used by RandBLAS carries the SIMD-capable MSVC option.
if (MSVC AND OpenMP_CXX_FOUND AND TARGET OpenMP::OpenMP_CXX)
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY
        INTERFACE_COMPILE_OPTIONS "/openmp:experimental")
    set(OpenMP_CXX_FLAGS "/openmp:experimental")
endif()

set(tmp FALSE)
if (OpenMP_CXX_FOUND)
    set(tmp TRUE)
endif()

set(RandBLAS_HAS_OpenMP ${tmp} CACHE BOOL "Set if we have a working OpenMP")
message(STATUS "Checking for OpenMP ... ${RandBLAS_HAS_OpenMP}")
