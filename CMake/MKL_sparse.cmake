option(RandBLAS_USE_MKL_SPARSE
    "Enable MKL sparse BLAS features (spgemm, accelerated spmm). \
Requires BLAS++ built with MKL. Set to OFF to disable MKL sparse \
features while still using MKL through BLAS++." ON)

message(STATUS "Checking for MKL sparse BLAS ...")

if (NOT RandBLAS_USE_MKL_SPARSE)
    message(STATUS "  Disabled by user (RandBLAS_USE_MKL_SPARSE=OFF)")
    set(RandBLAS_HAS_MKL FALSE CACHE BOOL "Set if MKL sparse BLAS is available" FORCE)
    message(STATUS "Checking for MKL sparse BLAS ... ${RandBLAS_HAS_MKL}")
    return()
endif()

# Check if blaspp was built with MKL.
# blasppConfig.cmake exports blaspp_defines as a string variable containing
# compile definitions like "-DBLAS_HAVE_MKL". We check that variable.
# Also check the target's INTERFACE_COMPILE_DEFINITIONS as a fallback.
set(tmp FALSE)

# Method 1: Check the blaspp_defines variable (set by blasppConfig.cmake)
if (DEFINED blaspp_defines AND blaspp_defines MATCHES "BLAS_HAVE_MKL")
    set(_mkl_found_via_blaspp TRUE)
else()
    # Method 2: Check the target property (in case blaspp exports it differently)
    get_target_property(_blaspp_defs blaspp INTERFACE_COMPILE_DEFINITIONS)
    if (_blaspp_defs AND _blaspp_defs MATCHES "BLAS_HAVE_MKL")
        set(_mkl_found_via_blaspp TRUE)
    else()
        set(_mkl_found_via_blaspp FALSE)
    endif()
endif()

if (_mkl_found_via_blaspp)
    # BLAS++ was built with MKL. Find the sparse BLAS header.
    find_path(MKL_SPARSE_INCLUDE_DIR mkl_spblas.h
        HINTS ENV MKLROOT $ENV{MKLROOT}/include
        PATH_SUFFIXES include)
    if (MKL_SPARSE_INCLUDE_DIR)
        set(tmp TRUE)
        message(STATUS "  MKL sparse BLAS header found: ${MKL_SPARSE_INCLUDE_DIR}/mkl_spblas.h")
    else()
        message(STATUS "  blaspp was built with MKL, but mkl_spblas.h not found. Set MKLROOT.")
    endif()
endif()

set(RandBLAS_HAS_MKL ${tmp} CACHE BOOL "Set if MKL sparse BLAS is available" FORCE)
message(STATUS "Checking for MKL sparse BLAS ... ${RandBLAS_HAS_MKL}")
