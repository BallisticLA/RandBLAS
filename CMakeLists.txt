cmake_minimum_required(VERSION 3.11)

project(RandBLAS)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

# configure the build
enable_testing()
include(rb_build_options)
include(rb_version)

# Select BLAS backend
set(RandBLAS_BLAS_BACKEND "blaspp" CACHE STRING
    "BLAS backend: 'blaspp' (requires BLAS++) or 'builtin' (RandBLAS built-in)")
set_property(CACHE RandBLAS_BLAS_BACKEND PROPERTY STRINGS blaspp builtin)

# find dependencies
if(RandBLAS_BLAS_BACKEND STREQUAL "blaspp")
    find_package(blaspp REQUIRED)
    set(RandBLAS_HAS_BLASPP TRUE CACHE BOOL "Set if using BLAS++ backend" FORCE)
    message(STATUS "BLAS backend: blaspp")
elseif(RandBLAS_BLAS_BACKEND STREQUAL "builtin")
    set(RandBLAS_HAS_BLASPP FALSE CACHE BOOL "Set if using BLAS++ backend" FORCE)
    message(STATUS "BLAS backend: builtin (RandBLAS built-in)")
else()
    message(FATAL_ERROR "Unknown RandBLAS_BLAS_BACKEND='${RandBLAS_BLAS_BACKEND}'. "
            "Valid values: blaspp, builtin")
endif()

find_package(Random123 REQUIRED)
#include(Random123)
include(OpenMP)
include(MKL_sparse)

# compile sources
add_subdirectory(RandBLAS)
add_subdirectory(test)

# export the configuration
include(rb_config)
