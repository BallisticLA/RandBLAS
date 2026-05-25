cmake_minimum_required(VERSION 3.12)

project(RandBLAS)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

# configure the build
enable_testing()
include(rb_build_options)
include(rb_version)

# find dependencies
find_package(blaspp REQUIRED)
find_package(Random123 REQUIRED)
include(OpenMP)
include(MKL_sparse)

# compile sources
add_subdirectory(RandBLAS)
if (BUILD_TESTS)
    add_subdirectory(test)
endif()

# export the configuration
include(rb_config)
