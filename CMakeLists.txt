cmake_minimum_required(VERSION 3.11)

project(RandBLAS)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

# configure the build
enable_testing()
include(rb_build_options)
include(rb_version)

# find dependencies
find_package(blaspp REQUIRED)
find_package(Random123 REQUIRED)
#include(Random123)
include(OpenMP)

# compile sources
add_subdirectory(RandBLAS)
add_subdirectory(test)

# export the configuration
include(rb_config)
