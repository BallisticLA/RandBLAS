cmake_minimum_required(VERSION 3.2)

project(RandBLAS)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

# configure the build
enable_testing()
include(build_options)
include(version)

# find dependencies
include(blaspp)
include(Random123)
include(OpenMP)

# compile sources
add_subdirectory(src)
add_subdirectory(test)

# export the configuration
include(config)
