# 3.21 is the real floor: native Windows builds need the TARGET_RUNTIME_DLLS
# generator expression (see CMake/RuntimeDLLs.cmake), which was added in 3.21.
# Nothing in the project uses a feature newer than that. Declaring one floor
# for every platform keeps the requirement honest -- the previous 3.12 here
# could never actually have built on Windows.
cmake_minimum_required(VERSION 3.21)

project(RandBLAS)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

# configure the build
enable_testing()
include(rb_build_options)
include(rb_version)
include(RuntimeDLLs)

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

# Report what was actually configured. Kept last so it can see every decision
# above, including whether the test subdirectory found GoogleTest.
include(rb_summary)
