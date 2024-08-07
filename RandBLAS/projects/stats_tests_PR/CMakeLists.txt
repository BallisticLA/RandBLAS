cmake_minimum_required(VERSION 3.10)

project(fisher_test)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3")
# ^ THAT'S SO IMPORTANT!!!!
#   TODO: just set CMAKE_BUILD_TYPE to Release if it's not already given.

message(STATUS "Current CMake module path: ${CMAKE_MODULE_PATH}")

# Try to find OpenMP
message(STATUS "Checking for OpenMP ...")
find_package(OpenMP REQUIRED)
message(STATUS "Found OpenMP: ${OpenMP_FOUND}")

# Set the path to the RandBLAS installation
set(RandBLAS_DIR "/Users/r9w/Coding/Projects/RandNLA/RandBLAS-install/lib/cmake")
message(STATUS "RandBLAS directory set to: ${RandBLAS_DIR}")

message(STATUS "Checking for RandBLAS ... ")
find_package(RandBLAS REQUIRED)
message(STATUS "Done checking for RandBLAS. Found RandBLAS: ${RandBLAS_FOUND}")

message(STATUS "Looking for LAPACK++ ... ")
find_package(lapackpp REQUIRED)
message(STATUS "Done looking for LAPACK++. Found LAPACK++: ${lapackpp_FOUND}")

# Add the executable
add_executable(fisher_test main.cc utils.hh)

# Include directories
target_include_directories(
    fisher_test PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${RandBLAS_INCLUDE_DIRS}
    /Users/r9w/Coding/Projects/RandNLA/RandBLAS/test/test_basic_rng
)
message(STATUS "RandBLAS include directories: ${RandBLAS_INCLUDE_DIRS}")

# Link libraries
target_link_libraries(
    fisher_test PUBLIC RandBLAS blaspp lapackpp OpenMP::OpenMP_CXX
)
message(STATUS "Linked libraries for fisher_test: RandBLAS::RandBLAS blaspp lapackpp OpenMP::OpenMP_CXX")

# Print the include directories for verification
get_target_property(include_dirs fisher_test INCLUDE_DIRECTORIES)
message(STATUS "Include directories for fisher_test: ${include_dirs}")

# Print the list of directories CMake can look into
message(STATUS "CMake prefix path: ${CMAKE_PREFIX_PATH}")
message(STATUS "CMake module path: ${CMAKE_MODULE_PATH}")
