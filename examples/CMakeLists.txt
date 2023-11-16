cmake_minimum_required(VERSION 3.10)

project(examples)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "Checking for RandBLAS ... ")
find_package(RandBLAS REQUIRED)
message(STATUS "Done checking for RandBLAS. ...")

message(STATUS "Looking for BLAS++ ... ")
find_package(blaspp REQUIRED)
message(STATUS "Done looking for BLAS++.")


message(STATUS "Looking for LAPACK++ ... ")
find_package(lapackpp REQUIRED)
message(STATUS "Done looking for LAPACK++.")


set(
    TLS_DenseSkOp_cxx
    TLS_DenseSkOp.cpp
)
add_executable(
    TLS_DenseSkOp ${TLS_DenseSkOp_cxx}
)
target_include_directories(
    TLS_DenseSkOp PUBLIC ${Random123_DIR}
)
target_link_libraries(
    TLS_DenseSkOp PUBLIC RandBLAS blaspp lapackpp 
)

set(
    TLS_SparseSkOp_cxx
    TLS_SparseSkOp.cpp
)
add_executable(
    TLS_SparseSkOp ${TLS_SparseSkOp_cxx}
)
target_include_directories(
    TLS_SparseSkOp PUBLIC ${Random123_DIR}
)
target_link_libraries(
    TLS_SparseSkOp PUBLIC RandBLAS blaspp lapackpp 
)

