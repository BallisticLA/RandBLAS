cmake_minimum_required(VERSION 3.10)

project(examples)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3")
# ^ THAT'S SO IMPORTANT!!!!

# TODO: do a try-catch pattern for finding OpenMP.
# If we get an error, then check if (1) we're macOS
# and (2) if we're using default gcc/g++ linked to system clang,
# when we need to use clang from homebrew.
find_package(OpenMP REQUIRED)

message(STATUS "Checking for RandBLAS ... ")
find_package(RandBLAS REQUIRED)
message(STATUS "Done checking for RandBLAS. ...")

message(STATUS "Looking for LAPACK++ ... ")
find_package(lapackpp REQUIRED)
message(STATUS "Done looking for LAPACK++.")


set(
    tls_dense_skop_cxx total-least-squares/tls_dense_skop.cc
)
add_executable(
    tls_dense_skop ${tls_dense_skop_cxx}
)
target_include_directories(
    tls_dense_skop PUBLIC ${Random123_DIR}
)
target_link_libraries(
    tls_dense_skop PUBLIC RandBLAS blaspp lapackpp 
)

set(
    tls_sparse_skop_cxx total-least-squares/tls_sparse_skop.cc
)
add_executable(
    tls_sparse_skop ${tls_sparse_skop_cxx}
)
target_include_directories(
    tls_sparse_skop PUBLIC ${Random123_DIR}
)
target_link_libraries(
    tls_sparse_skop PUBLIC RandBLAS blaspp lapackpp 
)

add_executable(
    slra_svd_synthetic sparse-low-rank-approx/svd_rank1_plus_noise.cc
)
target_include_directories(
    slra_svd_synthetic PUBLIC ${Random123_DIR}
)
target_link_libraries(
    slra_svd_synthetic PUBLIC RandBLAS blaspp lapackpp
)


include(FetchContent)
FetchContent_Declare(
    fast_matrix_market
    GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
    GIT_TAG main
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(
    fast_matrix_market
)

add_executable(
    slra_svd_fmm sparse-low-rank-approx/fmm_svd.cc
)
target_include_directories(
    slra_svd_fmm PUBLIC ${Random123_DIR}
)
target_link_libraries(
    slra_svd_fmm PUBLIC RandBLAS blaspp lapackpp fast_matrix_market::fast_matrix_market
)

add_executable(
    slra_qrcp sparse-low-rank-approx/qrcp.cc
)
target_include_directories(
    slra_qrcp PUBLIC ${Random123_DIR}
)
target_link_libraries(
    slra_qrcp PUBLIC RandBLAS blaspp lapackpp fast_matrix_market::fast_matrix_market
)
