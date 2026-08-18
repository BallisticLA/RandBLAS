# Matches the floor in the top-level RandBLAS CMakeLists.txt; RuntimeDLLs.cmake,
# included below, needs TARGET_RUNTIME_DLLS from 3.21 on native Windows.
cmake_minimum_required(VERSION 3.21)

project(examples)

include("${CMAKE_CURRENT_LIST_DIR}/../CMake/RuntimeDLLs.cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "default build type" FORCE)
endif()

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

# Keep the download inside the project tree by default rather than buried in
# the build directory, so a second configure reuses it and a user can find it.
# Overridable, and ignored entirely if the caller already set it.
if (NOT FETCHCONTENT_BASE_DIR)
    set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps"
        CACHE PATH "Where FetchContent stores downloaded dependencies")
endif()

# Pinned to a release tag. This was previously GIT_TAG main, which meant the
# examples built against whatever that branch happened to hold on the day --
# so two people running the same commit could get different code, and a
# breaking change upstream would land here without any change on our side.
# v1.7.6 is commit d8e50e9a58918fef6ff2674cd62abfaf7b9c42a2.
FetchContent_Declare(
    fast_matrix_market
    GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
    GIT_TAG v1.7.6
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(
    fast_matrix_market
)

add_executable(
    slra_svd_fmm sparse-low-rank-approx/svd_matrixmarket.cc
)
target_include_directories(
    slra_svd_fmm PUBLIC ${Random123_DIR}
)
target_link_libraries(
    slra_svd_fmm PUBLIC RandBLAS blaspp lapackpp fast_matrix_market::fast_matrix_market
)

add_executable(
    slra_qrcp sparse-low-rank-approx/qrcp_matrixmarket.cc
)
target_include_directories(
    slra_qrcp PUBLIC ${Random123_DIR}
)
target_link_libraries(
    slra_qrcp PUBLIC RandBLAS blaspp lapackpp fast_matrix_market::fast_matrix_market
)

add_executable(
    spmm_performance simple-kernel-benchmarks/spmm_performance.cc
)
target_include_directories(
    spmm_performance PUBLIC ${Random123_DIR}
)
target_link_libraries(
    spmm_performance PUBLIC RandBLAS blaspp lapackpp
)

add_executable(
    sketch_general_performance simple-kernel-benchmarks/sketch_general_performance.cc
)
target_include_directories(
    sketch_general_performance PUBLIC ${Random123_DIR}
)
target_link_libraries(
    sketch_general_performance PUBLIC RandBLAS blaspp lapackpp
)

add_executable(
    saso_sampling_performance simple-kernel-benchmarks/saso_sampling_performance.cc
)
target_include_directories(
    saso_sampling_performance PUBLIC ${Random123_DIR}
)
target_link_libraries(
    saso_sampling_performance PUBLIC RandBLAS blaspp lapackpp
)

foreach(example_target IN ITEMS
    tls_dense_skop
    tls_sparse_skop
    slra_svd_synthetic
    slra_svd_fmm
    slra_qrcp
    spmm_performance
    sketch_general_performance
    saso_sampling_performance
)
    randblas_stage_runtime_dlls(${example_target})
endforeach()
