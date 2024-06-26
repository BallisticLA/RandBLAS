message(STATUS "Checking for regression tests ...")
project(randblas_test)
find_package(GTest)
set(tmp FALSE)
if (GTest_FOUND)
    set(tmp TRUE)

    add_executable(RandBLAS_tests
        comparison.hh
        test_datastructures/test_denseskop.cc
        test_datastructures/test_sparseskop.cc

        test_matmul_cores/linop_common.hh
        test_matmul_cores/test_lskge3.cc
        test_matmul_cores/test_rskge3.cc
        test_matmul_cores/test_lskges.cc
        test_matmul_cores/test_rskges.cc

        test_matmul_wrappers/test_sketch_vector.cc
    )
    target_link_libraries(RandBLAS_tests
        RandBLAS
        GTest::GTest
        GTest::Main
    )
    gtest_discover_tests(RandBLAS_tests)

    add_executable(SparseRandBLAS_tests
        comparison.hh

        test_datastructures/test_spmats/common.hh
        test_datastructures/test_spmats/test_csc.cc
        test_datastructures/test_spmats/test_csr.cc
        test_datastructures/test_spmats/test_coo.cc
        test_datastructures/test_spmats/test_conversions.cc

        test_matmul_cores/test_spmm/spmm_test_helpers.hh
        test_matmul_cores/test_spmm/test_spmm_csc.cc
        test_matmul_cores/test_spmm/test_spmm_csr.cc
        test_matmul_cores/test_spmm/test_spmm_coo.cc
    )
    target_link_libraries(SparseRandBLAS_tests
        RandBLAS
        GTest::GTest
        GTest::Main
    )
    gtest_discover_tests(SparseRandBLAS_tests)

endif()
message(STATUS "Checking for regression tests ... ${tmp}")

add_executable(test_rng_speed test_basic_rng/benchmark_speed.cc)
target_link_libraries(test_rng_speed RandBLAS)
