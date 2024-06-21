message(STATUS "Checking for regression tests ...")
project(randblas_test)
find_package(GTest)
set(tmp FALSE)
if (GTest_FOUND)
    set(tmp TRUE)

    add_executable(RandBLAS_tests
        linop_common.hh
        comparison.hh
        test_skge_wrappers/test_sketch_vector.cc
        # dense sketching operators for dense data matrices
        test_skops/dense/test_construction.cc
        test_skops/dense/test_lskge3.cc
        test_skops/dense/test_rskge3.cc
        # sparse sketching operators for dense data matrices
        test_skops/sparse/test_construction.cc
        test_skops/sparse/test_lskges.cc
        test_skops/sparse/test_rskges.cc
    )
    target_link_libraries(RandBLAS_tests
        RandBLAS
        GTest::GTest
        GTest::Main
    )
    gtest_discover_tests(RandBLAS_tests)

    add_executable(SparseRandBLAS_tests
        # dense sketching operators for sparse matrices
        comparison.hh
        test_sparse_data/common.hh
        test_sparse_data/test_left_multiply.hh
        test_sparse_data/test_right_multiply.hh
        test_sparse_data/test_csc.cc
        test_sparse_data/test_csr.cc
        test_sparse_data/test_coo.cc
        test_sparse_data/test_conversions.cc
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
