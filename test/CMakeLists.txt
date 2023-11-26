message(STATUS "Checking for regression tests ...")
project(randblas_test)
find_package(GTest)
set(tmp FALSE)
if (GTest_FOUND)
    set(tmp TRUE)

    add_executable(RandBLAS_tests
        # dense sketching operators for dense data matrices
        test_dense/test_construction.cc
        test_dense/test_sketch_gefl3.cc
        test_dense/test_sketch_gefr3.cc
        # sparse sketching operators for dense data matrices
        test_sparse_skops/test_construction.cc
        test_sparse_skops/test_sketch_gefls.cc
        test_sparse_skops/test_sketch_gefrs.cc
    )
    target_link_libraries(RandBLAS_tests
        RandBLAS
        GTest::GTest
        GTest::Main
    )
    gtest_discover_tests(RandBLAS_tests)

    add_executable(SparseRandBLAS_tests
        # dense sketching operators for sparse matrices
        test_sparse_data/common.hh
        test_sparse_data/test_csr.cc
    )
    target_link_libraries(SparseRandBLAS_tests
        RandBLAS
        GTest::GTest
        GTest::Main
    )
    gtest_discover_tests(SparseRandBLAS_tests)

endif()
message(STATUS "Checking for regression tests ... ${tmp}")

add_executable(test_rng_speed test_rng_speed.cc)
target_link_libraries(test_rng_speed RandBLAS)
