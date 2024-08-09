message(STATUS "Checking for regression tests ...")
project(randblas_test)
find_package(GTest)
set(tmp FALSE)
if (GTest_FOUND)
    set(tmp TRUE)

    #####################################################################
    #
    #   Tests with dense data and wrappers
    #
    #####################################################################

    add_executable(RandBLAS_tests
        test_datastructures/test_denseskop.cc
        test_datastructures/test_sparseskop.cc

        test_matmul_cores/test_lskge3.cc
        test_matmul_cores/test_rskge3.cc
        test_matmul_cores/test_lskges.cc
        test_matmul_cores/test_rskges.cc

        test_matmul_wrappers/test_sketch_vector.cc
        test_matmul_wrappers/test_sketch_symmetric.cc
    )
    target_link_libraries(RandBLAS_tests RandBLAS GTest::GTest GTest::Main)
    gtest_discover_tests(RandBLAS_tests)

    #####################################################################
    #
    #   Tests with sparse data
    #
    #####################################################################

    add_executable(SparseRandBLAS_tests
        test_datastructures/test_spmats/test_csc.cc
        test_datastructures/test_spmats/test_csr.cc
        test_datastructures/test_spmats/test_coo.cc
        test_datastructures/test_spmats/test_conversions.cc

        test_matmul_cores/test_spmm/test_spmm_csc.cc
        test_matmul_cores/test_spmm/test_spmm_csr.cc
        test_matmul_cores/test_spmm/test_spmm_coo.cc
    )
    target_link_libraries(SparseRandBLAS_tests RandBLAS GTest::GTest GTest::Main)
    gtest_discover_tests(SparseRandBLAS_tests)

    #####################################################################
    #
    #   Statistical tests
    #
    #####################################################################

    add_executable(RandBLAS_stats
        test_basic_rng/test_r123.cc
        test_basic_rng/test_discrete.cc
        test_basic_rng/test_continuous.cc
    )
    target_link_libraries(RandBLAS_stats RandBLAS GTest::GTest GTest::Main)
    gtest_discover_tests(RandBLAS_stats)
    file(COPY test_basic_rng/r123_kat_vectors.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../test/)
    file(COPY test_basic_rng/r123_kat_vectors.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

    #####################################################################
    #
    #   Meta tests (test our testing infrastructure)
    #
    #####################################################################

    add_executable(MetaRandBLAS_tests
        test_handrolled_lapack.cc
    )
    target_link_libraries(MetaRandBLAS_tests RandBLAS GTest::GTest GTest::Main)
    gtest_discover_tests(MetaRandBLAS_tests)

endif()
message(STATUS "Checking for regression tests ... ${tmp}")

add_executable(test_rng_speed test_basic_rng/benchmark_speed.cc)
target_link_libraries(test_rng_speed RandBLAS)
