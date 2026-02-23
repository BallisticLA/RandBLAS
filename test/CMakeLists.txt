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

    set(DENSEDATA_SOURCES
        test_datastructures/test_denseskop.cc
        test_datastructures/test_sparseskop.cc

        test_matmul_cores/test_lskge3.cc
        test_matmul_cores/test_rskge3.cc
        test_matmul_cores/test_lskges.cc
        test_matmul_cores/test_rskges.cc

        test_matmul_wrappers/test_sketch_vector.cc
        test_matmul_wrappers/test_sketch_symmetric.cc
    )
    add_executable(densedata_tests ${DENSEDATA_SOURCES})
    target_link_libraries(densedata_tests RandBLAS GTest::GTest GTest::Main)
    gtest_discover_tests(densedata_tests)

    #####################################################################
    #
    #   Tests with sparse data
    #
    #####################################################################

    set(SPARSEDATA_SOURCES
        test_datastructures/test_spmats/test_csc.cc
        test_datastructures/test_spmats/test_csr.cc
        test_datastructures/test_spmats/test_coo.cc
        test_datastructures/test_spmats/test_conversions.cc
        test_datastructures/test_spmats/test_random_matrix.cc

        test_matmul_cores/test_spmm/test_spmm_csc.cc
        test_matmul_cores/test_spmm/test_spmm_csr.cc
        test_matmul_cores/test_spmm/test_spmm_coo.cc
        test_matmul_cores/test_spgemm/test_spgemm.cc
        test_sparse_trsm/test_sparse_trsm.cc

        test_matmul_wrappers/test_sketch_sparse.cc
    )
    add_executable(sparsedata_tests ${SPARSEDATA_SOURCES})
    target_link_libraries(sparsedata_tests RandBLAS GTest::GTest GTest::Main)
    gtest_discover_tests(sparsedata_tests)

    #####################################################################
    #
    #   Statistical tests
    #
    #####################################################################

    set(STAT_SOURCES
        test_basic_rng/test_r123.cc
        test_basic_rng/test_discrete.cc
        test_basic_rng/test_continuous.cc
        test_basic_rng/test_distortion.cc
    )
    add_executable(stat_tests ${STAT_SOURCES})
    target_link_libraries(stat_tests RandBLAS GTest::GTest GTest::Main)
    gtest_discover_tests(stat_tests)
    file(COPY test_basic_rng/r123_kat_vectors.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY test_basic_rng/r123_kat_vectors.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/..)
    file(COPY test_basic_rng/r123_kat_vectors.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../test/)

    #####################################################################
    #
    #   Meta tests (test our testing infrastructure)
    #
    #####################################################################

    set(META_SOURCES test_handrolled_lapack.cc)
    add_executable(meta_tests ${META_SOURCES})
    target_link_libraries(meta_tests RandBLAS GTest::GTest GTest::Main)
    gtest_discover_tests(meta_tests)

    #####################################################################
    #
    #   Literally any other tests that use the Googletest framework
    #
    #####################################################################

    set(MISC_SOURCES test_io.cc test_exceptions.cc)
    add_executable(misc_tests ${MISC_SOURCES})
    target_link_libraries(misc_tests RandBLAS GTest::GTest GTest::Main)
    gtest_discover_tests(misc_tests)

    #####################################################################
    #
    #   Combined executable for code coverage tools
    #
    #####################################################################

    add_executable(all_correctness_tests
        ${DENSEDATA_SOURCES}
        ${SPARSEDATA_SOURCES}
        ${MISC_SOURCES}
    )
    target_link_libraries(all_correctness_tests RandBLAS GTest::GTest GTest::Main)

endif()
message(STATUS "Checking for regression tests ... ${tmp}")

add_executable(test_rng_speed test_basic_rng/benchmark_speed.cc)
target_link_libraries(test_rng_speed RandBLAS)
