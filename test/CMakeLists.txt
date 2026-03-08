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
        datastructures/test_denseskop.cc
        datastructures/test_sparseskop.cc

        linops/test_lskge3.cc
        linops/test_rskge3.cc
        linops/test_lskges.cc
        linops/test_rskges.cc

        linops/test_sketch_vector.cc
        linops/test_sketch_symmetric.cc
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
        datastructures/test_csc_matrix.cc
        datastructures/test_csr_matrix.cc
        datastructures/test_coo_matrix.cc
        datastructures/test_spmat_conversions.cc

        linops/test_spgemm.cc
        linops/test_sparse_trsm.cc
        linops/test_spmm/test_spmm_csc.cc
        linops/test_spmm/test_spmm_csr.cc
        linops/test_spmm/test_spmm_coo.cc

        linops/test_sketch_sparse.cc
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
        basic_rng/test_r123.cc
        basic_rng/test_discrete.cc
        basic_rng/test_continuous.cc
        basic_rng/test_distortion.cc
    )
    add_executable(stat_tests ${STAT_SOURCES})
    target_link_libraries(stat_tests RandBLAS GTest::GTest GTest::Main)
    gtest_discover_tests(stat_tests)
    file(COPY basic_rng/r123_kat_vectors.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY basic_rng/r123_kat_vectors.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/..)
    file(COPY basic_rng/r123_kat_vectors.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../test/)

    #####################################################################
    #
    #   Meta tests (test our testing infrastructure)
    #
    #####################################################################

    set(META_SOURCES meta/test_lapack_like.cc meta/test_sparse_data_generators.cc meta/test_comparison.cc)
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

add_executable(test_rng_speed basic_rng/benchmark_speed.cc)
target_link_libraries(test_rng_speed RandBLAS)
