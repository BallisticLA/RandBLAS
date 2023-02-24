message(STATUS "Checking for regression tests ...")
project(randblas_test)
find_package(GTest)
set(tmp FALSE)
if (GTest_FOUND)
    set(tmp TRUE)

    set(RandBLAS_test_sources
        test_sparse.cc
        test_dense.cc
    )

    if (RandBLAS_HAS_CUDA)
        set_source_files_properties(${RandBLAS_test_sources} PROPERTIES LANGUAGE CUDA)
    endif()

    add_executable(RandBLAS_tests
        test_sparse.cc
        test_dense.cc
    )

    if (RandBLAS_HAS_CUDA)
        set_target_properties(RandBLAS_tests PROPERTIES CUDA_ARCHITECTURES "${RandBLAS_CUDA_ARCHITECTURES}")
    endif()

    target_link_libraries(RandBLAS_tests
        RandBLAS
        GTest::GTest
        GTest::Main
    )
    gtest_discover_tests(RandBLAS_tests)

endif()
message(STATUS "Checking for regression tests ... ${tmp}")


if (RandBLAS_HAS_CUDA)
    set_source_files_properties(test_rng_speed.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_executable(test_rng_speed test_rng_speed.cpp)
if (RandBLAS_HAS_CUDA)
    set_target_properties(test_rng_speed PROPERTIES CUDA_ARCHITECTURES "${RandBLAS_CUDA_ARCHITECTURES}")
endif()


target_link_libraries(test_rng_speed RandBLAS)
