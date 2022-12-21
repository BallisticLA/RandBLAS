message(STATUS "Checking for regression tests ...")
project(randblas_test)
find_package(GTest)
set(tmp FALSE)
if (GTest_FOUND)
    set(tmp TRUE)

    add_executable(RandBLAS_tests
        test_sparse.cc
        test_dense.cc
    )
    target_link_libraries(RandBLAS_tests
        RandBLAS
        GTest::GTest
        GTest::Main
    )
    gtest_discover_tests(RandBLAS_tests)

endif()
message(STATUS "Checking for regression tests ... ${tmp}")

add_executable(test_rng_speed test_rng_speed.cpp)
target_link_libraries(test_rng_speed Random123)
