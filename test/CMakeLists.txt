message(STATUS "Checking for regression tests ...")

project(randblas_test)

find_package(GTest)
set(tmp FALSE)
if (GTest_FOUND)
    set(tmp TRUE)

    add_executable(RandBLAS_tests
        src/rbtutil.cc
        src/test_sasos.cc
        src/test_dense_op.cc
    )
    target_include_directories(RandBLAS_tests PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/test/include>
    )
    target_link_libraries(RandBLAS_tests
        RandBLAS
        GTest::GTest
        GTest::Main
    )
    gtest_discover_tests(RandBLAS_tests)

endif()
message(STATUS "Checking for regression tests ... ${tmp}")
