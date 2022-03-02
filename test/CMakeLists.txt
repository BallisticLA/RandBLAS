message(STATUS "Checking for regression tests ...")
find_package(GTest)
set(tmp FALSE)
if (GTest_FOUND)
    set(tmp TRUE)

    add_executable(rblas_tests src/test_sjlts.cc)
    target_link_libraries(rblas_tests rblas GTest::GTest GTest::Main)
    gtest_discover_tests(rblas_tests)

endif()
message(STATUS "Checking for regression tests ... ${tmp}")
