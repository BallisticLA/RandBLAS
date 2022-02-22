cmake_minimum_required(VERSION 3.2)
project(rblas_tests)

add_subdirectory(lib/googletest)

include_directories(${RBLAS_HEADERS_DIR})
include_directories(lib/googletest/googletest/include)

add_executable(
    rblas_tests
    main.cc
)
target_link_libraries(
    rblas_tests
    rblas
    gtest
)
install(TARGETS rblas_tests DESTINATION bin)
