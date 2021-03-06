message(STATUS "Checking for Random123 ... ")
find_package(Random123 REQUIRED)

# interface libarary for use elsewhere in the project
add_library(RandBLAS_Random123 INTERFACE)

target_include_directories(RandBLAS_Random123
    SYSTEM INTERFACE "${Random123_INCLUDE_DIR}")

install(TARGETS RandBLAS_Random123 EXPORT RandBLAS_Random123)

install(EXPORT RandBLAS_Random123
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
    EXPORT_LINK_INTERFACE_LIBRARIES)

install(FILES CMake/FindRandom123.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake")
