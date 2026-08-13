# Values substituted into an installed CMake package file must use CMake path
# syntax. In particular, native Windows backslashes would otherwise be parsed
# as escape sequences when a downstream project loads RandBLASConfig.cmake.
file(TO_CMAKE_PATH "${blaspp_DIR}" RandBLAS_CONFIG_BLASPP_DIR)
file(TO_CMAKE_PATH "${Random123_DIR}" RandBLAS_CONFIG_RANDOM123_DIR)

configure_file(CMake/RandBLASConfig.cmake.in
    ${CMAKE_INSTALL_LIBDIR}/cmake/RandBLAS/RandBLASConfig.cmake @ONLY)

configure_file(CMake/RandBLASConfigVersion.cmake.in
    ${CMAKE_INSTALL_LIBDIR}/cmake/RandBLAS/RandBLASConfigVersion.cmake @ONLY)

if (PROJECT_NAME STREQUAL "RandBLAS")
    # RuntimeDLLs.cmake ships with the package because downstream Windows
    # consumers need randblas_stage_runtime_dlls() as much as this project
    # does: on Windows the loader searches the executable's own directory
    # first and PATH last, so an executable linking installed RandBLAS has no
    # way to find the BLAS DLLs unless they are staged beside it. Without
    # this, the function exists only in the build tree and every consumer has
    # to reinvent it.
    install(FILES CMake/FindRandom123.cmake CMake/RuntimeDLLs.cmake
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/RandBLAS")
endif()

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/RandBLAS/RandBLASConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/RandBLAS/RandBLASConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RandBLAS)
