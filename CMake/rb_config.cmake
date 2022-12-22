configure_file(CMake/RandBLASConfig.cmake.in
    ${CMAKE_INSTALL_LIBDIR}/cmake/RandBLASConfig.cmake @ONLY)

configure_file(CMake/RandBLASConfigVersion.cmake.in
    ${CMAKE_INSTALL_LIBDIR}/cmake/RandBLASConfigVersion.cmake @ONLY)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/RandBLASConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/RandBLASConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake)
