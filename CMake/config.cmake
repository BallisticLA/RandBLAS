configure_file(RandBLAS_config.h.in RandBLAS_config.h)
install(FILES ${CMAKE_BINARY_DIR}/RandBLAS_config.h DESTINATION include/RandBLAS)

configure_file(CMake/RandBLASConfig.cmake.in
    ${CMAKE_INSTALL_LIBDIR}/cmake/RandBLASConfig.cmake @ONLY)

configure_file(CMake/RandBLASConfigVersion.cmake.in
    ${CMAKE_INSTALL_LIBDIR}/cmake/RandBLASConfigVersion.cmake @ONLY)

install(FILES
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/RandBLASConfig.cmake
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/RandBLASConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake)
