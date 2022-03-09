configure_file(rblas_config.h.in rblas_config.h)
install(FILES ${CMAKE_BINARY_DIR}/rblas_config.h DESTINATION include/rblas)

configure_file(CMake/rblasConfig.cmake.in
    ${CMAKE_INSTALL_LIBDIR}/cmake/rblasConfig.cmake @ONLY)

configure_file(CMake/rblasConfigVersion.cmake.in
    ${CMAKE_INSTALL_LIBDIR}/cmake/rblasConfigVersion.cmake @ONLY)

install(FILES
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/rblasConfig.cmake
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/rblasConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake)
