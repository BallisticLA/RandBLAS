configure_file(CMake/RandBLASConfig.cmake.in
    ${CMAKE_INSTALL_LIBDIR}/cmake/RandBLAS/RandBLASConfig.cmake @ONLY)

configure_file(CMake/RandBLASConfigVersion.cmake.in
    ${CMAKE_INSTALL_LIBDIR}/cmake/RandBLAS/RandBLASConfigVersion.cmake @ONLY)

if (PROJECT_NAME STREQUAL "RandBLAS")
    install(TARGETS Random123 EXPORT Random123)
    install(EXPORT Random123
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/RandBLAS"
        EXPORT_LINK_INTERFACE_LIBRARIES)
    install(FILES CMake/FindRandom123.cmake
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/RandBLAS")
endif()

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/RandBLAS/RandBLASConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/RandBLAS/RandBLASConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RandBLAS)
