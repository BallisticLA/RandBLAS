message(STATUS "Checking for OpenMP ... ")
find_package(OpenMP)

set(tmp FALSE)
if (OpenMP_CXX_FOUND)

    set(tmp TRUE)

    # interface libarary for use elsewhere in the project
    add_library(rblas_OpenMP INTERFACE)

    target_include_directories(rblas_OpenMP
        SYSTEM INTERFACE "${OpenMP_CXX_INCLUDE_DIRS}")

    target_link_libraries(rblas_OpenMP INTERFACE ${OpenMP_CXX_LIBRARIES})

    install(TARGETS rblas_OpenMP EXPORT rblas_OpenMP)

    install(EXPORT rblas_OpenMP
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
        EXPORT_LINK_INTERFACE_LIBRARIES)

endif()
set(RBLAS_HAS_OpenMP ${tmp} CACHE BOOL "Set if we have a working OpenMP")
message(STATUS "Checking for OpenMP ... ${RBLAS_HAS_OpenMP}")
