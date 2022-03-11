message(STATUS "Checking for OpenMP ... ")
find_package(OpenMP)

set(tmp FALSE)
if (OpenMP_CXX_FOUND)

    set(tmp TRUE)

    # interface libarary for use elsewhere in the project
    add_library(RandBLAS_OpenMP INTERFACE)

    target_include_directories(RandBLAS_OpenMP
        SYSTEM INTERFACE "${OpenMP_CXX_INCLUDE_DIRS}")

    target_link_libraries(RandBLAS_OpenMP INTERFACE ${OpenMP_CXX_LIBRARIES})

    install(TARGETS RandBLAS_OpenMP EXPORT RandBLAS_OpenMP)

    install(EXPORT RandBLAS_OpenMP
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
        EXPORT_LINK_INTERFACE_LIBRARIES)

endif()
set(RandBLAS_HAS_OpenMP ${tmp} CACHE BOOL "Set if we have a working OpenMP")
message(STATUS "Checking for OpenMP ... ${RandBLAS_HAS_OpenMP}")
