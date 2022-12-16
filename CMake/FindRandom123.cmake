if (NOT Random123_FOUND)

# find the header
# first look where the user told us
if(Random123_DIR)
    find_path(Random123_INCLUDE_DIR Random123/philox.h
        PATHS "${Random123_DIR}" "${Random123_DIR}/include/"
        NO_DEFAULT_PATH)
endif()

# look in typical system locations
find_path(Random123_INCLUDE_DIR Random123/philox.h
  PATHS "/usr/include/" "/usr/local/include/"
  NO_DEFAULT_PATH)

# finally let CMake look
find_path(Random123_INCLUDE_DIR Random123/philox.h)

mark_as_advanced(Random123_INCLUDE_DIR)

# handle the QUIETLY and REQUIRED arguments and set Random123_FOUND
include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Random123
    "Failed to find Random123. Set -DRandom123_DIR=X with X pointing to the directory where the header files \"Random123/*.h\" are located."
    Random123_INCLUDE_DIR)

if (NOT TARGET Random123)

    # interface libarary for use elsewhere in the project
    add_library(Random123 INTERFACE)

    target_include_directories(Random123
        SYSTEM INTERFACE "${Random123_INCLUDE_DIR}")

    install(TARGETS Random123 EXPORT Random123)

    install(EXPORT Random123
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
        EXPORT_LINK_INTERFACE_LIBRARIES)

    install(FILES CMake/FindRandom123.cmake
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake")

endif()

endif()
