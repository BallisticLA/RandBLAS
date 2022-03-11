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
