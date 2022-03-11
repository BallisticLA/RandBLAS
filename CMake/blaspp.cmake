message(STATUS "Checking for blaspp ... ")
find_package(blaspp REQUIRED)
message(STATUS "Checking for blaspp ... ${blaspp_VERSION}")

# NOTE: blaspp sets the following variables so we know which GPU's are
# supported:
#
# blaspp_use_openmp
# blaspp_use_cuda
# blaspp_use_hip
#

# interface libarary for use elsewhere in the project
add_library(RandBLAS_blaspp INTERFACE)

target_link_libraries(RandBLAS_blaspp INTERFACE blaspp)

install(TARGETS RandBLAS_blaspp EXPORT RandBLAS_blaspp)

install(EXPORT RandBLAS_blaspp
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
    EXPORT_LINK_INTERFACE_LIBRARIES)
