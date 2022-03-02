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
add_library(rblas_blaspp INTERFACE)

target_link_libraries(rblas_blaspp INTERFACE blaspp)

install(TARGETS rblas_blaspp EXPORT rblas_blaspp)

install(EXPORT rblas_blaspp
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
    EXPORT_LINK_INTERFACE_LIBRARIES)
