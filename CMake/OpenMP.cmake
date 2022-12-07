message(STATUS "Checking for OpenMP ... ")
find_package(OpenMP COMPONENTS CXX)

set(tmp FALSE)
if (OpenMP_CXX_FOUND)
    set(tmp TRUE)
endif()

set(RandBLAS_HAS_OpenMP ${tmp} CACHE BOOL "Set if we have a working OpenMP")
message(STATUS "Checking for OpenMP ... ${RandBLAS_HAS_OpenMP}")
