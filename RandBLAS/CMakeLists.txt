
set(RandBLAS_libs blaspp Random123)
if (RandBLAS_HAS_OpenMP)
    list(APPEND RandBLAS_libs OpenMP::OpenMP_CXX)
endif()

# MKL sparse BLAS: libraries are already linked through blaspp,
# we only need the include path for mkl_spblas.h.
if (RandBLAS_HAS_MKL)
    set(RandBLAS_MKL_INCLUDE_DIR ${MKL_SPARSE_INCLUDE_DIR})
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in config.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/config.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/RandBLAS)

add_library(RandBLAS INTERFACE)

target_include_directories(RandBLAS INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/RandBLAS>)

if (RandBLAS_HAS_MKL)
    target_include_directories(RandBLAS INTERFACE ${RandBLAS_MKL_INCLUDE_DIR})
endif()

target_link_libraries(RandBLAS INTERFACE ${RandBLAS_libs})

set(RandBLAS_cxx_opts -Wall -Wextra -Wno-comment)
if (SANITIZE_ADDRESS)
    list(APPEND RandBLAS_cxx_opts -fsanitize=address)
    target_link_options(RandBLAS INTERFACE -fsanitize=address)
endif()

target_compile_options(RandBLAS INTERFACE ${RandBLAS_cxx_opts})

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../RandBLAS.hh"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DESTINATION include FILES_MATCHING PATTERN "*.hh")

install(TARGETS RandBLAS EXPORT RandBLAS
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/RandBLAS
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(EXPORT RandBLAS DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
  EXPORT_LINK_INTERFACE_LIBRARIES)
