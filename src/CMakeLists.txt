set(rblas_cxx_srcs
    sjlts.cc
    util.cc
    determiter.cc)

set(rblas_libs rblas_blaspp rblas_Random123)

if (RBLAS_HAS_OpenMP)
    list(APPEND rblas_libs rblas_OpenMP)
endif()

add_library(rblas ${rblas_cxx_srcs})

target_include_directories(rblas PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/rblas>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:include/rblas>)

target_link_libraries(rblas PUBLIC ${rblas_libs})

install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/"
  DESTINATION include FILES_MATCHING PATTERN "*.hh")

install(TARGETS rblas EXPORT rblas
  INCLUDES DESTINATION include/rblas ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(EXPORT rblas DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
  EXPORT_LINK_INTERFACE_LIBRARIES)
