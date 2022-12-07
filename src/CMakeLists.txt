set(RandBLAS_cxx_srcs
    sparse_op.cc
    util.cc
    dense_op.cc)

set(RandBLAS_libs RandBLAS_blaspp RandBLAS_Random123)

if (RandBLAS_HAS_OpenMP)
    list(APPEND RandBLAS_libs RandBLAS_OpenMP)
endif()

add_library(RandBLAS ${RandBLAS_cxx_srcs})

target_include_directories(RandBLAS PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/RandBLAS>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:include/RandBLAS>)

target_link_libraries(RandBLAS PUBLIC ${RandBLAS_libs})

install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/"
  DESTINATION include FILES_MATCHING PATTERN "*.hh")

install(TARGETS RandBLAS EXPORT RandBLAS
  INCLUDES DESTINATION include/RandBLAS ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(EXPORT RandBLAS DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
  EXPORT_LINK_INTERFACE_LIBRARIES)
