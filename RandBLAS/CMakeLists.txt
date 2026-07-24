
set(RandBLAS_libs blaspp Random123::Random123)
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
target_compile_features(RandBLAS INTERFACE cxx_std_20)

# RandBLAS is header-only, so these MSVC requirements must propagate to every
# translation unit that includes its headers. /Zc:__cplusplus makes MSVC report
# the selected language standard correctly (which Random123 inspects), while
# /EHsc enables standard C++ exception-unwind semantics for code using the API.
target_compile_options(RandBLAS INTERFACE
  $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>
  $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/Zc:__cplusplus>)

# Random123's boxmuller.hpp otherwise calls the nonstandard sincos/sincosf
# functions, which are unavailable in the MSVC runtime. Select Random123's
# portable separate-sine-and-cosine fallback for native MSVC consumers.
target_compile_definitions(RandBLAS INTERFACE
  $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:R123_NO_SINCOS=1>)


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

# Build-time only: warning and sanitizer flags should not leak into the installed target.
# Use each compiler family's warning options without passing Unix-style flags
# to MSVC. Keep -Wno-comment for GNU/Clang builds.
target_compile_options(RandBLAS INTERFACE
    $<BUILD_INTERFACE:$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall>>
    $<BUILD_INTERFACE:$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wextra>>
    $<BUILD_INTERFACE:$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wno-comment>>
    $<BUILD_INTERFACE:$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4>>)

if (SANITIZE_ADDRESS)
    if (MSVC)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(
            "/fsanitize=address"
            RANDBLAS_MSVC_ADDRESS_SANITIZER_SUPPORTED)
        if (NOT RANDBLAS_MSVC_ADDRESS_SANITIZER_SUPPORTED)
            message(FATAL_ERROR
                "SANITIZE_ADDRESS=ON requires an MSVC toolchain with "
                "/fsanitize=address support and its AddressSanitizer runtime "
                "libraries installed.")
        endif()
        # MSVC records the required AddressSanitizer runtime in each
        # instrumented object, so /fsanitize=address is a compile option rather
        # than a raw link.exe option.
        target_compile_options(RandBLAS INTERFACE
            $<BUILD_INTERFACE:/fsanitize=address>
            $<BUILD_INTERFACE:/Zi>)
        target_link_options(RandBLAS INTERFACE
            $<BUILD_INTERFACE:/DEBUG>)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(RandBLAS INTERFACE
            $<BUILD_INTERFACE:-fsanitize=address>)
        target_link_options(RandBLAS INTERFACE
            $<BUILD_INTERFACE:-fsanitize=address>)
    else()
        message(FATAL_ERROR
            "SANITIZE_ADDRESS=ON is not supported with "
            "${CMAKE_CXX_COMPILER_ID}.")
    endif()
endif()

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../RandBLAS.hh"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DESTINATION include FILES_MATCHING PATTERN "*.hh")

install(TARGETS RandBLAS EXPORT RandBLAS
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/RandBLAS
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(EXPORT RandBLAS DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RandBLAS)
