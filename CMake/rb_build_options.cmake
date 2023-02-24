set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_ENABLE_EXPORTS OFF)

# detect OpenMP. Do this first because we need to pass some compiler flags to
# nvcc that CMake leaves out.
set(tmp FALSE)
if (NOT DEFINED ENABLE_OpenMP OR ENABLE_OpenMP)
    message(STATUS "RandBLAS: Checking for OpenMP ... ")
    find_package(OpenMP COMPONENTS CXX)
    if (OpenMP_CXX_FOUND)
        set(tmp TRUE)
    endif()
    message(STATUS "RandBLAS: Checking for OpenMP ... ${tmp}")
endif()
set(RandBLAS_HAS_OpenMP ${tmp} CACHE BOOL "Set if we have a working OpenMP")

# C++. CMake fails to pass correct flags in many instances hence we set them
# our selves. If desired one can averride these on the command line.
if (NOT CMAKE_CXX_FLAGS)
    set(tmp "-fPIC -std=c++${CMAKE_CXX_STANDARD} -Wall -Wextra -fvisibility=hidden")
    if ((APPLE) AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
        set(tmp "${tmp} -stdlib=libc++")
    endif()
    if (RandBLAS_HAS_OpenMP)
        set(tmp "${tmp} -fopenmp")
    endif()

    set(CMAKE_CXX_FLAGS "${tmp}"
        CACHE STRING "RandBLAS compiler flag defaults"
        FORCE)

    string(REGEX REPLACE "-O[0-9]" "-O3" tmp "${CMAKE_CXX_FLAGS_RELEASE}")
    set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -march=native -mtune=native"
        CACHE STRING "RandBLAS compiler optimization flag defaults"
        FORCE)
endif()

# CUDA
set(tmp FALSE)
if (ENABLE_CUDA)
    # use the CMake language features, this is needed for compiling CUDA sources.
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "RandBLAS: CUDA features -- enabled (${HAMR_CUDA_ARCHITECTURES})")
    else()
        message(FATAL_ERROR "RandBLAS: CUDA is required for RandBLAS but was not found")
    endif()
    # use the CMake package, this is needed for linking the required libraries.
    find_package(CUDAToolkit 12.0 REQUIRED)
    # fine tune the compiler flags, CMake currently does not set these
    # correctly on its own. For that reason we set them ourselves. One
    # can override these on the command line.
    if (NOT CMAKE_CUDA_FLAGS)
        set(tmp "--default-stream per-thread --expt-relaxed-constexpr")
        if ("${CMAKE_BUILD_TYPE}" MATCHES "Release")
            set(tmp "${tmp} -Xcompiler -Wall,-Wextra,-O3,-march=native,-mtune=native,-fvisibility=hidden")
        elseif ("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
            set(tmp "${tmp} -g -G -Xcompiler -Wall,-Wextra,-O0,-g,-fvisibility=hidden")
        endif()
        if (RandBLAS_HAS_OpenMP)
            set(tmp "${tmp},-fopenmp")
        endif()

        set(CMAKE_CUDA_FLAGS "${tmp}"
            CACHE STRING "RandBLAS compiler flag defaults"
            FORCE)

        string(REGEX REPLACE "-O[0-9]" "-O3" tmp "${CMAKE_CUDA_FLAGS_RELEASE}")
        set(CMAKE_CUDA_FLAGS_RELEASE "${tmp}"
            CACHE STRING "RandBLAS compiler flag  defaults"
            FORCE)
    endif()
    set(tmp TRUE)
else()
    message(STATUS "RandBLAS: CUDA features -- disabled")
endif()
set(RandBLAS_HAS_CUDA ${tmp} CACHE BOOL "Set when CUDA features are available")
set(RandBLAS_CUDA_ARCHITECTURES "60;75;80" CACHE STRING "Compile for these CUDA archiectures")

option(BUILD_SHARED_LIBS OFF "Configure to build shared or static libraries")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release"
  CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(SANITIZE_ADDRESS OFF CACHE BOOL "Add address sanitizer flags to the library")

include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")

