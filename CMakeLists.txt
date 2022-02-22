cmake_minimum_required(VERSION 3.2)

project(rblas)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -g")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
set(BLASPP_DIR /opt/slate/blaspp)

find_library(BLASPP blaspp ${BLASPP_DIR}/lib)

message(WARNING ${BLASPP})

set(RBLAS_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(RBLAS_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)
set(RBLAS_HEADERS_DIR ${PROJECT_SOURCE_DIR}/src/)

set(BLASPP_HEADERS_DIR ${BLASPP_DIR}/include)
set(RANDOM123_DIR /opt/random123/include)

include_directories(${RBLAS_INSTALL_INCLUDE_DIR})
include_directories(${RBLAS_HEADERS_DIR})
include_directories(${BLASPP_HEADERS_DIR})
include_directories(${RANDOM123_DIR})

add_library(
    rblas
    ${BLASPP_HEADERS_DIR}/blas.hh
    ${RANDOM123_DIR}/Random123/philox.h
    ${RANDOM123_DIR}/Random123/uniform.hpp
    src/sparseops.cc
    src/determiter.cc 
    src/rblasutil.cc
)
target_link_libraries(
    rblas
    PUBLIC
    ${BLASPP}
)

install(TARGETS rblas DESTINATION ${RBLAS_INSTALL_LIB_DIR})

add_subdirectory(test)
