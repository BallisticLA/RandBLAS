set(tmp)
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE}
        --git-dir=${CMAKE_CURRENT_SOURCE_DIR}/.git describe
        --tags --match "[0-9]*.[0-9]*.[0-9]*"
        OUTPUT_VARIABLE tmp OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
endif()
if(NOT tmp)
    set(tmp "0.0.0")
endif()
set(RandBLAS_VERSION ${tmp} CACHE STRING "RandBLAS version" FORCE)

string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*$)"
  "\\1" RandBLAS_VERSION_MAJOR ${RandBLAS_VERSION})

string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*$)"
  "\\2" RandBLAS_VERSION_MINOR ${RandBLAS_VERSION})

string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*$)"
  "\\3" RandBLAS_VERSION_PATCH ${RandBLAS_VERSION})

message(STATUS "RandBLAS_VERSION_MAJOR=${RandBLAS_VERSION_MAJOR}")
message(STATUS "RandBLAS_VERSION_MINOR=${RandBLAS_VERSION_MINOR}")
message(STATUS "RandBLAS_VERSION_PATCH=${RandBLAS_VERSION_PATCH}")
