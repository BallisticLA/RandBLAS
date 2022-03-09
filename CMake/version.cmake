set(tmp)
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE}
        --git-dir=${CMAKE_SOURCE_DIR}/.git describe --tags
        OUTPUT_VARIABLE tmp OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
endif()
if(NOT tmp)
    set(tmp "0.0.0")
endif()
set(RBLAS_VERSION ${tmp} CACHE STRING "RBLAS version" FORCE)

string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*$)"
  "\\1" RBLAS_VERSION_MAJOR ${RBLAS_VERSION})

string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*$)"
  "\\2" RBLAS_VERSION_MINOR ${RBLAS_VERSION})

string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*$)"
  "\\3" RBLAS_VERSION_PATCH ${RBLAS_VERSION})

message(STATUS "RBLAS_VERSION_MAJOR=${RBLAS_VERSION_MAJOR}")
message(STATUS "RBLAS_VERSION_MINOR=${RBLAS_VERSION_MINOR}")
message(STATUS "RBLAS_VERSION_PATCH=${RBLAS_VERSION_PATCH}")
