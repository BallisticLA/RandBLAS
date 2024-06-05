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
    set(tmp "0.0.0-0-gunknown")
endif()
set(RandBLAS_FULL_VERSION ${tmp} CACHE STRING "RandBLAS version" FORCE)

string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)-(.*$)"
  "\\1" RandBLAS_VERSION_MAJOR ${RandBLAS_FULL_VERSION})

string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)-(.*$)"
  "\\2" RandBLAS_VERSION_MINOR ${RandBLAS_FULL_VERSION})

string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)-(.*$)"
  "\\3" RandBLAS_VERSION_PATCH ${RandBLAS_FULL_VERSION})

string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)-(.*$)"
  "\\4" RandBLAS_VERSION_CHANGES ${RandBLAS_FULL_VERSION})

string(REGEX REPLACE "^([0-9]+)-g([a-zA-Z0-9]+)" "\\1" RandBLAS_COMMITS_SINCE_RELEASE ${RandBLAS_VERSION_CHANGES})
string(REGEX REPLACE "^([0-9]+)-g([a-zA-Z0-9]+)" "\\2" RandBLAS_COMMIT_HASH ${RandBLAS_VERSION_CHANGES})


message(STATUS " ")
message(STATUS "RandBLAS version information")
message(STATUS "  The nominal version number is ${RandBLAS_VERSION_MAJOR}.${RandBLAS_VERSION_MINOR}.${RandBLAS_VERSION_PATCH}.")
message(STATUS "  The commit hash for the current RandBLAS source code is ${RandBLAS_COMMIT_HASH}.")
message(STATUS "  There have been ${RandBLAS_COMMITS_SINCE_RELEASE} commits since the nominal version was assigned.")
message(STATUS " ")
