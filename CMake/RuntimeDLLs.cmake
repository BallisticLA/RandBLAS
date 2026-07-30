# Native Windows builds use TARGET_RUNTIME_DLLS to stage imported shared-library
# dependencies beside executables. This generator expression requires CMake 3.21.
if (WIN32 AND CMAKE_VERSION VERSION_LESS 3.21)
    message(FATAL_ERROR "RandBLAS native Windows builds require CMake 3.21 or later.")
endif()

function(randblas_stage_runtime_dlls target)
    if (WIN32)
        add_custom_command(
            TARGET ${target}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:${target}>
                    $<TARGET_FILE_DIR:${target}>
            COMMAND_EXPAND_LISTS
            VERBATIM
        )
    endif()
endfunction()
