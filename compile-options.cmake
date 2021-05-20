##################################################
# Define anodes::compile_options - a target
# representing a set of common compile options
# for a project.
##################################################

include_guard()

add_library(anodes-compile-options INTERFACE)
add_library(anodes::compile-options ALIAS anodes-compile-options)

if(NOT ACTION_NODES_STANDALONE)
    # use client defined options.
elseif(MSVC)
    target_compile_options(anodes-compile-options
        INTERFACE
        /EHsc
        /Wall
        /WX
        /Zc:__cplusplus
    )
else()
    target_compile_options(anodes-compile-options
        INTERFACE
        -Wall
        -Wcast-align
        -Wconversion
        -Werror
        -Wextra
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Woverloaded-virtual
        -Wpedantic
        -Wshadow
        -Wsign-conversion
    )
endif()

target_compile_features(anodes-compile-options INTERFACE cxx_std_20)
