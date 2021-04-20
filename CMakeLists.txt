cmake_minimum_required(VERSION 3.13)

project(depgraph VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Common compile options.
if (MSVC)
    add_compile_options(
        /EHsc
        /W4
        /WX
    )
else()
    add_compile_options(
        -Wall
        -Wcast-align
        -Wconversion
        -Wctor-dtor-privacy
        -Werror
        -Wextra
        -Wno-unknown-attributes
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Woverloaded-virtual
        -Wpedantic
        -Wshadow
        -Wsign-conversion
    )
endif()

add_executable(depgraph)

target_sources(depgraph
    PRIVATE
    depgraph.cpp)

find_package(Threads REQUIRED)
target_link_libraries(depgraph
    PRIVATE
    Threads::Threads)




