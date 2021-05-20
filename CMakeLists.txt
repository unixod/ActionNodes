cmake_minimum_required(VERSION 3.16)

project(action-nodes
    VERSION 1.0.0
    LANGUAGES CXX
)

include(GNUInstallDirs)

##############################################################################
# Setup project general properties
##############################################################################

# Check if this is a standalone project or included by other project
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(ACTION_NODES_STANDALONE On)
endif()

# Although this library is supposed to be dependency free its unit tests
# are based on Catch2 unit test framework. Specifying CMAKE_CXX_STANDARD
# in the following block is necessary for correct building of and linking
# with Catch2 and potentialy any other fetched dependency because they
# need to be informed somehow about the C++ standard being used.
if(ACTION_NODES_STANDALONE AND NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

option(ACTION_NODES_INSTALL "Enable installation of action-nodes target" ACTION_NODES_STANDALONE)

include(compile-options.cmake)
include(deps-config.cmake)


add_subdirectory(graph)
add_subdirectory(utils)
add_subdirectory(bench-cli)
