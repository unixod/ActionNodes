##############################################################################
# Setup action-nodes::action-nodes target
##############################################################################
add_library(anodes-graph INTERFACE)
add_library(anodes::graph ALIAS anodes-graph)

target_include_directories(anodes-graph
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include>)

find_package(Threads REQUIRED)

target_link_libraries(anodes-graph
    INTERFACE
    Threads::Threads
    anodes::compile-options)

##############################################################################
# Setup unit tests
##############################################################################
#if(ACTION_NODES_STANDALONE)
#    include(CTest)

#    if (BUILD_TESTING)
#        include(tests-config.cmake)
#        add_subdirectory(tests)
#    endif()
#endif()





