##############################################################################
# Setup action::utils target
##############################################################################
add_library(anodes-utils INTERFACE)
add_library(anodes::utils ALIAS anodes-utils)

target_include_directories(anodes-utils
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include>)

find_package(Threads REQUIRED)

target_link_libraries(anodes-utils
    INTERFACE
    Threads::Threads
    anodes::compile-options)





