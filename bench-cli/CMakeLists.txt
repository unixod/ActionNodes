add_executable(bench-cli)

target_sources(bench-cli
    PRIVATE
    bench-cli.cpp)

target_include_directories(bench-cli
    PRIVATE
    include)

target_link_libraries(bench-cli
    PRIVATE
    ez::support
    ez::utils
    anodes::graph
    anodes::compile-options)
