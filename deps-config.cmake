include_guard()

include(FetchContent)

###########################################
# Fetch dependencies
###########################################
FetchContent_Declare(ez-support
    GIT_REPOSITORY git@github.com:unixod/ez-support.git
    GIT_TAG 3561b5d3ae027d90f9407c83760993979af4fdd1
    GIT_SHALLOW On
)

FetchContent_Declare(ez-utils
    GIT_REPOSITORY git@github.com:unixod/ez-utils.git
    GIT_TAG 599bd48e55a96ab8573323b6573c30c3ffbcd4c2
    GIT_SHALLOW On
)

FetchContent_MakeAvailable(ez-support ez-utils)

