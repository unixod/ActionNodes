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
    GIT_TAG eed2c476a4dbf8b53e3d83dd16974b58a3c98490
    GIT_SHALLOW On
)

FetchContent_MakeAvailable(ez-support ez-utils)

