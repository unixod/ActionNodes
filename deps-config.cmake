include_guard()

include(FetchContent)

###########################################
# Fetch dependencies
###########################################
FetchContent_Declare(ez-support
    GIT_REPOSITORY git@github.com:unixod/ez-support.git
    GIT_TAG 090ec3b07569c87e3647698b4d0fcc67e73ffa18
    GIT_SHALLOW On
)

FetchContent_Declare(ez-utils
    GIT_REPOSITORY git@github.com:unixod/ez-utils.git
    GIT_TAG e98680bf7dd22a7596663e6a7b294ffde16e872d
    GIT_SHALLOW On
)

FetchContent_MakeAvailable(ez-support ez-utils)

