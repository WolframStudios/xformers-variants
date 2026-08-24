cmake_minimum_required(VERSION 3.18)
project(xformers_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(GTest REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(XFORMERS_COMMON_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

set(XFORMERS_COMMON_LINK_LIBS
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    torch
    torch_cuda
    cuda
)

include_directories(${XFORMERS_COMMON_INCLUDE_DIRS})

file(GLOB_RECURSE XFORMERS_PORT_SUPPORT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/.github/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/docs/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/stubs/*.cpp"
)

if(XFORMERS_PORT_SUPPORT_SOURCES)
    add_library(xformers_port_support STATIC
        ${XFORMERS_PORT_SUPPORT_SOURCES}
    )

    set_target_properties(xformers_port_support PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )

    target_include_directories(xformers_port_support PUBLIC
        ${XFORMERS_COMMON_INCLUDE_DIRS}
    )

    target_link_libraries(xformers_port_support PUBLIC
        ${XFORMERS_COMMON_LINK_LIBS}
    )
endif()

add_subdirectory(xformers)
add_subdirectory(tests)
