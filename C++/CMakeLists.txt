cmake_minimum_required(VERSION 3.18)
project(xformers_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/backends")

option(XFORMERS_USE_CUDA "Enable NVIDIA CUDA backend" ON)
option(XFORMERS_USE_ROCM "Enable AMD ROCm/HIP backend" OFF)
option(XFORMERS_USE_VULKAN "Enable Vulkan backend" OFF)
option(XFORMERS_USE_OPENCL "Enable OpenCL backend" OFF)
option(XFORMERS_USE_OPENVINO "Enable OpenVINO backend" OFF)
option(XFORMERS_USE_IPEX "Enable Intel Extension for PyTorch backend" OFF)
option(XFORMERS_USE_OPENGL "Enable OpenGL backend" OFF)
option(XFORMERS_BUILD_TESTS "Build the C++ test suite" ON)

set(XFORMERS_BACKEND_DEFINES "")

if(XFORMERS_USE_CUDA)
    list(APPEND XFORMERS_BACKEND_DEFINES XFORMERS_USE_CUDA)
endif()
if(XFORMERS_USE_ROCM)
    list(APPEND XFORMERS_BACKEND_DEFINES XFORMERS_USE_ROCM)
endif()
if(XFORMERS_USE_VULKAN)
    list(APPEND XFORMERS_BACKEND_DEFINES XFORMERS_USE_VULKAN)
endif()
if(XFORMERS_USE_OPENCL)
    list(APPEND XFORMERS_BACKEND_DEFINES XFORMERS_USE_OPENCL)
endif()
if(XFORMERS_USE_OPENVINO)
    list(APPEND XFORMERS_BACKEND_DEFINES XFORMERS_USE_OPENVINO)
endif()
if(XFORMERS_USE_IPEX)
    list(APPEND XFORMERS_BACKEND_DEFINES XFORMERS_USE_IPEX)
endif()
if(XFORMERS_USE_OPENGL)
    list(APPEND XFORMERS_BACKEND_DEFINES XFORMERS_USE_OPENGL)
endif()

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
if(XFORMERS_BUILD_TESTS)
    find_package(GTest REQUIRED)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(XFORMERS_COMMON_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

set(XFORMERS_COMMON_LINK_LIBS
    ${TORCH_LIBRARIES}
    torch
)

if(XFORMERS_USE_CUDA)
    include(backends/cuda)
    list(APPEND XFORMERS_COMMON_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
    list(APPEND XFORMERS_COMMON_LINK_LIBS ${XFORMERS_CUDA_LIBS} torch_cuda)
endif()

if(XFORMERS_USE_ROCM)
    include(backends/rocm)
    list(APPEND XFORMERS_COMMON_LINK_LIBS ${XFORMERS_ROCM_LIBS})
endif()

if(XFORMERS_USE_VULKAN)
    include(backends/vulkan)
    list(APPEND XFORMERS_COMMON_LINK_LIBS ${XFORMERS_VULKAN_LIBS})
endif()

if(XFORMERS_USE_OPENCL)
    include(backends/opencl)
    list(APPEND XFORMERS_COMMON_LINK_LIBS ${XFORMERS_OPENCL_LIBS})
endif()

if(XFORMERS_USE_OPENVINO)
    include(backends/openvino)
    list(APPEND XFORMERS_COMMON_LINK_LIBS ${XFORMERS_OPENVINO_LIBS})
endif()

if(XFORMERS_USE_IPEX)
    include(backends/ipex)
    list(APPEND XFORMERS_COMMON_LINK_LIBS ${XFORMERS_IPEX_LIBS})
endif()

if(XFORMERS_USE_OPENGL)
    include(backends/opengl)
    list(APPEND XFORMERS_COMMON_LINK_LIBS ${XFORMERS_OPENGL_LIBS})
endif()

if(NOT XFORMERS_BACKEND_DEFINES)
    message(WARNING "No backend selected. Backend-guarded translation units will compile into stubbed placeholders only.")
endif()

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

    target_compile_definitions(xformers_port_support PUBLIC
        ${XFORMERS_BACKEND_DEFINES}
    )

    target_link_libraries(xformers_port_support PUBLIC
        ${XFORMERS_COMMON_LINK_LIBS}
    )
endif()

add_subdirectory(xformers)
if(XFORMERS_BUILD_TESTS)
    add_subdirectory(tests)
endif()
