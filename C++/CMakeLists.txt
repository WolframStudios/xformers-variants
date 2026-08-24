cmake_minimum_required(VERSION 3.18)
project(xformers_cpp LANGUAGES CXX)

# ── C++ Standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── CMake module path ─────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/backends")

# ── Backend options (all OFF by default – select what you need) ───────────────
option(XFORMERS_USE_CUDA     "Enable NVIDIA CUDA backend"          OFF)
option(XFORMERS_USE_ROCM     "Enable AMD ROCm/HIP backend"         OFF)
option(XFORMERS_USE_VULKAN   "Enable Vulkan compute backend"       OFF)
option(XFORMERS_USE_OPENVINO "Enable Intel OpenVINO backend"       OFF)
option(XFORMERS_USE_IPEX     "Enable Intel Extension for PyTorch"  OFF)
option(XFORMERS_USE_OPENML   "Enable OpenML backend"              OFF)
option(XFORMERS_USE_OPENGL   "Enable OpenGL compute backend"       OFF)
option(XFORMERS_BUILD_TESTS  "Build test suite"                    ON)

# ── Accumulate backend compile definitions ────────────────────────────────────
set(XFORMERS_BACKEND_DEFINES "")

# ── CUDA ──────────────────────────────────────────────────────────────────────
if(XFORMERS_USE_CUDA)
    include(backends/cuda)
    list(APPEND XFORMERS_BACKEND_DEFINES XFORMERS_USE_CUDA)
endif()

# ── ROCm / HIP ────────────────────────────────────────────────────────────────
if(XFORMERS_USE_ROCM)
    include(backends/rocm)
    list(APPEND XFORMERS_BACKEND_DEFINES XFORMERS_USE_ROCM)
endif()

# ── Vulkan ────────────────────────────────────────────────────────────────────
if(XFORMERS_USE_VULKAN)
    include(backends/vulkan)
    list(APPEND XFORMERS_BACKEND_DEFINES XFORMERS_USE_VULKAN)
endif()

# ── OpenVINO ──────────────────────────────────────────────────────────────────
if(XFORMERS_USE_OPENVINO)
    include(backends/openvino)
    list(APPEND XFORMERS_BACKEND_DEFINES XFORMERS_USE_OPENVINO)
endif()

# ── IPEX (Intel Extension for PyTorch) ────────────────────────────────────────
if(XFORMERS_USE_IPEX)
    include(backends/ipex)
    list(APPEND XFORMERS_BACKEND_DEFINES XFORMERS_USE_IPEX)
endif()

# ── OpenML
if(XFORMERS_USE_OPENML)
    include(backends/openml)
    list(APPEND XFORMERS_BACKEND_DEFINES XFORMERS_USE_OPENML)
endif()

# ── OpenGL ────────────────────────────────────────────────────────────────────
if(XFORMERS_USE_OPENGL)
    include(backends/opengl)
    list(APPEND XFORMERS_BACKEND_DEFINES XFORMERS_USE_OPENGL)
endif()

# ── Warn if no backend selected ───────────────────────────────────────────────
if(NOT XFORMERS_BACKEND_DEFINES)
    message(WARNING
        "No backend selected. Pass at least one of: "
        "-DXFORMERS_USE_CUDA=ON -DXFORMERS_USE_ROCM=ON "
        "-DXFORMERS_USE_VULKAN=ON -DXFORMERS_USE_OPENVINO=ON "
        "-DXFORMERS_USE_IPEX=ON -DXFORMERS_USE_OPENGL=ON")
endif()

# ── LibTorch (required by all backends) ───────────────────────────────────────
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ── Google Test ───────────────────────────────────────────────────────────────
if(XFORMERS_BUILD_TESTS)
    find_package(GTest QUIET)
endif()

# ── Global includes ───────────────────────────────────────────────────────────
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

# ── Subdirectories ────────────────────────────────────────────────────────────
add_subdirectory(xformers)

if(XFORMERS_BUILD_TESTS AND GTest_FOUND AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()

# ── Configuration summary ─────────────────────────────────────────────────────
message(STATUS "=== xformers build configuration ===")
message(STATUS "  CUDA     : ${XFORMERS_USE_CUDA}")
message(STATUS "  ROCm     : ${XFORMERS_USE_ROCM}")
message(STATUS "  Vulkan   : ${XFORMERS_USE_VULKAN}")
message(STATUS "  OpenVINO : ${XFORMERS_USE_OPENVINO}")
message(STATUS "  IPEX     : ${XFORMERS_USE_IPEX}")
message(STATUS "  OpenML   : ${XFORMERS_USE_OPENML}")
message(STATUS "  OpenGL   : ${XFORMERS_USE_OPENGL}")
message(STATUS "  Tests    : ${XFORMERS_BUILD_TESTS}")
message(STATUS "=====================================")
