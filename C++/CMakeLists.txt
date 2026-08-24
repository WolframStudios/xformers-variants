cmake_minimum_required(VERSION 3.18)
project(xformers_cpp LANGUAGES CXX CUDA)

# ── C++ / CUDA Standards ────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── CUDA Architectures ──────────────────────────────────────────────────────
# Volta (sm_70), Turing (sm_75), Ampere (sm_80/sm_86), Ada (sm_89), Hopper (sm_90)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# ── Find Dependencies ───────────────────────────────────────────────────────
find_package(CUDA REQUIRED)

# LibTorch – set TORCH_HOME or pass -DCMAKE_PREFIX_PATH=/path/to/libtorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Google Test (optional – required for tests target)
find_package(GTest QUIET)

# ── Global Include Directories ──────────────────────────────────────────────
include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ── Subdirectories ──────────────────────────────────────────────────────────
add_subdirectory(xformers)

if(GTest_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()
