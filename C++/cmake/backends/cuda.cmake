# cmake/backends/cuda.cmake
# ── NVIDIA CUDA backend ───────────────────────────────────────────────────────
# Enables CUDA language, sets architectures, and exposes XFORMERS_CUDA_LIBS.

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# sm_70 Volta | sm_75 Turing | sm_80/sm_86 Ampere | sm_89 Ada | sm_90 Hopper
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

find_package(CUDA REQUIRED)

set(XFORMERS_CUDA_LIBS
    ${CUDA_LIBRARIES}
    cuda
    cublas
    cublasLt
)

include_directories(${CUDA_INCLUDE_DIRS})

message(STATUS "[backend] CUDA ${CUDA_VERSION}, archs: ${CMAKE_CUDA_ARCHITECTURES}")
