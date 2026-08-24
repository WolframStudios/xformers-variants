# cmake/backends/rocm.cmake
# ── AMD ROCm / HIP backend ────────────────────────────────────────────────────
# Supports Linux and Windows (ROCm 5.x+ has experimental Windows support).

# Default ROCm installation paths
if(NOT DEFINED ROCM_PATH)
    if(WIN32)
        set(ROCM_PATH "C:/Program Files/AMD/ROCm/5.7")
    else()
        set(ROCM_PATH "/opt/rocm")
    endif()
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}" "${ROCM_PATH}/hip" "${ROCM_PATH}/lib/cmake")

# Enable HIP language
enable_language(HIP)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# GPU targets – gfx900 Vega10 | gfx906 Vega20 | gfx908 MI100
#               gfx90a MI200 | gfx940/941/942 MI300 | gfx1030/1100 RDNA
if(NOT DEFINED GPU_TARGETS)
    set(GPU_TARGETS "gfx906;gfx908;gfx90a;gfx1030;gfx1100" CACHE STRING "AMD GPU targets")
endif()
set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})

find_package(hip REQUIRED HINTS "${ROCM_PATH}/hip/lib/cmake/hip")
find_package(rocblas QUIET HINTS "${ROCM_PATH}/lib/cmake/rocblas")
find_package(rocthrust QUIET HINTS "${ROCM_PATH}/lib/cmake/rocthrust")
find_package(hipblaslt QUIET HINTS "${ROCM_PATH}/lib/cmake/hipblaslt")

set(XFORMERS_ROCM_LIBS hip::host)
if(rocblas_FOUND)
    list(APPEND XFORMERS_ROCM_LIBS roc::rocblas)
endif()
if(hipblaslt_FOUND)
    list(APPEND XFORMERS_ROCM_LIBS roc::hipblaslt)
endif()

include_directories(
    "${ROCM_PATH}/include"
    "${ROCM_PATH}/hip/include"
)

message(STATUS "[backend] ROCm ${hip_VERSION}, GPU targets: ${GPU_TARGETS}")
