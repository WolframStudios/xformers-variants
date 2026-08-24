# xformers/ops CMakeLists.txt

# ── Collect sources ─────────────────────────────────────────────────────────
file(GLOB OPS_CXX_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

file(GLOB OPS_CUDA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

# ── CUDA kernels sub-library (_triton → CUDA) ────────────────────────────────
file(GLOB TRITON_CUDA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/_triton/*.cu"
)

# ── fmha sub-library ─────────────────────────────────────────────────────────
file(GLOB FMHA_CXX_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/fmha/*.cpp"
)

file(GLOB FMHA_CUDA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/fmha/*.cu"
)

# ── Build ops library ────────────────────────────────────────────────────────
set(ALL_OPS_SOURCES
    ${OPS_CXX_SOURCES}
    ${OPS_CUDA_SOURCES}
    ${TRITON_CUDA_SOURCES}
    ${FMHA_CXX_SOURCES}
    ${FMHA_CUDA_SOURCES}
)

if(ALL_OPS_SOURCES)
    add_library(xformers_ops
        ${ALL_OPS_SOURCES}
    )

    set_target_properties(xformers_ops PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    target_include_directories(xformers_ops PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${TORCH_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
    )

    target_link_libraries(xformers_ops PUBLIC
        ${TORCH_LIBRARIES}
        ${CUDA_LIBRARIES}
        cuda
    )
endif()
