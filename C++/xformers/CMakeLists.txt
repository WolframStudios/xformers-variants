# xformers library CMakeLists.txt

# ── Collect sources ─────────────────────────────────────────────────────────
file(GLOB XFORMERS_CXX_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

file(GLOB XFORMERS_CUDA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

# ── Subdirectories ──────────────────────────────────────────────────────────
add_subdirectory(ops)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sparse/CMakeLists.txt")
    add_subdirectory(sparse)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/profiler/CMakeLists.txt")
    add_subdirectory(profiler)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CMakeLists.txt")
    add_subdirectory(benchmarks)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/flash_attn_3/CMakeLists.txt")
    add_subdirectory(flash_attn_3)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/components/CMakeLists.txt")
    add_subdirectory(components)
endif()

# ── Build xformers core library ─────────────────────────────────────────────
if(XFORMERS_CXX_SOURCES OR XFORMERS_CUDA_SOURCES)
    add_library(xformers
        ${XFORMERS_CXX_SOURCES}
        ${XFORMERS_CUDA_SOURCES}
    )

    set_target_properties(xformers PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    target_include_directories(xformers PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
    )

    target_link_libraries(xformers PUBLIC
        ${TORCH_LIBRARIES}
        ${CUDA_LIBRARIES}
        cuda
    )
endif()
