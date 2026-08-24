# xformers library CMakeLists.txt

file(GLOB XFORMERS_CORE_CXX_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
file(GLOB XFORMERS_CORE_CUDA_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)
file(GLOB_RECURSE XFORMERS_SUPPORT_CXX_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/components/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/flash_attn_3/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/profiler/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sparse/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/triton/*.cpp"
)
file(GLOB_RECURSE XFORMERS_SUPPORT_CUDA_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/components/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/flash_attn_3/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/profiler/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/sparse/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/triton/*.cu"
)

add_subdirectory(ops)

set(XFORMERS_LIBRARY_SOURCES
    ${XFORMERS_CORE_CXX_SOURCES}
    ${XFORMERS_CORE_CUDA_SOURCES}
    ${XFORMERS_SUPPORT_CXX_SOURCES}
    ${XFORMERS_SUPPORT_CUDA_SOURCES}
)

if(XFORMERS_LIBRARY_SOURCES)
    add_library(xformers
        ${XFORMERS_LIBRARY_SOURCES}
    )

    set_target_properties(xformers PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    target_include_directories(xformers PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${XFORMERS_COMMON_INCLUDE_DIRS}
    )

    target_link_libraries(xformers PUBLIC
        xformers_ops
        ${XFORMERS_COMMON_LINK_LIBS}
    )
endif()
