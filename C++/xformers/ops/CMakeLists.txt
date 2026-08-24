# xformers/ops CMakeLists.txt

file(GLOB_RECURSE XFORMERS_OPS_CXX_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

set(XFORMERS_OPS_SOURCES
    ${XFORMERS_OPS_CXX_SOURCES}
)

if(XFORMERS_USE_CUDA OR XFORMERS_USE_ROCM)
    file(GLOB_RECURSE XFORMERS_OPS_CUDA_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
    )
    list(APPEND XFORMERS_OPS_SOURCES ${XFORMERS_OPS_CUDA_SOURCES})
endif()

if(XFORMERS_OPS_SOURCES)
    add_library(xformers_ops
        ${XFORMERS_OPS_SOURCES}
    )

    set_target_properties(xformers_ops PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    target_include_directories(xformers_ops PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${XFORMERS_COMMON_INCLUDE_DIRS}
    )

    target_compile_definitions(xformers_ops PUBLIC
        ${XFORMERS_BACKEND_DEFINES}
    )

    target_link_libraries(xformers_ops PUBLIC
        ${XFORMERS_COMMON_LINK_LIBS}
    )
endif()
