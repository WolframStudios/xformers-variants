# xformers/ops CMakeLists.txt

file(GLOB_RECURSE XFORMERS_OPS_CXX_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
file(GLOB_RECURSE XFORMERS_OPS_CUDA_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

set(XFORMERS_OPS_SOURCES
    ${XFORMERS_OPS_CXX_SOURCES}
    ${XFORMERS_OPS_CUDA_SOURCES}
)

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

    target_link_libraries(xformers_ops PUBLIC
        ${XFORMERS_COMMON_LINK_LIBS}
    )
endif()
