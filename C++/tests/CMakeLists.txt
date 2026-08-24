find_package(GTest REQUIRED)

file(GLOB TEST_CXX_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

set(TEST_CUDA_SOURCES)
if(XFORMERS_USE_CUDA OR XFORMERS_USE_ROCM)
    file(GLOB TEST_CUDA_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
    )
endif()

set(ALL_TEST_SOURCES ${TEST_CXX_SOURCES} ${TEST_CUDA_SOURCES})

if(ALL_TEST_SOURCES)
    add_executable(xformers_tests
        ${ALL_TEST_SOURCES}
    )

    if(XFORMERS_USE_CUDA OR XFORMERS_USE_ROCM)
        set_target_properties(xformers_tests PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
    endif()

    target_include_directories(xformers_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${XFORMERS_COMMON_INCLUDE_DIRS}
        ${GTEST_INCLUDE_DIRS}
    )

    target_compile_definitions(xformers_tests PRIVATE
        ${XFORMERS_BACKEND_DEFINES}
    )

    target_link_libraries(xformers_tests PRIVATE
        xformers
        xformers_ops
        ${XFORMERS_COMMON_LINK_LIBS}
        GTest::gtest
        GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(xformers_tests)
endif()
