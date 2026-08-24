# tests CMakeLists.txt

find_package(GTest REQUIRED)

file(GLOB TEST_CXX_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

file(GLOB TEST_CUDA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

set(ALL_TEST_SOURCES ${TEST_CXX_SOURCES} ${TEST_CUDA_SOURCES})

if(ALL_TEST_SOURCES)
    add_executable(xformers_tests
        ${ALL_TEST_SOURCES}
    )

    set_target_properties(xformers_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_include_directories(xformers_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${TORCH_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${GTEST_INCLUDE_DIRS}
    )

    target_link_libraries(xformers_tests PRIVATE
        xformers
        xformers_ops
        ${TORCH_LIBRARIES}
        ${CUDA_LIBRARIES}
        cuda
        GTest::GTest
        GTest::Main
    )

    include(GoogleTest)
    gtest_discover_tests(xformers_tests)
endif()
