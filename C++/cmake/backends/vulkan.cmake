# cmake/backends/vulkan.cmake
# ── Vulkan compute backend ────────────────────────────────────────────────────
# Works on Linux and Windows; requires Vulkan SDK.
# Set VULKAN_SDK env-var or pass -DVULKAN_SDK=/path to cmake.

find_package(Vulkan REQUIRED)

# Optional: glslangValidator for compiling GLSL compute shaders to SPIR-V
find_program(GLSLC_EXECUTABLE glslc
    HINTS
        "${Vulkan_INCLUDE_DIR}/../Bin"   # Windows SDK layout
        "/usr/bin"
        "/usr/local/bin"
)

if(GLSLC_EXECUTABLE)
    message(STATUS "[backend] Vulkan glslc: ${GLSLC_EXECUTABLE}")
else()
    message(WARNING "[backend] Vulkan: glslc not found – SPIR-V shaders won't be compiled automatically")
endif()

# Helper function: compile a GLSL compute shader to SPIR-V at build time
# Usage: xformers_vulkan_shader(target shader.comp OUTPUT shader.spv)
function(xformers_vulkan_shader target glsl_file output_spv)
    if(GLSLC_EXECUTABLE)
        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${output_spv}"
            COMMAND ${GLSLC_EXECUTABLE} -fshader-stage=compute
                    "${CMAKE_CURRENT_SOURCE_DIR}/${glsl_file}"
                    -o "${CMAKE_CURRENT_BINARY_DIR}/${output_spv}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${glsl_file}"
            COMMENT "Compiling Vulkan shader ${glsl_file} → ${output_spv}"
        )
        target_sources(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/${output_spv}")
    endif()
endfunction()

set(XFORMERS_VULKAN_LIBS Vulkan::Vulkan)

include_directories(${Vulkan_INCLUDE_DIRS})

message(STATUS "[backend] Vulkan ${Vulkan_VERSION}")
