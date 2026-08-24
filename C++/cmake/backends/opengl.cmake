# cmake/backends/opengl.cmake
# ── OpenGL compute (via GLSL Compute Shaders) backend ─────────────────────────
# Works on Linux and Windows.
# Uses OpenGL + GLFW for context creation and GLEW for extension loading.
# Install: apt install libgl1-mesa-dev libglfw3-dev libglew-dev  (Linux)
#          vcpkg install glfw3 glew opengl                       (Windows)

find_package(OpenGL REQUIRED)

# GLFW – window/context creation
find_package(glfw3 QUIET)
if(NOT glfw3_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GLFW3 QUIET glfw3)
    endif()
endif()

# GLEW – OpenGL extension wrangler
find_package(GLEW QUIET)

set(XFORMERS_OPENGL_LIBS OpenGL::GL)

if(glfw3_FOUND)
    list(APPEND XFORMERS_OPENGL_LIBS glfw)
elseif(GLFW3_FOUND)
    list(APPEND XFORMERS_OPENGL_LIBS ${GLFW3_LIBRARIES})
    include_directories(${GLFW3_INCLUDE_DIRS})
else()
    message(WARNING "[backend] OpenGL: GLFW not found – context creation unavailable")
endif()

if(GLEW_FOUND)
    list(APPEND XFORMERS_OPENGL_LIBS GLEW::GLEW)
else()
    message(WARNING "[backend] OpenGL: GLEW not found – extension loading unavailable")
endif()

# On Linux, also need X11 / EGL
if(UNIX AND NOT APPLE)
    find_package(X11 QUIET)
    if(X11_FOUND)
        list(APPEND XFORMERS_OPENGL_LIBS ${X11_LIBRARIES})
    endif()
    # EGL (headless rendering / compute-only)
    find_library(EGL_LIBRARY EGL)
    if(EGL_LIBRARY)
        list(APPEND XFORMERS_OPENGL_LIBS ${EGL_LIBRARY})
    endif()
endif()

message(STATUS "[backend] OpenGL – libs: ${XFORMERS_OPENGL_LIBS}")
