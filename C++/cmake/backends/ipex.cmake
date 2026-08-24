# cmake/backends/ipex.cmake
# ── Intel Extension for PyTorch (IPEX) backend ────────────────────────────────
# Works on Linux and Windows; supports Intel CPUs and Intel GPUs (Arc/Xe).
# Requires Intel Extension for PyTorch installed alongside LibTorch.
# Pass -DIPEX_DIR=/path/to/ipex or set IPEX_DIR env variable.

if(NOT DEFINED IPEX_DIR)
    if(DEFINED ENV{IPEX_DIR})
        set(IPEX_DIR "$ENV{IPEX_DIR}")
    else()
        # Try to discover via Python if available
        find_package(Python3 QUIET COMPONENTS Interpreter)
        if(Python3_FOUND)
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c
                    "import intel_extension_for_pytorch as ipex; import os; print(os.path.dirname(ipex.__file__))"
                OUTPUT_VARIABLE _IPEX_PYTHON_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(_IPEX_PYTHON_DIR)
                set(IPEX_DIR "${_IPEX_PYTHON_DIR}")
            endif()
        endif()
    endif()
endif()

if(IPEX_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${IPEX_DIR}")
    find_package(IPEX QUIET HINTS "${IPEX_DIR}")
endif()

if(IPEX_FOUND)
    set(XFORMERS_IPEX_LIBS ${IPEX_LIBRARIES})
    include_directories(${IPEX_INCLUDE_DIRS})
    message(STATUS "[backend] IPEX ${IPEX_VERSION} at ${IPEX_DIR}")
else()
    # Fallback: header-only mode; link manually at build time
    if(IPEX_DIR)
        include_directories("${IPEX_DIR}/include")
    endif()
    set(XFORMERS_IPEX_LIBS "")
    message(WARNING
        "[backend] IPEX cmake config not found – using header-only fallback. "
        "Set -DIPEX_DIR=/path/to/ipex for full support.")
endif()
