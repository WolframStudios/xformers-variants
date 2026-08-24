# cmake/backends/openml.cmake
# ── OpenML backend ────────────────────────────────────────────────────────────
# OpenML is an open platform for sharing datasets, algorithms, and experiments.
# C++ integration via the OpenML C++ API (https://github.com/openml/openml-cpp).
# Works on Linux and Windows.
#
# Set -DOPENML_DIR=/path/to/openml-cpp or set the OPENML_DIR env variable.

if(NOT DEFINED OPENML_DIR)
    if(DEFINED ENV{OPENML_DIR})
        set(OPENML_DIR "$ENV{OPENML_DIR}")
    else()
        if(WIN32)
            set(OPENML_DIR "C:/openml-cpp")
        else()
            set(OPENML_DIR "/usr/local/openml-cpp")
        endif()
    endif()
endif()

list(APPEND CMAKE_PREFIX_PATH "${OPENML_DIR}")

find_package(OpenML QUIET HINTS "${OPENML_DIR}/lib/cmake")

if(OpenML_FOUND)
    set(XFORMERS_OPENML_LIBS OpenML::OpenML)
    message(STATUS "[backend] OpenML ${OpenML_VERSION} at ${OPENML_DIR}")
else()
    # Header-only fallback: manually locate headers and lib
    find_path(OPENML_INCLUDE_DIR openml/openml.h
        HINTS
            "${OPENML_DIR}/include"
            "/usr/local/include"
            "/usr/include"
    )

    find_library(OPENML_LIBRARY
        NAMES openml openml-cpp
        HINTS
            "${OPENML_DIR}/lib"
            "/usr/local/lib"
            "/usr/lib"
    )

    if(OPENML_INCLUDE_DIR AND OPENML_LIBRARY)
        set(XFORMERS_OPENML_LIBS ${OPENML_LIBRARY})
        include_directories(${OPENML_INCLUDE_DIR})
        message(STATUS "[backend] OpenML (manual) headers: ${OPENML_INCLUDE_DIR}")
    else()
        set(XFORMERS_OPENML_LIBS "")
        message(WARNING
            "[backend] OpenML not found – set -DOPENML_DIR=/path/to/openml-cpp. "
            "Source: https://github.com/openml/openml-cpp")
    endif()
endif()

# OpenML requires libcurl for REST API calls (both Linux and Windows)
find_package(CURL REQUIRED)
list(APPEND XFORMERS_OPENML_LIBS CURL::libcurl)

# On Linux, also needs pthread
if(UNIX)
    list(APPEND XFORMERS_OPENML_LIBS Threads::Threads)
    find_package(Threads REQUIRED)
endif()
