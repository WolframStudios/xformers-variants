# cmake/backends/openvino.cmake
# ── Intel OpenVINO backend ────────────────────────────────────────────────────
# Works on Linux and Windows.
# Set OpenVINO_DIR to the OpenVINO cmake config dir, e.g.:
#   Linux  : /opt/intel/openvino_2024/runtime/cmake
#   Windows: C:/Program Files (x86)/Intel/openvino_2024/runtime/cmake

if(NOT DEFINED OpenVINO_DIR)
    if(WIN32)
        set(OpenVINO_DIR "C:/Program Files (x86)/Intel/openvino_2024/runtime/cmake")
    else()
        set(OpenVINO_DIR "/opt/intel/openvino_2024/runtime/cmake")
    endif()
endif()

find_package(OpenVINO REQUIRED
    COMPONENTS Runtime
    HINTS "${OpenVINO_DIR}"
)

# OpenVINO 2022+ uses openvino::runtime target
if(TARGET openvino::runtime)
    set(XFORMERS_OPENVINO_LIBS openvino::runtime)
elseif(TARGET IE::inference_engine)
    # Older InferenceEngine API
    set(XFORMERS_OPENVINO_LIBS IE::inference_engine)
else()
    message(FATAL_ERROR "[backend] OpenVINO: could not locate runtime target")
endif()

message(STATUS "[backend] OpenVINO ${OpenVINO_VERSION}")
