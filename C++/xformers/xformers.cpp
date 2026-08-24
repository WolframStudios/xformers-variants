// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// 
// This source code is licensed under the BSD license found in the
// LICENSE file in the root directory of this source tree.
//
#include "xformers.h"

#include <cstdlib>

namespace xformers {
namespace {
auto g_is_triton_available = compute_once<bool>([]() -> bool {
#if defined(XFORMERS_USE_CUDA) || defined(XFORMERS_USE_ROCM)
  if (const char* enabled = std::getenv("XFORMERS_ENABLE_TRITON")) {
    if (std::string(enabled) == "1") {
      return true;
    }
  }
  if (!torch::cuda::is_available()) {
    return false;
  }
  if (const char* disabled = std::getenv("XFORMERS_FORCE_DISABLE_TRITON")) {
    if (std::string(disabled) == "1") {
      return false;
    }
  }
  const auto capability = torch::cuda::getDeviceCapability();
  return capability.first >= 8;
#else
  return false;
#endif
});

auto g_python_library = compute_once<PythonLibraryHandle>([]() -> PythonLibraryHandle {
  return PythonLibraryHandle{"DEF", "xformers_python"};
});
} // namespace

bool has_cpp_library() {
  return true;
}

bool is_triton_available() {
  return g_is_triton_available();
}

PythonLibraryHandle get_python_lib() {
  return g_python_library();
}

} // namespace xformers
