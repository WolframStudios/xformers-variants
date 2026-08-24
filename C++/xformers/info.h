// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// 
// This source code is licensed under the BSD license found in the
// LICENSE file in the root directory of this source tree.
//
//
#pragma once

#include <iostream>
#include <string>
#include <unordered_map>

#include "xformers.h"

namespace xformers {

inline std::unordered_map<std::string, std::string> get_features_status() {
  std::unordered_map<std::string, std::string> features;
  features["is_triton_available"] = is_triton_available() ? "true" : "false";
  features["build.info"] = has_cpp_library() ? "available" : "none";
  features["source.privacy"] = kIsOpenSource ? "open source" : "fairinternal";
  return features;
}

inline void print_info(std::ostream& os = std::cout) {
  auto features = get_features_status();
  os << "xFormers " << kVersion << '\n';
  features["pytorch.version"] = TORCH_VERSION;
#if defined(XFORMERS_USE_CUDA) || defined(XFORMERS_USE_ROCM)
  features["pytorch.cuda"] = torch::cuda::is_available() ? "available" : "not available";
  if (torch::cuda::is_available()) {
    const auto capability = torch::cuda::getDeviceCapability();
    features["gpu.compute_capability"] = std::to_string(capability.first) + "." + std::to_string(capability.second);
    features["gpu.name"] = torch::cuda::getDeviceName();
  }
#else
  features["pytorch.cuda"] = "disabled";
#endif
  for (const auto& [name, status] : features) {
    os << name << ": " << status << '\n';
  }
}

} // namespace xformers
