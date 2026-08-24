// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// 
// This source code is licensed under the BSD license found in the
// LICENSE file in the root directory of this source tree.
//
//
#include "utils.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace xformers {

std::vector<std::string> import_all_modules(const std::filesystem::path& root,
                                            const std::string& base_module) {
  std::vector<std::string> modules;
  if (!std::filesystem::exists(root)) {
    return modules;
  }
  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto ext = entry.path().extension().string();
    const auto stem = entry.path().stem().string();
    if ((ext == ".py" || ext == ".pyc") && !stem.empty() && stem[0] != '_') {
      modules.push_back(base_module + "." + stem);
    }
  }
  std::sort(modules.begin(), modules.end());
  return modules;
}

double do_bench_cudagraph(const std::function<void()>& fn,
                          int64_t rep,
                          const std::optional<std::vector<torch::Tensor>>& grad_to_none) {
#if defined(XFORMERS_USE_CUDA) || defined(XFORMERS_USE_ROCM)
  if (rep <= 0) {
    throw std::invalid_argument("rep must be positive");
  }
  fn();
  torch::cuda::synchronize();
  const auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < rep; ++i) {
    if (grad_to_none.has_value()) {
      for (auto& tensor : *grad_to_none) {
        if (tensor.defined()) {
          tensor.mutable_grad() = torch::Tensor();
        }
      }
    }
    fn();
  }
  torch::cuda::synchronize();
  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
  return elapsed.count() / static_cast<double>(rep);
#else
  (void)fn;
  (void)rep;
  (void)grad_to_none;
  throw std::runtime_error("do_bench_cudagraph requires a CUDA- or ROCm-enabled backend");
#endif
}

} // namespace xformers
