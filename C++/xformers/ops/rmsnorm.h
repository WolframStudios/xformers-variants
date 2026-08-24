// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// 
// This source code is licensed under the BSD license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <optional>

#include <torch/torch.h>

namespace xformers::ops {

torch::Tensor rms_norm(const torch::Tensor& x,
                       const std::optional<torch::Tensor>& weight = std::nullopt,
                       double eps = 1.0e-6);

torch::Tensor rms_norm_add(torch::Tensor& x,
                           const torch::Tensor& y,
                           const std::optional<torch::Tensor>& weight = std::nullopt,
                           double eps = 1.0e-6);

struct RMSNormImpl : torch::nn::Module {
  RMSNormImpl(int64_t dim, bool include_weight = true, double eps = 1.0e-6);

  torch::Tensor forward(const torch::Tensor& x);
  torch::Tensor increment_and_forward_(torch::Tensor& x, const torch::Tensor& y);

  double eps_{1.0e-6};
  std::optional<torch::Tensor> weight_;
};

TORCH_MODULE(RMSNorm);

} // namespace xformers::ops
