// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// 
// This source code is licensed under the BSD license found in the
// LICENSE file in the root directory of this source tree.
#include "rmsnorm.h"

namespace xformers::ops {
namespace {
torch::Tensor rms_norm_impl(const torch::Tensor& x,
                            const std::optional<torch::Tensor>& weight,
                            double eps) {
  auto x_float = x.to(torch::kFloat32);
  auto variance = x_float.pow(2).mean(-1, true);
  auto output = x_float * torch::rsqrt(variance + eps);
  if (weight.has_value()) {
    output = output * weight->to(output.options());
  }
  return output.to(x.scalar_type());
}
} // namespace

torch::Tensor rms_norm(const torch::Tensor& x,
                       const std::optional<torch::Tensor>& weight,
                       double eps) {
  return rms_norm_impl(x, weight, eps);
}

torch::Tensor rms_norm_add(torch::Tensor& x,
                           const torch::Tensor& y,
                           const std::optional<torch::Tensor>& weight,
                           double eps) {
  x.add_(y);
  return rms_norm_impl(x, weight, eps);
}

RMSNormImpl::RMSNormImpl(int64_t dim, bool include_weight, double eps) : eps_(eps) {
  if (include_weight) {
    auto parameter = register_parameter("weight", torch::ones({dim}));
    weight_ = parameter;
  }
}

torch::Tensor RMSNormImpl::forward(const torch::Tensor& x) {
  return rms_norm(x, weight_, eps_);
}

torch::Tensor RMSNormImpl::increment_and_forward_(torch::Tensor& x, const torch::Tensor& y) {
  return rms_norm_add(x, y, weight_, eps_);
}

} // namespace xformers::ops
