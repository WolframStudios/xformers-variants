// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// 
// This source code is licensed under the BSD license found in the
// LICENSE file in the root directory of this source tree.
//
//
#include "ops.h"

#include <limits>

namespace xformers {
namespace ops {

void initialize_ops() {}

torch::Tensor masked_matmul(const torch::Tensor& a,
                            const torch::Tensor& b,
                            const std::optional<torch::Tensor>& mask) {
  auto att = torch::matmul(a, b);
  if (!mask.has_value()) {
    return att;
  }
  if (mask->scalar_type() == torch::kBool) {
    auto expanded = *mask;
    if (expanded.dim() == 2) {
      expanded = expanded.unsqueeze(0).expand({att.size(0), -1, -1});
    }
    auto neg_inf = torch::full_like(att, -std::numeric_limits<double>::infinity());
    return torch::where(expanded, att, neg_inf);
  }
  return att + *mask;
}

} // namespace ops
} // namespace xformers
