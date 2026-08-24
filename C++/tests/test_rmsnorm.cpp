// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
//
// This source code is licensed under the BSD license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "../xformers/ops/rmsnorm.h"

namespace tests {
namespace {

torch::Tensor baseline_rms_norm(const torch::Tensor& x,
                                const std::optional<torch::Tensor>& weight,
                                double eps = 1.0e-6) {
  auto output = x.to(torch::kFloat32) *
                torch::rsqrt(x.to(torch::kFloat32).pow(2).mean(-1, true) + eps);
  if (weight.has_value()) {
    output = output * weight->to(output.options());
  }
  return output.to(x.scalar_type());
}

void expect_allclose(const torch::Tensor& lhs,
                     const torch::Tensor& rhs,
                     double atol,
                     double rtol) {
  ASSERT_TRUE(torch::allclose(lhs, rhs, rtol, atol))
      << "lhs=" << lhs << "\nrhs=" << rhs;
}

} // namespace

TEST(Rmsnorm, ForwardFloat32MatchesBaseline) {
  torch::manual_seed(1);

  constexpr int64_t kBatch = 31;
  constexpr int64_t kSeq = 27;
  constexpr int64_t kDim = 273;

  auto layer = xformers::ops::RMSNorm(kDim);
  auto input = torch::rand({kBatch, kSeq, kDim}, torch::TensorOptions().dtype(torch::kFloat32));

  auto output = layer->forward(input);
  ASSERT_EQ(output.sizes(), input.sizes());

  auto expected = baseline_rms_norm(input, layer->weight_, layer->eps_);
  expect_allclose(output, expected, 1.0e-8, 1.0e-5);

  auto weighted = output.clone();
  layer->weight_->copy_(torch::full({kDim}, 2.0f));
  auto changed = layer->forward(input);
  ASSERT_FALSE(torch::allclose(weighted, changed));

  auto no_weight_layer = xformers::ops::RMSNorm(kDim, false);
  auto no_weight_output = no_weight_layer->forward(input);
  auto no_weight_expected = baseline_rms_norm(input, std::nullopt, no_weight_layer->eps_);
  expect_allclose(no_weight_output, no_weight_expected, 1.0e-8, 1.0e-5);
}

TEST(Rmsnorm, IncrementMutatesInputAndMatchesForward) {
  torch::manual_seed(1);

  constexpr int64_t kBatch = 31;
  constexpr int64_t kSeq = 27;
  constexpr int64_t kDim = 273;

  auto layer = xformers::ops::RMSNorm(kDim, true);
  layer->weight_->copy_(torch::randn({kDim}));

  auto x_orig = torch::rand({kBatch, kSeq, kDim}, torch::TensorOptions().dtype(torch::kFloat32));
  auto y_orig = torch::rand({kBatch, kSeq, kDim}, torch::TensorOptions().dtype(torch::kFloat32));
  auto x = x_orig.clone();
  auto y = y_orig.clone();

  auto expected = layer->forward(x_orig + y_orig);
  auto out = layer->increment_and_forward_(x, y);

  expect_allclose(out, expected, 1.0e-8, 1.0e-5);
  expect_allclose(x, x_orig + y_orig, 1.0e-8, 1.0e-5);
  expect_allclose(y, y_orig, 1.0e-8, 1.0e-5);
}

} // namespace tests
