// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// 
// This source code is licensed under the BSD license found in the
// LICENSE file in the root directory of this source tree.
//
//
#pragma once

#include <any>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <torch/torch.h>

namespace xformers {

struct ProfileMetadata {
  std::string name;
  double time_taken{0.0};
  double memory_used{0.0};
  int64_t curr_idx{0};
  std::any output_ids;
  std::tuple<int64_t, int64_t> inplace_info{0, 0};
  bool is_view_like{false};
  bool is_rand_op{false};
};

using PolicyFunction = std::function<bool(const std::string&)>;

void warn_deprecated(const std::string& name, const std::string& alternative);
std::vector<std::string> list_operators(const std::function<void()>& function);
PolicyFunction selective_checkpoint_context_fn(const std::optional<PolicyFunction>& policy_fn = std::nullopt);
torch::Tensor checkpoint(const std::function<torch::Tensor()>& function,
                         bool preserve_rng_state = true,
                         const std::optional<PolicyFunction>& policy_fn = std::nullopt);
PolicyFunction get_optimal_checkpoint_policy(const std::function<void()>& function,
                                             double memory_budget);

class SelectiveCheckpointWrapper : public torch::nn::Module {
 public:
  SelectiveCheckpointWrapper(torch::nn::AnyModule mod,
                             std::optional<double> memory_budget = std::nullopt,
                             std::optional<PolicyFunction> policy_fn = std::nullopt);

  torch::Tensor forward(const std::vector<torch::Tensor>& args = {});

 private:
  torch::nn::AnyModule wrapped_module_;
  std::optional<double> memory_budget_;
  std::optional<PolicyFunction> policy_fn_;
};

std::shared_ptr<SelectiveCheckpointWrapper> selective_checkpoint_wrapper(
    torch::nn::AnyModule module,
    std::optional<double> memory_budget = std::nullopt,
    std::optional<PolicyFunction> policy_fn = std::nullopt);

} // namespace xformers
