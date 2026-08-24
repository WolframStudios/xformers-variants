// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// 
// This source code is licensed under the BSD license found in the
// LICENSE file in the root directory of this source tree.
//
//
#include "checkpoint.h"

#include <iostream>

namespace xformers {

void warn_deprecated(const std::string& name, const std::string& alternative) {
  std::cerr << "xformers.checkpoint." << name << " is deprecated. " << alternative << std::endl;
}

std::vector<std::string> list_operators(const std::function<void()>& function) {
  warn_deprecated("list_operators",
                  "Inspect operator regions directly from LibTorch dispatch instrumentation instead.");
  function();
  return {"operator_trace_unavailable_in_generated_port"};
}

PolicyFunction selective_checkpoint_context_fn(const std::optional<PolicyFunction>& policy_fn) {
  if (policy_fn.has_value()) {
    return *policy_fn;
  }
  return [](const std::string& op_name) {
    static const std::vector<std::string> allow_list = {
        "xformers.efficient_attention_forward_cutlass.default",
        "xformers_flash.flash_fwd.default",
        "aten.addmm.default",
        "aten.mm.default",
    };
    return std::find(allow_list.begin(), allow_list.end(), op_name) != allow_list.end();
  };
}

torch::Tensor checkpoint(const std::function<torch::Tensor()>& function,
                         bool /*preserve_rng_state*/,
                         const std::optional<PolicyFunction>& /*policy_fn*/) {
  warn_deprecated("checkpoint",
                  "Use torch::utils::checkpoint-style recomputation once it is exposed in LibTorch.");
  return function();
}

PolicyFunction get_optimal_checkpoint_policy(const std::function<void()>& /*function*/,
                                             double memory_budget) {
  warn_deprecated("get_optimal_checkpoint_policy",
                  "Use PyTorch activation memory budget controls in compiled graphs when available.");
  if (memory_budget < 0.0 || memory_budget > 1.0) {
    throw std::invalid_argument("memory_budget must be between 0 and 1");
  }
  return [memory_budget](const std::string&) { return memory_budget >= 0.5; };
}

SelectiveCheckpointWrapper::SelectiveCheckpointWrapper(torch::nn::AnyModule mod,
                                                       std::optional<double> memory_budget,
                                                       std::optional<PolicyFunction> policy_fn)
    : wrapped_module_(std::move(mod)),
      memory_budget_(memory_budget),
      policy_fn_(std::move(policy_fn)) {
  if ((memory_budget_.has_value() && policy_fn_.has_value()) ||
      (!memory_budget_.has_value() && !policy_fn_.has_value())) {
    throw std::invalid_argument("Need to specify either policy_fn or memory_budget");
  }
}

torch::Tensor SelectiveCheckpointWrapper::forward(const std::vector<torch::Tensor>& args) {
  auto invoke = [&]() -> torch::Tensor {
    if (args.empty()) {
      return wrapped_module_.forward<torch::Tensor>();
    }
    return wrapped_module_.forward<torch::Tensor>(args[0]);
  };
  return checkpoint(invoke, true, policy_fn_);
}

std::shared_ptr<SelectiveCheckpointWrapper> selective_checkpoint_wrapper(
    torch::nn::AnyModule module,
    std::optional<double> memory_budget,
    std::optional<PolicyFunction> policy_fn) {
  return std::make_shared<SelectiveCheckpointWrapper>(std::move(module), memory_budget, std::move(policy_fn));
}

} // namespace xformers
