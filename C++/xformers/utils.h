// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// 
// This source code is licensed under the BSD license found in the
// LICENSE file in the root directory of this source tree.
//
//
#pragma once

#include <any>
#include <chrono>
#include <filesystem>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

namespace xformers {

struct Item {
  std::string constructor_name;
  std::any config;
};

std::vector<std::string> import_all_modules(const std::filesystem::path& root,
                                            const std::string& base_module);

template <typename ReferenceClass>
auto get_registry_decorator(std::unordered_map<std::string, Item>& class_registry,
                            std::set<std::string>& name_registry,
                            std::any default_config) {
  return [&class_registry, &name_registry, default_config](const std::string& name,
                                                           std::any config = std::any()) {
    std::any chosen = config.has_value() ? config : default_config;
    return [&class_registry, &name_registry, name, chosen](const std::string& class_name) {
      if (class_registry.count(name) != 0) {
        throw std::invalid_argument("Cannot register duplicate item (" + name + ")");
      }
      if (name_registry.count(class_name) != 0) {
        throw std::invalid_argument("Cannot register item with duplicate class name (" + class_name + ")");
      }
      class_registry.emplace(name, Item{class_name, chosen});
      name_registry.insert(class_name);
      return class_name;
    };
  };
}

template <typename Config>
Config generate_matching_config(const std::unordered_map<std::string, std::any>& /*superset*/) {
  return Config{};
}

double do_bench_cudagraph(const std::function<void()>& fn,
                          int64_t rep = 20,
                          const std::optional<std::vector<torch::Tensor>>& grad_to_none = std::nullopt);

} // namespace xformers
