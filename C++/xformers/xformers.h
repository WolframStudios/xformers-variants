// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// 
// This source code is licensed under the BSD license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <functional>
#include <mutex>
#include <optional>
#include <string>

#include <torch/torch.h>

namespace xformers {

inline constexpr const char* kVersion = "0.0.0";
inline constexpr bool kIsOpenSource = true;

struct PythonLibraryHandle {
  std::string kind;
  std::string namespace_name;
};

template <typename T, typename Fn>
std::function<T()> compute_once(Fn fn) {
  auto value = std::make_shared<std::optional<T>>();
  auto once = std::make_shared<std::once_flag>();
  return [fn = std::move(fn), value, once]() mutable -> T {
    std::call_once(*once, [&]() { *value = fn(); });
    return **value;
  };
}

bool has_cpp_library();
bool is_triton_available();
PythonLibraryHandle get_python_lib();

} // namespace xformers
