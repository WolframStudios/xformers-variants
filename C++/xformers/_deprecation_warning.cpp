// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// 
// This source code is licensed under the BSD license found in the
// LICENSE file in the root directory of this source tree.
//
#include <torch/torch.h>
#include <any>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include "../translation_support.h"

/*
Original Python source: xformers/_deprecation_warning.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import warnings


def deprecated_function(self):
    name = repr(self)  # self.__name__
    msg = f"{name} is deprecated and is not maintained anymore. It might be removed in a future version of xFormers"
    warnings.warn(msg, FutureWarning, stacklevel=2)
*/

namespace xformers {

void deprecated_function(std::any self) {
  translation::unsupported_void("Translated placeholder for xformers/_deprecation_warning.py::deprecated_function");
}

} // namespace xformers
