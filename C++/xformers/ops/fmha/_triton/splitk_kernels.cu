#if defined(XFORMERS_USE_CUDA) || defined(XFORMERS_USE_ROCM)
// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// 
// This source code is licensed under the BSD license found in the
// LICENSE file in the root directory of this source tree.
//
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <any>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include "../../../../translation_support.h"

/*
Original Python source: xformers/ops/fmha/_triton/splitk_kernels.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import sys

if sys.version_info >= (3, 9):
    from mslk.attention.fmha._triton.splitk_kernels import (  # noqa: E402, F401
        AUTOTUNER_KEY,
        get_autotuner_cache,
        set_autotuner_cache,
    )
*/

namespace xformers {
namespace ops {
namespace fmha {
namespace _triton {

// Module placeholder for xformers/ops/fmha/_triton/splitk_kernels.py

} // namespace _triton
} // namespace fmha
} // namespace ops
} // namespace xformers
#else
// This translation unit is only enabled for CUDA/ROCm-class backends.
#endif
