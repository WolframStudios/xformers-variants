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
#include "../../../translation_support.h"

/*
Original Python source: xformers/ops/fmha/common.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from mslk.attention.fmha.common import (  # noqa: E402, F401
    _attn_bias_apply,
    _is_bias_type_supported_in_BMK,
    AttentionBwOpBase,
    AttentionFwOpBase,
    AttentionOp,
    AttentionOpBase,
    bmk2bmhk,
    check_lastdim_alignment_stride1,
    Context,
    Gradients,
    Inputs,
    pack_fp8_tensorwise_per_head,
    ScaledTensor,
)
*/

namespace xformers {
namespace ops {
namespace fmha {

// Module placeholder for xformers/ops/fmha/common.py

} // namespace fmha
} // namespace ops
} // namespace xformers
