// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// 
// This source code is licensed under the BSD license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <string>

/*
Original Python source: xformers/sparse/__init__.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from .blocksparse_tensor import BlockSparseTensor  # noqa: F401

*/

namespace xformers {
namespace sparse {

void initialize_sparse();

} // namespace sparse
} // namespace xformers
