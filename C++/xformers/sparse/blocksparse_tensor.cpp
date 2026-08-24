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
#include "../../translation_support.h"

/*
Original Python source: xformers/sparse/blocksparse_tensor.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from xformers.ops import masked_matmul

logger = logging.getLogger("xformers")


# TODO: This is all now deprecated because PyTorch has its own blocksparse ops
def _spmm(b, layout, values):
    N, nnz, _, block_size = values.shape
    br = b.reshape(
        b.shape[0], b.shape[1], b.shape[2] // block_size, block_size, b.shape[3]
    )
    # perform matmul on blocks
    h, r, c = layout.nonzero(as_tuple=True)
    temp = values @ br[:, h, c, :]

    linear_idx = h * (b.shape[2] // block_size) + r
    out = torch.zeros(
        N,
        b.shape[1] * layout.shape[-2],
        block_size,
        b.shape[3],
        dtype=b.dtype,
        device=b.device,
    )
    # now aggregate the results of the different blocks
    out.index_add_(1, linear_idx.to(b.device), temp)
    out = out.reshape(N, b.shape[1], -1, b.shape[3])
    return out


def _softmax(layout, values):
    h, r, c = layout.nonzero(as_tuple=True)
    norms = torch.logsumexp(values, dim=-1, keepdim=True)
    linear_idx = h * layout.shape[1] + r

    out_t = torch.zeros(
        norms.shape[0],
        layout.shape[0] * layout.shape[1],
        norms.shape[2],
        norms.shape[3],
        dtype=norms.dtype,
        device=norms.device,
    )
    max_val = norms.max()
    out_t.index_add_(
        1, linear_idx.to(values.device), (norms - max_val).exp()
    ).clamp_min_(1e-24).log_().add_(max_val)
    out = torch.exp(values - out_t[:, linear_idx])
    return out


def _sddmm(a, b, layout):
    block_size = a.shape[-2] // layout.shape[-2]
    a = a.reshape(
        a.shape[0], a.shape[1], a.shape[2] // block_size, block_size, a.shape[3]
    )
    b = b.reshape(
        b.shape[0], b.shape[1], b.shape[2] // block_size, block_size, b.shape[3]
    )

    h, r, c = layout.nonzero(as_tuple=True)

    out = torch.einsum("nhik,nhjk->nhij", a[:, h, r, :, :], b[:, h, c, :, :])
    return out


class BlockSparseTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, values, layout):
        kwargs = {}
        kwargs["device"] = values.device
        kwargs["dtype"] = values.dtype
        kwargs["layout"] = values.layout
        kwargs["requires_grad"] = values.requires_grad
        assert values.ndim == 4
        B, _, block_size, _ = values.shape
        C, h, w = layout.shape
        # TODO validate shape of layout vs values
        shape = (B, C, block_size * h, block_size * w)
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, values, layout):
        assert values.shape[-2] == values.shape[-1]
        assert (
            values.device == layout.device
        ), "Both values and layout need to reside on the same device"
        block_size = values.shape[-1]
        # TODO: make this check conditioned on the use of Triton
        assert block_size >= 16, "Minimum block size is 16, for now at least"

        # Pure blocksparse data
        self.__values = values
        self.__layout = layout

    def __repr__(self):
        return f"block_sparse_tensor(shape={self.shape}, values={self.__values})"

    def values(self):
        return self.__values

    @classmethod
    def _raw_wrap(cls, values, layout):
        matrix = cls.__new__(cls, values, layout)
        matrix.__values = values
        matrix.__layout = layout
        return matrix

    @classmethod
    def _wrap(cls, values, bmat):
        matrix = cls.__new__(cls, values, bmat.__layout)
        matrix.__values = values
        matrix.__layout = bmat.__layout
        return matrix

    @classmethod
    def _bmm(cls, arg0, arg1):
        if not (isinstance(arg0, cls) and type(arg1) is torch.Tensor):
            return NotImplemented
        res = _spmm(arg1, arg0.__layout, arg0.__values)
        return res

    @classmethod
    def _masked_matmul(cls, a, b, mask):
        if not (type(a) is torch.Tensor and type(b) is torch.Tensor):
            return NotImplemented
        b = b.transpose(-2, -1)
        assert b.is_contiguous()
        res = _sddmm(a, b, mask.__layout)
        return cls._wrap(res, mask)

    @classmethod
    def _softmax(cls, arg0, dim):
        if not (dim == -1 or dim == 2):
            return NotImplemented
        res = _softmax(arg0.__layout, arg0.__values)
        return cls._wrap(res, arg0)

    @classmethod
    def _to(cls, arg0, device):
        if isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        return cls(
            arg0.__values.to(device=device),
            arg0.__layout,
        )

    @classmethod
    def _copy(cls, arg0, arg1):
        if not (isinstance(arg0, cls) and isinstance(arg1, cls)):
            return NotImplemented
        assert arg0.shape == arg1.shape
        av0, av1 = arg0.__values, arg1.__values
        av0.resize_as_(av1).copy_(av1)
        av0, av1 = arg0.__layout, arg1.__layout
        av0.resize_as_(av1).copy_(av1)
        return arg0

    @classmethod
    def _equal(cls, arg0, arg1):
        if not (isinstance(arg0, cls) and isinstance(arg1, cls)):
            return NotImplemented
        if arg0.shape != arg1.shape:
            return False
        if not torch.equal(arg0.__values, arg1.__values):
            return False
        if not torch.equal(arg0.__layout, arg1.__layout):
            return False
        return True

    @classmethod
    def _to_dense(cls, arg0):
        # out = torch.zeros(arg0.shape, dtype=arg0.dtype, device=arg0.device, requires_grad=arg0.requires_grad)
        out = torch.zeros(arg0.shape, dtype=arg0.dtype, device=arg0.device)
        values = arg0.__values
        layout = arg0.__layout
        block_size = values.shape[-1]
        blocks_i = layout.shape[-2]
        blocks_j = layout.shape[-1]

        out_r = out.reshape(
            arg0.shape[0], arg0.shape[1], blocks_i, block_size, blocks_j, block_size
        )

        for idx, (h, i, j) in enumerate(zip(*layout.nonzero(as_tuple=True))):
            out_r[:, h, i, :, j, :] = values[:, idx, :, :]

        return out

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in [
            torch.Tensor.bmm,
            torch.bmm,
            torch.Tensor.__matmul__,
            torch.matmul,
            torch.Tensor.matmul,
        ]:
            assert len(args) == 2
            return cls._bmm(args[0], args[1])

        if func in [torch.Tensor.softmax, torch.nn.functional.softmax, torch.softmax]:
            return cls._softmax(args[0], kwargs["dim"])

        if func == masked_matmul:
            assert len(args) == 3
            return cls._masked_matmul(args[0], args[1], args[2])

        if func in [torch.nn.functional.dropout, torch.dropout, torch.dropout_]:
            x = args[0]
            values = x.__values.clone()
            values = func(values, *args[1:], **kwargs)
            return cls._wrap(values, x)

        if func == torch.Tensor.to:
            # print(args, kwargs)
            assert len(args) >= 2
            return cls._to(args[0], args[1])
            # return cls._to(args[0], kwargs["device"])

        if func in [torch.Tensor.copy_]:
            assert len(args) == 2
            return cls._copy(args[0], args[1])

        if func in [torch.Tensor.equal, torch.equal]:
            assert len(args) == 2
            return cls._equal(args[0], args[1])

        if func == torch.Tensor.to_dense:
            assert len(args) == 1
            return cls._to_dense(args[0])

        if func == torch.Tensor.detach:
            x = args[0]
            values = x.__values.clone()
            values = func(values, *args[1:], **kwargs)
            return cls._wrap(values, x)

        if func == torch.Tensor.__deepcopy__:
            x = args[0]
            memo = args[1]
            return cls._raw_wrap(
                x.__values.__deepcopy__(memo),
                x.__layout.__deepcopy__(memo),
            )

        if func in [torch.Tensor.grad.__get__, torch.Tensor._grad.__get__]:
            assert len(args) == 1
            assert len(kwargs) == 0
            x = args[0]
            return cls._wrap(x.__values.grad, x)

        if func == torch.Tensor.requires_grad_:
            func(args[0].__values)

        with torch._C.DisableTorchFunction():
            ret = func(*args, **kwargs)
            # TODO: check this
            if func in torch.overrides.get_default_nowrap_functions():
                return ret
            return torch._tensor._convert(ret, cls)

        return NotImplemented

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        return NotImplemented
*/

namespace xformers {
namespace sparse {

std::any _spmm(torch::Tensor b, std::any layout, std::any values) {
  return translation::unsupported<std::any>("Translated placeholder for xformers/sparse/blocksparse_tensor.py::_spmm");
}

std::any _softmax(std::any layout, std::any values) {
  return translation::unsupported<std::any>("Translated placeholder for xformers/sparse/blocksparse_tensor.py::_softmax");
}

std::any _sddmm(torch::Tensor a, torch::Tensor b, std::any layout) {
  return translation::unsupported<std::any>("Translated placeholder for xformers/sparse/blocksparse_tensor.py::_sddmm");
}

class BlockSparseTensor : public Tensor {
 public:
  BlockSparseTensor() = default;
  std::any __new__(std::any values, std::any layout) {
    return translation::unsupported<std::any>("Translated placeholder for xformers/sparse/blocksparse_tensor.py::__new__");
  }
  void __init__(std::any values, std::any layout) {
    translation::unsupported_void("Translated placeholder for xformers/sparse/blocksparse_tensor.py::__init__");
  }
  std::any __repr__() {
    return translation::unsupported<std::any>("Translated placeholder for xformers/sparse/blocksparse_tensor.py::__repr__");
  }
  std::any values() {
    return translation::unsupported<std::any>("Translated placeholder for xformers/sparse/blocksparse_tensor.py::values");
  }
  std::any _raw_wrap(std::any values, std::any layout) {
    return translation::unsupported<std::any>("Translated placeholder for xformers/sparse/blocksparse_tensor.py::_raw_wrap");
  }
  std::any _wrap(std::any values, std::any bmat) {
    return translation::unsupported<std::any>("Translated placeholder for xformers/sparse/blocksparse_tensor.py::_wrap");
  }
  std::any _bmm(std::any arg0, std::any arg1) {
    return translation::unsupported<std::any>("Translated placeholder for xformers/sparse/blocksparse_tensor.py::_bmm");
  }
  std::any _masked_matmul(torch::Tensor a, torch::Tensor b, torch::Tensor mask) {
    return translation::unsupported<std::any>("Translated placeholder for xformers/sparse/blocksparse_tensor.py::_masked_matmul");
  }
  std::any _softmax(std::any arg0, int64_t dim) {
    return translation::unsupported<std::any>("Translated placeholder for xformers/sparse/blocksparse_tensor.py::_softmax");
  }
  std::any _to(std::any arg0, std::any device) {
    return translation::unsupported<std::any>("Translated placeholder for xformers/sparse/blocksparse_tensor.py::_to");
  }
  std::any _copy(std::any arg0, std::any arg1) {
    return translation::unsupported<std::any>("Translated placeholder for xformers/sparse/blocksparse_tensor.py::_copy");
  }
  std::any _equal(std::any arg0, std::any arg1) {
    return translation::unsupported<std::any>("Translated placeholder for xformers/sparse/blocksparse_tensor.py::_equal");
  }
  std::any _to_dense(std::any arg0) {
    return translation::unsupported<std::any>("Translated placeholder for xformers/sparse/blocksparse_tensor.py::_to_dense");
  }
  std::any __torch_function__(std::any func, std::any types, std::any args, std::any kwargs) {
    return translation::unsupported<std::any>("Translated placeholder for xformers/sparse/blocksparse_tensor.py::__torch_function__");
  }
  std::any __torch_dispatch__(std::any func, std::any types, std::any args, std::any kwargs) {
    return translation::unsupported<std::any>("Translated placeholder for xformers/sparse/blocksparse_tensor.py::__torch_dispatch__");
  }
};

} // namespace sparse
} // namespace xformers
