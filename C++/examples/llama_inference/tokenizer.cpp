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
Original Python source: examples/llama_inference/tokenizer.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os

from sentencepiece import SentencePieceProcessor


class Tokenizer:
    """Encoding/decoding text using SentencePiece."""

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a SentencePiece model.

        Args:
            model_path (str): The path to the SentencePiece model file.
        """
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        print(
            f"loaded SentencePiece model: "
            f"#words: {self.n_words} - "
            f"bos id: {self.bos_id} - "
            f"eos id: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool = True, eos: bool = False) -> list[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            list[int]: A list of token IDs.
        """
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: list[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (list[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.sp_model.decode(t)
*/

namespace examples {
namespace llama_inference {

class Tokenizer {
 public:
  /*
  Encoding/decoding text using SentencePiece.
  */
  Tokenizer() = default;
  void __init__(std::string model_path) {
    /*
    Initializes the Tokenizer with a SentencePiece model.
  
    Args:
        model_path (str): The path to the SentencePiece model file.
    */
    translation::unsupported_void("Translated placeholder for examples/llama_inference/tokenizer.py::__init__");
  }
  list[int64_t] encode(std::string s, bool bos = true, bool eos = false) {
    /*
    Encodes a string into a list of token IDs.
  
    Args:
        s (str): The input string to be encoded.
        bos (bool): Whether to prepend the beginning-of-sequence token.
        eos (bool): Whether to append the end-of-sequence token.
  
    Returns:
        list[int]: A list of token IDs.
    */
    return translation::unsupported<list[int64_t]>("Translated placeholder for examples/llama_inference/tokenizer.py::encode");
  }
  std::string decode(list[int64_t] t) {
    /*
    Decodes a list of token IDs into a string.
  
    Args:
        t (list[int]): The list of token IDs to be decoded.
  
    Returns:
        str: The decoded string.
    */
    return translation::unsupported<std::string>("Translated placeholder for examples/llama_inference/tokenizer.py::decode");
  }
};

} // namespace llama_inference
} // namespace examples
