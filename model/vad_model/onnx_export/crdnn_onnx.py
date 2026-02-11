# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.03.04
""" NOTE: Due to cache-based streaming design, Crdnn onnx export 
    should be seperated into 2 parts, CrdnnInit, and CrdnnInterence
    for Onnx export seems only accept forward method of the model.
"""

import torch
import torch.nn as nn

from typing import Tuple, List

from model.vad_model.crdnn import CrdnnConfig, Crdnn


class CrdnnOnnxInit(Crdnn):
  """ Crdnn init for onnx export """

  def __init__(self, config: CrdnnConfig):
    super(CrdnnOnnxInit, self).__init__(config)

  def forward(self, dummy_input: torch.Tensor):
    # Override original forward as initialize_cache
    return self.initialize_cache()


class CrdnnOnnxInference(Crdnn):
  """ Crdnn Inference for onnx export """

  def __init__(self, config: CrdnnConfig):
    super(CrdnnOnnxInference, self).__init__(config)

  def forward(self, x: torch.Tensor, cache: Tuple[List[torch.Tensor], Tuple[torch.Tensor,
                                                                            torch.Tensor]]):
    # Override original forward as inference
    logits, next_cache = self.inference(x, cache)

    return logits, next_cache
