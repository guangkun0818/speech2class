# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.13
""" Vad models interface, streaming infer impl required. """

import torch
import torch.nn as nn

from typing import Dict, Tuple

from model.vad_model.crdnn import Crdnn, CrdnnConfig


class VadModel(nn.Module):
  """ Interface of all designed VAD models """

  def __init__(self, config) -> None:
    super(VadModel, self).__init__()

    if config["model"] == "Crdnn":
      self.vad_model = Crdnn(config=CrdnnConfig(**config["config"]))
    else:
      raise ValueError("Vad model {} not supported.".format(config["model"]))

  def forward(self, feats):
    """ Training graph """
    return self.vad_model(feats)

  @torch.jit.export
  def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
    # Compute softmax logits.
    return self.vad_model.compute_logits(x)

  @torch.jit.export
  def initialize_cache(self):
    # Cache init API
    return self.vad_model.initialize_cache()

  @torch.jit.export
  @torch.inference_mode(mode=True)
  def inference(self, feats: torch.Tensor, cache: Dict):
    """ Inference graph interface, cache strategy for streaming support 
        """
    return self.vad_model.inference(feats, cache)
