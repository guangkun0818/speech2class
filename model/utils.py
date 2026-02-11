# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.15
""" Utilities for training, metric specificly """

import dataclasses
import glog
import torch
import torch.nn as nn

from typing import List, Tuple


@dataclasses.dataclass
class MetricConfig:
  """ Config of metrics, supporting both Vad and Vpr tasks. """
  task: str = "VPR"
  top_ks: Tuple[int] = (1, 5)


class Metric(object):
  """ Metric for training """

  def __init__(self, config: MetricConfig):
    # Initialization
    if config.task == "VPR":
      self._compute_acc = self._vpr_accuarcy
    elif config.task == "VAD":
      self._compute_acc = self._vad_accuarcy
    self._top_ks = config.top_ks

  def _vpr_accuarcy(self, preds: torch.Tensor, labels: torch.Tensor, top_k):
    """ Compute ACC with given top_k of VPR task
            Args:
                preds: torch.Size(Batch_size, Num_classes) 
                labels: torch.Size(Batch_size) 
                top_k: Top_k ACC
        """
    batch_size = labels.shape[0]
    preds_top_k = preds.topk(top_k, dim=1, largest=True, sorted=True).indices
    labels = labels.unsqueeze(0).transpose(0, 1)
    # Compute num matches within batch
    num_matched = torch.eq(preds_top_k, labels).sum(dim=1).sum().float()

    # ACC of top_k
    return num_matched / batch_size

  @classmethod
  def vad_infer_metrics(cls, preds: torch.Tensor, labels: torch.Tensor):
    # Interface for inference metrics
    acc = cls._vad_accuarcy(cls, preds=preds, labels=labels)
    far, frr = cls._vad_far_frr(cls, preds=preds, labels=labels)
    return {"acc": acc, "far": far, "frr": frr}

  def _vad_accuarcy(self, preds: torch.Tensor, labels: torch.Tensor, top_k=None):
    """ Compute ACC with of VAD task
            Args:
                preds: torch.Size(Batch_size, Seq_len, 2) 
                labels: torch.Size(Batch_size, Seq_len)
                top_k: for API compliance
        """
    preds_labels = torch.argmax(preds, dim=-1)
    num_matched = torch.eq(preds_labels, labels).sum(dim=1).sum().float()

    # ACC num_matched_frames / total frame within a batch
    return num_matched / labels.numel()

  def _vad_far_frr(self, preds: torch.Tensor, labels: torch.Tensor):
    # False Accept Rate:
    #   Prediction 1 (speech) / Ground_truth 0 (non-speech)
    # False Reject Rate:
    #   Prediction 0 (non-speech) / Ground_truth 1 (speech)
    pred_labels = torch.argmax(preds, dim=-1)
    torch_sub = torch.sub(pred_labels, labels)

    fa_matched = torch.eq(torch_sub, 1).sum(dim=1).sum().float()
    fr_matched = torch.eq(torch_sub, -1).sum(dim=1).sum().float()
    return fa_matched / labels.numel(), fr_matched / labels.numel()

  def __call__(self, preds: torch.Tensor, labels: torch.Tensor):
    """ Call of Metrics func """

    # Return metrics as dic during eval stage
    metrics = {}
    if self._top_ks:
      for k in self._top_ks:
        metrics["top_{}_acc".format(k)] = self._compute_acc(preds=preds, labels=labels, top_k=k)
    else:
      metrics["acc"] = self._compute_acc(preds=preds, labels=labels, top_k=None)

    return metrics
