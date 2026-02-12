# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.15
""" Implement of Cross-Entropy loss """

import dataclasses
import glog
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class CELossConfig:
  """ Config of CE loss. If no configurable parameter, please 
        maintain the dummy config for the API compliance.    
    """
  dummy: int = -1


class CELoss(nn.Module):
  """ Cross-Entropy Loss Implement, specifically for Vad task. """

  def __init__(self, config: CELossConfig) -> None:
    super(CELoss, self).__init__()

    self._criterion = torch.nn.CrossEntropyLoss()

  def forward(self, logits: torch.Tensor, labels: torch.LongTensor):
    """ Training step for loss backpropagation 
            logits: (B, T, 2), output before softmax since softmax will 
                compute within CELoss.
            labels: (B, T), where 1 indicating speech frame, 0 indicating non-speech
                [[1, 1, 1, 0, 0, ..., 1, 1]
                 [0, 0, 0, 1, 1, ..., 0, 0]
                 ...
                 [1, 1, 0, 1, 1, ..., 0, 0]]
        """
    # Transform scalar label into vector.
    batch_size = logits.shape[0]
    seq_len = logits.shape[1]
    one_hot_encoding = F.one_hot(labels, num_classes=2).float()
    # Flatten both logits and labels from (B, T, 2) -> (B * T, 2),
    # where dim=1 indicating speech frame.
    logits = logits.contiguous().reshape(batch_size * seq_len, -1)
    one_hot_encoding = one_hot_encoding.contiguous().reshape(
        batch_size * seq_len, -1)

    # Compute final loss
    loss = self._criterion(logits, one_hot_encoding)
    return loss

  @torch.inference_mode(mode=True)
  def predict(self, logits: torch.Tensor, labels: torch.LongTensor = None):
    """ Predict step for metric compute during eval stage """
    return logits
