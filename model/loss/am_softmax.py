# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.14
""" Implement of AM-softmax loss from
    https://arxiv.org/pdf/1801.05599.pdf
"""

import dataclasses
import glog
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class AmSoftmaxLossConfig:
    """ Config of AmSoftmax-Loss, future implemented loss 
        should config itself in this fashion.
    """
    embedding_dim: int = 192  # Default embedding dim used in ECAPA_TDNN
    num_classes: int = 10  # Default numclasses of sample_data
    scale_factor: float = 30.0  # Default scaling factor
    margin: float = 0.4  # Default margin refered from paper


class AmSoftmaxLoss(nn.Module):
    """ Additive Margin Softmax Loss Implementation """

    def __init__(self, config: AmSoftmaxLossConfig) -> None:
        super(AmSoftmaxLoss, self).__init__()

        # Initialize config from AmSoftmaxConfig
        self._embedding_dim = config.embedding_dim
        self._num_classes = config.num_classes
        self._scale_factor = config.scale_factor
        self._margin = config.margin

        # Initialize weights matrix as ground truth embedding to compute cosine sim
        self._weights = nn.Parameter(
            torch.FloatTensor(self._num_classes, self._embedding_dim))
        nn.init.xavier_uniform_(self._weights)

        self._criterion = torch.nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        """ Training step for loss backpropagation """

        # Transform labels as one hot_encodings
        one_hot_encoding = F.one_hot(labels,
                                     num_classes=self._num_classes).float()

        # Compute cosine sim
        cosine = F.linear(F.normalize(embeddings), F.normalize(self._weights))

        # Add margin on cosine when encounter same class according to paper
        # https://arxiv.org/pdf/1801.05599.pdf
        phi = cosine - self._margin
        cosine_margined = one_hot_encoding * phi
        pred = cosine_margined + (1.0 - one_hot_encoding) * cosine
        pred *= self._scale_factor

        # Process CE loss
        loss = self._criterion(pred, one_hot_encoding)
        return loss

    @torch.inference_mode(mode=True)
    def predict(self, embeddings, labels):
        """ Predict step for metric compute during eval stage """
        # TODO: should compute acc with added margin?

        # Transform labels as one hot_encodings
        one_hot_encoding = F.one_hot(labels,
                                     num_classes=self._num_classes).float()

        # Compute cosine sim
        cosine = F.linear(F.normalize(embeddings), F.normalize(self._weights))

        # Add margin on cosine when encounter same class according to paper
        # https://arxiv.org/pdf/1801.05599.pdf
        phi = cosine - self._margin
        cosine_margined = one_hot_encoding * phi
        pred = cosine_margined + (1.0 - one_hot_encoding) * cosine
        pred *= self._scale_factor

        return pred
