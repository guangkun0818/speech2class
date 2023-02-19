# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.19
""" Interface of all losses """

import torch
import torch.nn as nn

from model.loss.am_softmax import AmSoftmaxLoss, AmSoftmaxLossConfig
from model.loss.cross_entropy import CELoss, CELossConfig


class Loss(nn.Module):
    """ Loss interface for all designed losses """

    def __init__(self, config) -> None:
        super(Loss, self).__init__()
        if config["model"] == "AM-Softmax":
            self.loss = AmSoftmaxLoss(config=AmSoftmaxLossConfig(
                **config["config"]))
        elif config["model"] == "CELoss":
            self.loss = CELoss(config=CELossConfig(**config["config"]))
        else:
            NotImplementedError

    def forward(self, batch):
        """ Loss training graph """

        assert "embeddings" in batch or "logits" in batch
        assert "labels" in batch

        if batch.get("embeddings") is not None:
            # VPR tasks batch API
            return self.loss(batch["embeddings"], batch["labels"])
        elif batch.get("logits") is not None:
            # VAD tasks batch API
            return self.loss(batch["logits"], batch["labels"])

    @torch.inference_mode(mode=True)
    def predict(self, batch):
        """ Predict step for metric compute """

        assert "embeddings" in batch or "logits" in batch
        assert "labels" in batch

        if batch.get("embeddings") is not None:
            # VPR tasks batch API
            return self.loss.predict(batch["embeddings"], batch["labels"])
        elif batch.get("logits") is not None:
            # VAD tasks batch API
            return self.loss.predict(batch["logits"], batch["labels"])
