# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.13
""" Interface of embedding model for speaker recognition """

import torch
import torch.nn as nn

from model.embedding_model.resnet import ResNet, ResNetConfig
from model.embedding_model.ecapa_tdnn import EcapaTdnn, EcapaTdnnConfig


class EmbeddingModel(nn.Module):
    """ Interface of all designed Embedding models """

    def __init__(self, config) -> None:
        super(EmbeddingModel, self).__init__()

        if config["model"] == "Ecapa_Tdnn":
            self._embedding_model = EcapaTdnn(config=EcapaTdnnConfig(
                **config["config"]))
        elif config["model"] == "ResNet":
            self._embedding_model = ResNet(config=ResNetConfig(
                **config["config"]))
        else:
            raise ValueError("Embedding model {} is not implemented.".format(
                config["model"]))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # Training graph
        return self._embedding_model(feats)

    @torch.inference_mode(mode=True)
    def inference(self, feats: torch.Tensor) -> torch.Tensor:
        # Inference graph
        return self._embedding_model.inference(feats)
