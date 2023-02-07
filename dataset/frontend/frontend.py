# Author: Xiaoyue Yang
# Email: 609946862@qq.com
# Created on 2023.02.07
""" Frontend of wav processing """

import torchaudio
import torch
import torch.nn as nn


class EcapaFrontend(nn.Module):
    """ Frontend of VPR model to extract acoustic features.
        Currently use pretrained Huggingface/ECAPA-TDNN frontend.
    """

    def __init__(self) -> None:
        super(EcapaFrontend, self).__init__()

        from speechbrain.pretrained import EncoderClassifier

        # Load pretrained model from Huggingface with only
        # `compute_features` loaded for feature_extraction
        self._feature_extractor = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        ).mods["compute_features"]

    @torch.no_grad()
    def forward(self, pcm: torch.Tensor):
        feats = self._feature_extractor(pcm)
        return feats.squeeze(0)