# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.13
""" ECAPA-TDNN pretrained with HuggingFace
    https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
    which was rank 1 in 2020 Voxceleb Speaker Recognition contest.
"""

import dataclasses
import torch
import torch.nn as nn


@dataclasses.dataclass
class EcapaTdnnConfig:
    """ Config of EcapaTdnn, future implemented embedding model 
        should config itself in this fashion. 
    """
    dummy: int = -1


class EcapaTdnn(nn.Module):
    """ Pretrained ECAPA-TDNN """

    def __init__(self, config: EcapaTdnnConfig) -> None:
        super(EcapaTdnn, self).__init__()

        # NOTE: Weird shit happens when apply torch.jit.trace(KaldiWaveFeature)
        # along with below import. Seems like saome stupid marcos conflict when sanity
        # check when jit trace. Details refer to frontend_test.py
        from speechbrain.pretrained import EncoderClassifier

        # Load pretrain model from Huggingface with only
        self._embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        ).mods["embedding_model"]

        self._activate_params()  # Activate frozen params

    def _activate_params(self):
        for key, params in self.named_parameters():
            params.requires_grad = True

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """ Training graph """
        return self._embedding_model(feats).squeeze(
            1)  # [batch_size, embedding_dim]

    @torch.inference_mode(mode=True)
    def inference(self, feats: torch.Tensor):
        """ Inference graph """
        return self._embedding_model(feats).squeeze(
            1)  # [batch_size, embedding_dim]
