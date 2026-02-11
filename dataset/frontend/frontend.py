# Author: guangkun0818
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

    # NOTE: Weird shit happens when apply torch.jit.trace(KaldiWaveFeature)
    # along with below import. Seems like saome stupid marcos conflict when sanity
    # check when jit trace. Details refer to frontend_test.py
    from speechbrain.pretrained import EncoderClassifier

    # Load pretrained model from Huggingface with only
    # `compute_features` loaded for feature_extraction
    self._feature_extractor = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb").mods["compute_features"]

  @torch.no_grad()
  def forward(self, pcm: torch.Tensor) -> torch.Tensor:
    feats = self._feature_extractor(pcm)
    return feats.squeeze(0)


class KaldiWaveFeature(nn.Module):
  """ Frontend of train, eval and test dataset to get feature from
        PCMs in kaldi-style
    """

  def __init__(self,
               num_mel_bins=64,
               frame_length=25,
               frame_shift=10,
               dither=0.0,
               samplerate=16000) -> None:
    super(KaldiWaveFeature, self).__init__()

    self._feature_extractor = torchaudio.compliance.kaldi.fbank
    self._num_mel_bins = num_mel_bins
    self._frame_length = frame_length
    self._frame_shift = frame_shift
    self._dither = dither
    self._samplerate = samplerate

  @torch.no_grad()
  def forward(self, pcm: torch.Tensor) -> torch.Tensor:
    features = self._feature_extractor(pcm,
                                       num_mel_bins=self._num_mel_bins,
                                       frame_length=self._frame_length,
                                       frame_shift=self._frame_shift,
                                       dither=self._dither,
                                       energy_floor=0.0,
                                       sample_frequency=self._samplerate)
    return features
