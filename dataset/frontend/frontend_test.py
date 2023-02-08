# Author: Xiaoyue Yang
# Email: 609946862@qq.com
# Created on 2023.02.07
""" Unittest of Frontend """

import glog
import torch
import torchaudio
import unittest

from dataset.frontend.frontend import EcapaFrontend, KaldiWaveFeature


class FrontendTest(unittest.TestCase):
    """ Unittest of EcapaFrontend """

    def setUp(self) -> None:
        self._frontend = EcapaFrontend()
        self._test_data_1 = "sample_data/data/wavs/251-136532-0007.wav"
        self._test_data_2 = "sample_data/data/wavs/1462-170138-0015.wav"

    def test_frontend(self):
        pcms, _ = torchaudio.load(self._test_data_1)
        feats = self._frontend(pcms)
        glog.info("MFCC feature: {}".format(feats.shape))
        # According to paper http://arxiv.org/pdf/2010.11255.pdf
        # 80 dims MFCCs applied in ECAPA-TDNN systems
        self.assertEqual(feats.shape[-1], 80)
        self.assertEqual(len(feats.shape), 2)

    def test_frontend_torchscript(self):
        # Frontend torchscript export unittest
        pcms, _ = torchaudio.load(self._test_data_1)
        torchscript_frontend = torch.jit.trace(self._frontend,
                                               example_inputs=pcms)
        torchscript_frontend = torch.jit.script(torchscript_frontend)

        # Torchscript frontend precision check
        pcms, _ = torchaudio.load(self._test_data_2)
        pt_feats = self._frontend(pcms)
        ts_feats = torchscript_frontend(pcms)
        glog.info("Torchscript output :{}".format(ts_feats.shape))
        glog.info("Checkpoint output: {}".format(pt_feats.shape))
        self.assertTrue(torch.allclose(pt_feats, ts_feats))


class KaldiFbankTest(unittest.TestCase):
    """ Unittest of KaldiWaveFeature frontend """

    def setUp(self) -> None:
        self._test_data_1 = "sample_data/data/wavs/251-136532-0007.wav"
        self._test_data_2 = "sample_data/data/wavs/1462-170138-0015.wav"

        self._config = {
            "feat_config": {
                "num_mel_bins": 80,
                "frame_length": 25,
                "frame_shift": 10,
                "dither": 0.0,
                "samplerate": 16000,
            },
        }
        self._kaldi_frontend = KaldiWaveFeature(**self._config["feat_config"])

    def test_frontend_wave_feature(self):
        # Frontend forward unittest
        pcms, _ = torchaudio.load(self._test_data_1)
        feats = self._kaldi_frontend(pcms)
        glog.info("Fbank feature: {}".format(feats.shape))
        self.assertEqual(feats.shape[-1], self._kaldi_frontend._num_mel_bins)

    def test_frontend_torchscript(self):
        # Frontend torchscript export unittest

        pcms = torch.rand(1, 41360)
        # NOTE: Shit happens here, If we use pcms other than shape == (1, 41360) along with
        # speechbrain import, RuntimeError pops up: The size of tensor a (323) must match
        # the size of tensor b (257) at non-singleton dimension 0. After I bust my ass checking
        # source codes of torch.jit.trace and speechbrain, still no clean explanation of it. But
        # it is partially certained that jit.trace check_tensor_case is static and output frame
        # size will always be 257 when speechbrain imported. Marcos conflict maybe? Anyway, fix
        # example_input of pcms shape as (1, 41360), which can be precisly extracted 257 frames
        # features solving this unittest issue temporaily.
        pt_feats = self._kaldi_frontend(pcms)
        torchscript_frontend = torch.jit.trace(self._kaldi_frontend,
                                               example_inputs=pcms)
        torchscript_frontend = torch.jit.script(torchscript_frontend)

        # Torchscript frontend precision check
        ts_feats = torchscript_frontend(pcms)
        glog.info("Torchscript output :{}".format(ts_feats.shape))
        glog.info("Checkpoint output: {}".format(pt_feats.shape))
        self.assertTrue(torch.allclose(pt_feats, ts_feats))


if __name__ == "__main__":
    unittest.main()
