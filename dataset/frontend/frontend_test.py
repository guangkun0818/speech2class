# Author: Xiaoyue Yang
# Email: 609946862@qq.com
# Created on 2023.02.07
""" Unittest of Frontend """

import glog
import torch
import torchaudio
import unittest

from dataset.frontend.frontend import EcapaFrontend


class FrontendTest(unittest.TestCase):
    """ Unittest of EcapaFrontend """

    def setUp(self) -> None:
        self._frontend = EcapaFrontend()
        # TODO: Set test wavs

    def test_frontend(self):
        pcms = torch.rand(1, 32000)
        feats = self._frontend(pcms)
        glog.info("MFCC feature: {}".format(feats.shape))
        # According to paper http://arxiv.org/pdf/2010.11255.pdf
        # 80 dims MFCCs applied in ECAPA-TDNN systems
        self.assertEqual(feats.shape[-1], 80)
        self.assertEqual(len(feats.shape), 2)

    def test_frontend_torchscript(self):
        # Frontend torchscript export unittest
        pcms = pcms = torch.rand(1, 32000)
        pt_feats = self._frontend(pcms)

        torchscript_frontend = torch.jit.trace(self._frontend,
                                               example_inputs=pcms)
        torchscript_frontend = torch.jit.script(torchscript_frontend)

        # Torchscript frontend precision check
        ts_feats = torchscript_frontend(pcms)
        glog.info("Torchscript output :{}".format(ts_feats.shape))
        glog.info("Checkpoint output: {}".format(pt_feats.shape))
        self.assertTrue(torch.allclose(pt_feats, ts_feats))


if __name__ == "__main__":
    unittest.main()
