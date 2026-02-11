# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.13
""" Unittest of ECAPA-TDNN """

import glog
import torch
import unittest

from model.embedding_model.ecapa_tdnn import EcapaTdnn, EcapaTdnnConfig


class TestEcapaTdnn(unittest.TestCase):
  """ Unittest of ECAPA-TDNN """

  def setUp(self) -> None:
    config = {"dummy": -1}
    self._ecapa_tdnn = EcapaTdnn(config=EcapaTdnnConfig(**config))

  def test_ecapa_tdnn_forward(self):
    glog.info("Forward test.....")
    feats = torch.rand(10, 300, 80)
    embeddings = self._ecapa_tdnn(feats)
    self.assertEqual(len(embeddings.shape), 2)
    self.assertEqual(embeddings.shape[1], 192)
    glog.info("Embedding size: {}".format(embeddings.shape))


if __name__ == "__main__":
  unittest.main()
