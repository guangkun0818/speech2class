# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.15
""" Unittest of Cross-Entropy loss """

import glog
import torch
import unittest

import torch.nn.functional as F

from parameterized import parameterized
from model.loss.cross_entropy import CELoss, CELossConfig


class TestCrossEntropyLoss(unittest.TestCase):
  """ Unittest of Cross-Entropy loss """

  def setUp(self) -> None:
    self._ce_loss = CELoss(config=CELossConfig)

  @parameterized.expand([([[1, 0, 1, 0]],), ([[0, 0, 1, 1, 1, 0, 0]],)])
  def test_one_hot_encoding(self, labels):
    # Unittest of one_hot_encoding
    one_hot_encoding = F.one_hot(torch.LongTensor(labels), num_classes=2).float()
    self.assertEqual(one_hot_encoding.shape[0], len(labels))
    self.assertEqual(one_hot_encoding.shape[1], len(labels[0]))
    self.assertEqual(one_hot_encoding.shape[2], 2)
    self.assertTrue(torch.allclose(torch.LongTensor(labels), torch.argmax(one_hot_encoding,
                                                                          dim=-1)))

  @parameterized.expand([
      ([[1, 0, 1, 0], [1, 0, 1, 0]],),
      ([[1, 0, 1, 0]],),
      ([[0, 0, 1, 1, 1, 0, 0]],),
  ])
  def test_ce_loss_forward(self, labels):
    # Unittest of CE Loss forward
    labels = torch.LongTensor(labels)
    logits = torch.rand(labels.shape[0], labels.shape[1], 2)
    glog.info("Logits: {}".format(logits.shape))

    loss = self._ce_loss(logits, labels)
    glog.info("Loss: {}".format(loss))


if __name__ == "__main__":
  unittest.main()
