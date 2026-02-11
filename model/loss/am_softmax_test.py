# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.14
""" Unittest of AmSoftmaxLoss """

import glog
import torch
import unittest

from model.loss.am_softmax import AmSoftmaxLoss, AmSoftmaxLossConfig


class TestAmSoftmaxLoss(unittest.TestCase):
  """ Unittest of AM-softmax loss """

  def setUp(self) -> None:
    config = {"embedding_dim": 256, "num_classes": 20, "scale_factor": 30.0, "margin": 0.4}
    self._loss = AmSoftmaxLoss(config=AmSoftmaxLossConfig(**config))

  def test_config(self):
    self.assertEqual(self._loss._embedding_dim, 256)
    self.assertEqual(self._loss._num_classes, 20)
    self.assertEqual(self._loss._margin, 0.4)

  def test_am_softmax_loss_forward(self):
    embeddings = torch.rand(10, 256)
    labels = torch.Tensor([9, 5, 6, 2, 4, 1, 8, 0, 7, 3]).long()
    loss = self._loss(embeddings, labels)
    glog.info("Loss: {}".format(loss))

  def test_am_softmax_loss_predict(self):
    embeddings = torch.rand(10, 256)
    labels = torch.Tensor([9, 5, 6, 2, 4, 1, 8, 0, 7, 3]).long()
    preds = self._loss.predict(embeddings, labels)
    glog.info("Predicts: {}".format(preds.shape))


if __name__ == "__main__":
  unittest.main()
