# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.13
""" Unittest of ResNet """

import glog
import torch
import unittest

from model.embedding_model.resnet import BasicBlock
from model.embedding_model.resnet import BottleNeckBlock
from model.embedding_model.resnet import StatisticPooling
from model.embedding_model.resnet import EmbeddingLayer
from model.embedding_model.resnet import ResNetConfig
from model.embedding_model.resnet import ResNet


class TestBasicBlock(unittest.TestCase):
  """ Unittest of BasicBlock """

  def setUp(self) -> None:
    self._config = {
        "in_channels": 32,
        "out_channels": 32,
        "stride": 1,
    }
    self._basic_block = BasicBlock(**self._config)

  def test_basicblock(self):
    # Unittest of BasicBlock
    feats = torch.rand(1, 32, 80, 200)
    output = self._basic_block(feats)
    glog.info("The output of basicblock shape: {}".format(output.shape))
    self.assertEqual(feats.shape[1] * self._basic_block.EXPANSION,
                     output.shape[1])


class TestBottleNeckBlock(unittest.TestCase):
  """ Unittest of BottleNeckBlock """

  def setUp(self) -> None:
    self._config = {
        "in_channels": 32,
        "out_channels": 32,
        "stride": 1,
    }
    self._bottleneck_block = BottleNeckBlock(**self._config)

  def test_bottleneck_block(self):
    # Unittest of BottleNeckBlock
    feats = torch.rand(1, 32, 80, 200)
    output = self._bottleneck_block(feats)
    glog.info("The output of bottlenectblock shape: {}".format(output.shape))
    self.assertEqual(feats.shape[1] * self._bottleneck_block.EXPANSION,
                     output.shape[1])


class TestStatisticPooling(unittest.TestCase):
  """ Unittest of StatisticPooling """

  def setUp(self) -> None:
    self._config = {"input_dim": 2560}
    self._statistic_pooling = StatisticPooling(**self._config)

  def test_statistic_pooling(self):
    # Unittest of StatisticPooling
    feats = torch.rand(1, 256, 10, 25)
    output = self._statistic_pooling(feats)
    glog.info("The input feats shape: {}".format(feats.shape))
    glog.info("The output statistic pooling shape: {}".format(output.shape))


class TestEmbeddingLayer(unittest.TestCase):
  """ Unittest of Embedding Layer """

  def setUp(self) -> None:

    self._config_1 = {"input_dim": 5120, "embedding_dims": [256]}
    self._config_2 = {"input_dim": 5120, "embedding_dims": [256, 256]}
    self._embedding_layer_1 = EmbeddingLayer(**self._config_1)
    self._embedding_layer_2 = EmbeddingLayer(**self._config_2)

  def test_embedding_layer(self):
    # Training stage
    glog.info("Training stage.....")
    feats = torch.rand(2, 5120)
    output_1 = self._embedding_layer_1(feats)
    glog.info("The input feats shape: {}".format(feats.shape))
    glog.info("The output embedding_layer_1 shape: {}".format(output_1.shape))

    output_2 = self._embedding_layer_2(feats)
    glog.info("The input feats shape: {}".format(feats.shape))
    glog.info("The output embedding_layer_2 shape: {}".format(output_2.shape))

    # Eval stage
    glog.info("Eval stage.....")
    feats = torch.rand(1, 5120)
    self._embedding_layer_1.eval()
    output_1 = self._embedding_layer_1(feats)
    glog.info("The input feats shape: {}".format(feats.shape))
    glog.info("The output embedding_layer_1 shape: {}".format(output_1.shape))

    self._embedding_layer_2.eval()
    output_2 = self._embedding_layer_2(feats)
    glog.info("The input feats shape: {}".format(feats.shape))
    glog.info("The output embedding_layer_2 shape: {}".format(output_2.shape))


class TestResNetModel(unittest.TestCase):
  """ Unittest of ResNet """

  def setUp(self) -> None:
    # Set up different type ResNet model config
    self._resnet18_config = {
        "feats_dim": 80,
        "block_type": "BasicBlock",
        "num_blocks": [2, 2, 2, 2],
        "in_channels": 32,
        "embedding_dims": [256]
    }
    self._resnet34_config = {
        "feats_dim": 80,
        "block_type": "BasicBlock",
        "num_blocks": [3, 4, 6, 3],
        "in_channels": 32,
        "embedding_dims": [128]
    }
    self._resnet50_config = {
        "feats_dim": 80,
        "block_type": "BottleNeckBlock",
        "num_blocks": [3, 4, 6, 3],
        "in_channels": 32,
        "embedding_dims": [256]
    }
    self._resnet101_config = {
        "feats_dim": 80,
        "block_type": "BottleNeckBlock",
        "num_blocks": [3, 4, 23, 3],
        "in_channels": 32,
        "embedding_dims": [128]
    }

    self._resnet18 = ResNet(config=ResNetConfig(**self._resnet18_config))
    self._resnet34 = ResNet(config=ResNetConfig(**self._resnet34_config))
    self._resnet50 = ResNet(config=ResNetConfig(**self._resnet50_config))
    self._resnet101 = ResNet(config=ResNetConfig(**self._resnet101_config))

  def test_resnet18(self):
    # Unitest of ResNet18
    feats = torch.rand(2, 200, 80)
    output = self._resnet18(feats)
    glog.info("ResNet18 embeddings shape: {}".format(output.shape))
    self.assertEqual(output.shape[1], 256)

    self._resnet18.eval()
    output = self._resnet18.inference(feats)
    glog.info("ResNet18 embeddings shape: {}".format(output.shape))
    self.assertEqual(output.shape[1], 256)

  def test_resnet34(self):
    # Unitest of ResNet34
    feats = torch.rand(2, 200, 80)
    output = self._resnet34(feats)
    glog.info("ResNet34 embeddings: {}".format(output.shape))
    self.assertEqual(output.shape[1], 128)

    self._resnet34.eval()
    output = self._resnet34.inference(feats)
    glog.info("ResNet34 embeddings shape: {}".format(output.shape))
    self.assertEqual(output.shape[1], 128)

  def test_resnet50(self):
    # Unitest of ResNet18
    feats = torch.rand(2, 200, 80)
    output = self._resnet50(feats)
    glog.info("ResNet50 embeddings: {}".format(output.shape))
    self.assertEqual(output.shape[1], 256)

    self._resnet50.eval()
    output = self._resnet50.inference(feats)
    glog.info("ResNet50 embeddings shape: {}".format(output.shape))
    self.assertEqual(output.shape[1], 256)

  def test_resnet101(self):
    # Unitest of ResNet101
    feats = torch.rand(2, 200, 80)
    output = self._resnet101(feats)
    glog.info("ResNet101 embeddings: {}".format(output.shape))
    self.assertEqual(output.shape[1], 128)

    self._resnet101.eval()
    output = self._resnet101.inference(feats)
    glog.info("ResNet101 embeddings shape: {}".format(output.shape))
    self.assertEqual(output.shape[1], 128)


if __name__ == "__main__":
  unittest.main()
