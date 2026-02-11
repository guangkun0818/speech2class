# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.12
""" Unitest of Crdnn """

import glog
import unittest
import torch
from parameterized import parameterized
from model.vad_model.crdnn import Conv1dCnnBlock, Conv2dCnnBlock, CnnBlockConfig, Crdnn
from model.vad_model.crdnn import DnnBlock, DnnBlockConfig
from model.vad_model.crdnn import LstmRnnBlock, GruRnnBlock, RnnBlockConfig
from model.vad_model.crdnn import Crdnn, CrdnnConfig


class TestCnnBlock(unittest.TestCase):
  """ Unittest of Cnn Block """

  def setUp(self) -> None:
    cnn_1d_config = {
        "num_layers": 3,
        "conv_type": "conv1d",
        "input_dim": 64,
        "in_channels_config": [64, 128, 128],
        "out_channels_config": [128, 128, 64],
        "kernel_configs": [3, 5, 3],
        "dilation_config": [2, 1, 2]
    }
    self._cnn_block_1d = Conv1dCnnBlock(config=CnnBlockConfig(**cnn_1d_config))

    cnn_2d_config = {
        "num_layers": 3,
        "conv_type": "conv2d",
        "input_dim": 80,
        "in_channels_config": [1, 64, 128],
        "out_channels_config": [64, 128, 64],
        "kernel_configs": [(3, 3), (5, 3), (3, 3)],
        "stride_configs": [2, 2, 2],
        "dilation_config": [(2, 2), (1, 1), (2, 2)]
    }
    self._cnn_block_2d = Conv2dCnnBlock(config=CnnBlockConfig(**cnn_2d_config))

    cldnn_config = {
        "num_layers": 3,
        "conv_type": "conv2d",
        "input_dim": 64,
        "in_channels_config": [1, 64, 128],
        "out_channels_config": [64, 128, 64],
        "kernel_configs": [(1, 3), (1, 5), (1, 3)],
        "stride_configs": [2, 2, 2],
        "dilation_config": [(1, 2), (1, 1), (1, 2)]
    }
    self._cldnn = Conv2dCnnBlock(config=CnnBlockConfig(**cldnn_config))

  def test_model_config(self):
    # Unittest of building model from config
    self.assertListEqual(self._cldnn._in_channels_config, [1, 64, 128])
    self.assertListEqual(self._cldnn._kernel_configs, [(1, 3), (1, 5), (1, 3)])
    self.assertListEqual(self._cnn_block_2d._kernel_configs, [(3, 3), (5, 3), (3, 3)])

  @parameterized.expand([(101,), (156,), (271,)])
  def test_cnn_block_1d_forward(self, feat_len):
    # Unitest of Cnn Block forward with Conv1d. default feats_dim = 64
    feats = torch.rand(4, feat_len, self._cnn_block_1d._input_dim)
    output = self._cnn_block_1d(feats)
    self.assertEqual(output.shape[1], feat_len)
    self.assertEqual(output.shape[2], self._cnn_block_1d._out_channels_config[-1])
    self.assertEqual(output.shape[-1], self._cnn_block_1d.output_dim)

  @parameterized.expand([(101,), (156,), (271,)])
  def test_cnn_block_1d_streaming_inference(self, feat_len):
    # Unitest of CNN Block Streaming inference with Conv1d. default feats_dim = 64
    self._cnn_block_1d.train(False)
    feats = torch.rand(1, feat_len, self._cnn_block_1d._input_dim)
    cache = self._cnn_block_1d.initialize_cnn_cache()
    non_steam_output = self._cnn_block_1d(feats)  # Non-streaming

    # Streaming mode
    stream_output = []
    for frame_id in range(0, feat_len, 1):
      # Simulate streaming inference
      output, cache = self._cnn_block_1d.inference(feats[:, frame_id:frame_id + 1, :], cache)
      stream_output.append(output)
    stream_output = torch.concat(stream_output, dim=1)
    self.assertEqual(stream_output.shape[1], feat_len)

    # with maximum abs diff of precision with 2e-7
    glog.info(torch.max(stream_output - non_steam_output))
    self.assertTrue(torch.allclose(non_steam_output, stream_output, rtol=3e-5, atol=3e-7))

  @parameterized.expand([(101,), (156,), (271,)])
  def test_cnn_block_2d_forward(self, feat_len):
    # Unitest of Cnn Block forward with Conv2d.
    feats = torch.rand(4, feat_len, self._cnn_block_2d._input_dim)
    output = self._cnn_block_2d(feats)
    self.assertEqual(output.shape[1], feat_len)
    self.assertEqual(output.shape[-1], self._cnn_block_2d.output_dim)

  @parameterized.expand([(101,), (156,), (271,), (473,)])
  def test_cnn_block_2d_streaming_inference(self, feat_len):
    # Unitest of Cnn Block Streaming inference with Conv2d.
    self._cnn_block_2d.train(False)
    feats = torch.rand(1, feat_len, self._cnn_block_2d._input_dim)
    cache = self._cnn_block_2d.initialize_cnn_cache()
    non_steam_output = self._cnn_block_2d(feats)  # Non-streaming

    # Streaming mode
    stream_output = []
    for frame_id in range(0, feat_len, 1):
      # Simulate streaming inference
      output, cache = self._cnn_block_2d.inference(feats[:, frame_id:frame_id + 1, :], cache)
      stream_output.append(output)
    stream_output = torch.concat(stream_output, dim=1)
    self.assertEqual(stream_output.shape[1], feat_len)

    # with maximum abs diff of precision with 2e-7
    glog.info(torch.max(stream_output - non_steam_output))
    self.assertTrue(torch.allclose(non_steam_output, stream_output, rtol=3e-5, atol=3e-7))

  @parameterized.expand([(101,), (156,), (271,)])
  def test_cldnn_forward(self, feat_len):
    # Unitest of Cnn Block forward of Cldnn.
    feats = torch.rand(4, feat_len, self._cldnn._input_dim)
    output = self._cldnn(feats)
    self.assertEqual(output.shape[1], feat_len)
    self.assertEqual(output.shape[-1], self._cldnn.output_dim)

  @parameterized.expand([(101,), (156,), (271,), (473,)])
  def test_cldnn_streaming_inference(self, feat_len):
    # Unitest of Cnn Block Streaming inference of Cldnn.
    self._cldnn.train(False)
    feats = torch.rand(1, feat_len, self._cldnn._input_dim)
    cache = self._cldnn.initialize_cnn_cache()
    non_steam_output = self._cldnn(feats)  # Non-streaming

    # Streaming mode
    stream_output = []
    for frame_id in range(0, feat_len, 1):
      # Simulate streaming inference
      output, cache = self._cldnn.inference(feats[:, frame_id:frame_id + 1, :], cache)
      stream_output.append(output)
    stream_output = torch.concat(stream_output, dim=1)
    self.assertEqual(stream_output.shape[1], feat_len)

    # with maximum abs diff of precision with 2e-7
    glog.info(torch.max(stream_output - non_steam_output))
    self.assertTrue(torch.allclose(non_steam_output, stream_output, rtol=3e-5, atol=3e-7))


class TestDnnBlock(unittest.TestCase):
  """ Unittest of DnnBLock """

  def setUp(self) -> None:
    config = {"num_layers": 4, "hidden_dim": 64}
    self._dnn_block = DnnBlock(_input_dim=320, config=DnnBlockConfig(**config))

  @parameterized.expand([(101,), (156,), (271,)])
  def test_dnn_block_forward(self, feat_len):
    # Unittest of Dnn Block forward, unittest of inference will be ommitted
    # since shareing same graph with training.
    feats = torch.rand(4, feat_len, self._dnn_block._input_dim)
    output = self._dnn_block(feats)
    self.assertEqual(output.shape[1], feat_len)
    self.assertEqual(output.shape[-1], self._dnn_block._hidden_dim)


class TestRnnBlock(unittest.TestCase):
  """ Unittest of RnnBlock """

  def setUp(self) -> None:
    config = {
        "rnn_type": "lstm",
        "hidden_size": 256,
        "num_layers": 2,
        "batch_first": True,
        "dropout": 0.0,
        "bidirectional": False
    }
    self._lstm_rnn_block = LstmRnnBlock(_input_dim=64, config=RnnBlockConfig(**config))
    config = {
        "rnn_type": "gru",
        "hidden_size": 512,
        "num_layers": 2,
        "batch_first": True,
        "dropout": 0.0,
        "bidirectional": False
    }
    self._gru_rnn_block = GruRnnBlock(_input_dim=128, config=RnnBlockConfig(**config))

  @parameterized.expand([(101,), (156,), (271,)])
  def test_lstm_rnn_block_forward(self, feat_len):
    # Unittest of LSTM Training graph
    feats = torch.rand(4, feat_len, self._lstm_rnn_block._input_size)
    output = self._lstm_rnn_block(feats)

    self.assertEqual(output.shape[1], feat_len)
    self.assertEqual(output.shape[-1], self._lstm_rnn_block.output_dim)

  @parameterized.expand([(101,), (156,), (271,)])
  def test_lstm_rnn_block_streaming_inference(self, feat_len):
    # Unittest of LSTM streaming inference graph
    self._lstm_rnn_block.train(False)
    feats = torch.rand(1, feat_len, self._lstm_rnn_block._input_size)
    cache = self._lstm_rnn_block.initialize_rnn_cache()
    non_stream_output = self._lstm_rnn_block(feats)

    # Streaming mode
    stream_output = []
    for frame_id in range(0, feat_len, 1):
      # Simulate streaming inference
      output, cache = self._lstm_rnn_block.inference(feats[:, frame_id:frame_id + 1, :], cache)
      stream_output.append(output)
    stream_output = torch.concat(stream_output, dim=1)

    # with maximum abs diff of precision with 2e-7
    glog.info(torch.max(stream_output - non_stream_output))
    self.assertTrue(torch.allclose(non_stream_output, stream_output, rtol=3e-5, atol=3e-7))

  @parameterized.expand([(101,), (156,), (271,)])
  def test_gru_rnn_block_forward(self, feat_len):
    # Unittest of GRU Training graph
    feats = torch.rand(4, feat_len, self._gru_rnn_block._input_size)
    output = self._gru_rnn_block(feats)

    self.assertEqual(output.shape[1], feat_len)
    self.assertEqual(output.shape[-1], self._gru_rnn_block.output_dim)

  @parameterized.expand([(101,), (156,), (271,)])
  def test_gru_rnn_block_streaming_inference(self, feat_len):
    # Unittest of GRU streaming inference graph
    self._gru_rnn_block.train(False)
    feats = torch.rand(1, feat_len, self._gru_rnn_block._input_size)
    cache = self._gru_rnn_block.initialize_rnn_cache()
    non_stream_output = self._gru_rnn_block(feats)

    # Streaming mode
    stream_output = []
    for frame_id in range(0, feat_len, 1):
      # Simulate streaming inference
      output, cache = self._gru_rnn_block.inference(feats[:, frame_id:frame_id + 1, :], cache)
      stream_output.append(output)
    stream_output = torch.concat(stream_output, dim=1)

    # with maximum abs diff of precision with 2e-7
    glog.info(torch.max(stream_output - non_stream_output))
    self.assertTrue(torch.allclose(non_stream_output, stream_output, rtol=3e-5, atol=3e-7))


class TestGruCrdnn(unittest.TestCase):
  """ Unittest of Crdnn with GRU RnnBlock """

  def setUp(self) -> None:
    config = {
        "cnn_block_config": {
            "num_layers": 3,
            "conv_type": "conv2d",
            "input_dim": 64,
            "in_channels_config": [1, 32, 32],
            "out_channels_config": [32, 32, 32],
            "kernel_configs": [(3, 3), (5, 5), (3, 3)],
            "stride_configs": [2, 2, 2],
            "dilation_config": [(1, 1), (2, 2), (1, 1)]
        },
        "dnn_block_config": {
            "num_layers": 1,
            "hidden_dim": 64
        },
        "rnn_block_config": {
            "rnn_type": "gru",
            "hidden_size": 128,
            "num_layers": 1
        }
    }
    self._crdnn = Crdnn(config=CrdnnConfig(**config))

  def test_gru_crdnn_model_config_check(self):
    # check params of model are configured right
    self.assertListEqual(self._crdnn._cnn_blocks._kernel_configs, [(3, 3), (5, 5), (3, 3)])
    self.assertListEqual(self._crdnn._cnn_blocks._dilation_config, [(1, 1), (2, 2), (1, 1)])
    self.assertListEqual(self._crdnn._cnn_blocks._in_channels_config, [1, 32, 32])
    self.assertEqual(self._crdnn._rnn_blocks._rnn_type, "gru")

    self.assertEqual(self._crdnn._rnn_blocks._hidden_size, 128)
    self.assertEqual(self._crdnn._rnn_blocks._batch_first, True)

  @parameterized.expand([(101,), (156,), (271,)])
  def test_gru_crdnn_forward(self, feat_len):
    # Unittest of erdnn training graph
    feats = torch.rand(4, feat_len, self._crdnn._cnn_blocks._input_dim)
    output = self._crdnn(feats)
    output = self._crdnn.compute_logits(output)

    self.assertEqual(self._crdnn._cnn_blocks.output_dim, self._crdnn._dnn_blocks._input_dim)
    self.assertEqual(self._crdnn._dnn_blocks.output_dim, 64)
    self.assertEqual(self._crdnn._rnn_blocks._input_size, self._crdnn._dnn_blocks.output_dim)
    self.assertEqual(self._crdnn._rnn_blocks.output_dim, 128)

    # Frame size maintained through blocks
    self.assertEqual(output.shape[1], feat_len)
    self.assertEqual(output.shape[-1], 2)

    # validation of logits
    self.assertTrue(torch.allclose(torch.sum(output, dim=-1), torch.ones(4, feat_len)))

  @parameterized.expand([(101,), (156,), (271,), (456,), (786,)])
  def test_gru_crdnn_streaming_inference(self, feat_len):
    # Unittest of crdnn streaming inference graph
    self._crdnn.train(False)
    feats = torch.rand(1, feat_len, self._crdnn._cnn_blocks._input_dim)
    cache = self._crdnn.initialize_cache()
    non_stream_output = self._crdnn(feats)
    non_stream_output = self._crdnn.compute_logits(non_stream_output)

    # Streaming mode
    stream_output = []
    for frame_id in range(0, feat_len, 1):
      # Simulate streaming inference
      output, cache = self._crdnn.inference(feats[:, frame_id:frame_id + 1, :], cache)
      stream_output.append(output)
    stream_output = torch.concat(stream_output, dim=1)

    # with maximum abs diff of precision with 2e-7
    glog.info(torch.max(stream_output - non_stream_output))
    self.assertTrue(torch.allclose(non_stream_output, stream_output, rtol=3e-5, atol=3e-7))

  @parameterized.expand([(101,), (156,), (271,), (456,), (786,)])
  def test_gru_crdnn_torchscript_export(self, feat_len):
    # Conv2d + GRU torchscript export check
    self._crdnn.train(False)
    torchscript_crdnn = torch.jit.script(self._crdnn)
    feats = torch.rand(1, feat_len, self._crdnn._cnn_blocks._input_dim)
    cache = torchscript_crdnn.initialize_cache()
    non_stream_output = torchscript_crdnn(feats)
    non_stream_output = torchscript_crdnn.compute_logits(non_stream_output)

    # Streaming mode
    stream_output = []
    for frame_id in range(0, feat_len, 1):
      # Simulate streaming inference
      output, cache = torchscript_crdnn.inference(feats[:, frame_id:frame_id + 1, :], cache)
      stream_output.append(output)
    stream_output = torch.concat(stream_output, dim=1)

    # with maximum abs diff of precision with 2e-7
    glog.info(torch.max(stream_output - non_stream_output))
    self.assertTrue(torch.allclose(non_stream_output, stream_output, rtol=3e-5, atol=3e-7))

  def test_gru_crdnn_model_quant(self):
    # Unittest of vad mode1 quantization
    self._crdnn.train(False)
    model_quant = self._crdnn

    # Ops fuse
    for layer_id in range(self._crdnn._cnn_blocks._num_layers):
      model_quant = torch.quantization.fuse_modules(model_quant, [
          "_cnn_blocks._cnn_layers.{}.0".format(layer_id),
          "_cnn_blocks._cnn_layers.{}.1".format(layer_id)
      ])

    # Dynamic quantize Linear and RNN layer
    model_quant = torch.quantization.quantize_dynamic(
        model_quant,  # the original model 
        {torch.nn.Linear, torch.nn.GRU},  # a set of layers to dynamically quantize 
        dtype=torch.qint8)

    # 524KB orig_model -> 280kB int8_model
    ts_crdnn = torch.jit.script(self._crdnn)
    ts_int8_crdnn = torch.jit.script(model_quant)


class TestLstmCrdnn(unittest.TestCase):
  """ Unittest of Crdnn with GRU RnnBlock """

  def setUp(self) -> None:
    config = {
        "cnn_block_config": {
            "num_layers": 3,
            "conv_type": "conv1d",
            "input_dim": 64,
            "in_channels_config": [64, 128, 256],
            "out_channels_config": [128, 256, 512],
            "kernel_configs": [5, 5, 5],
            "stride_configs": None,
            "dilation_config": [1, 1, 1]
        },
        "dnn_block_config": {
            "num_layers": 4,
            "hidden_dim": 64
        },
        "rnn_block_config": {
            "rnn_type": "lstm",
            "hidden_size": 512,
            "num_layers": 2
        }
    }
    self._crdnn = Crdnn(config=CrdnnConfig(**config))

  def test_lstm_crdnn_model_config_check(self):
    # check params of model are configured right
    self.assertListEqual(self._crdnn._cnn_blocks._kernel_configs, [5, 5, 5])
    self.assertListEqual(self._crdnn._cnn_blocks._dilation_config, [1, 1, 1])
    self.assertListEqual(self._crdnn._cnn_blocks._in_channels_config, [64, 128, 256])
    self.assertEqual(self._crdnn._rnn_blocks._rnn_type, "lstm")

    self.assertEqual(self._crdnn._rnn_blocks._hidden_size, 512)
    self.assertEqual(self._crdnn._rnn_blocks._batch_first, True)

  @parameterized.expand([(101,), (156,), (271,)])
  def test_lstm_crdnn_forward(self, feat_len):
    # Unittest of erdnn training graph
    feats = torch.rand(4, feat_len, self._crdnn._cnn_blocks._input_dim)
    output = self._crdnn(feats)
    output = self._crdnn.compute_logits(output)

    self.assertEqual(self._crdnn._cnn_blocks.output_dim, self._crdnn._dnn_blocks._input_dim)
    self.assertEqual(self._crdnn._dnn_blocks.output_dim, 64)
    self.assertEqual(self._crdnn._rnn_blocks._input_size, self._crdnn._dnn_blocks.output_dim)
    self.assertEqual(self._crdnn._rnn_blocks.output_dim, 512)

    # Frame size maintained through blocks
    self.assertEqual(output.shape[1], feat_len)
    self.assertEqual(output.shape[-1], 2)

    # validation of logits
    self.assertTrue(torch.allclose(torch.sum(output, dim=-1), torch.ones(4, feat_len)))

  @parameterized.expand([(101,), (156,), (271,), (456,), (786,)])
  def test_lstm_crdnn_streaming_inference(self, feat_len):
    # Unittest of crdnn streaming inference graph
    self._crdnn.train(False)
    feats = torch.rand(1, feat_len, self._crdnn._cnn_blocks._input_dim)
    cache = self._crdnn.initialize_cache()
    non_stream_output = self._crdnn(feats)
    non_stream_output = self._crdnn.compute_logits(non_stream_output)

    # Streaming mode
    stream_output = []
    for frame_id in range(0, feat_len, 1):
      # Simulate streaming inference
      output, cache = self._crdnn.inference(feats[:, frame_id:frame_id + 1, :], cache)
      stream_output.append(output)
    stream_output = torch.concat(stream_output, dim=1)

    # with maximum abs diff of precision with 2e-7
    glog.info(torch.max(stream_output - non_stream_output))
    self.assertTrue(torch.allclose(non_stream_output, stream_output, rtol=3e-5, atol=3e-7))

  @parameterized.expand([(101,), (156,), (271,), (456,), (786,)])
  def test_lstm_crdnn_torchscript_export(self, feat_len):
    # Conv1d + LSTM torchscript export check
    self._crdnn.train(False)
    torchscript_crdnn = torch.jit.script(self._crdnn)
    feats = torch.rand(1, feat_len, self._crdnn._cnn_blocks._input_dim)
    cache = torchscript_crdnn.initialize_cache()
    non_stream_output = torchscript_crdnn(feats)
    non_stream_output = torchscript_crdnn.compute_logits(non_stream_output)

    # Streaming mode
    stream_output = []
    for frame_id in range(0, feat_len, 1):
      # Simulate streaming inference
      output, cache = torchscript_crdnn.inference(feats[:, frame_id:frame_id + 1, :], cache)
      stream_output.append(output)
    stream_output = torch.concat(stream_output, dim=1)

    # with maximum abs diff of precision with 2e-7
    glog.info(torch.max(stream_output - non_stream_output))
    self.assertTrue(torch.allclose(non_stream_output, stream_output, rtol=3e-5, atol=3e-7))

  def test_lstm_crdnn_model_quant(self):
    # Unittest of vad mode1 quantization
    self._crdnn.train(False)
    model_quant = self._crdnn

    # Ops fuse
    for layer_id in range(self._crdnn._cnn_blocks._num_layers):
      model_quant = torch.quantization.fuse_modules(model_quant, [
          "_cnn_blocks._cnn_layers.{}.0".format(layer_id),
          "_cnn_blocks._cnn_layers.{}.1".format(layer_id)
      ])

    # Dynamic quantize Linear and RNN layer
    model_quant = torch.quantization.quantize_dynamic(
        model_quant,  # the original model 
        {torch.nn.Linear, torch.nn.LSTM},  # a set of layers to dynamically quantize 
        dtype=torch.qint8)

    # 17MB orig_model -> 6.6MB int8_model
    ts_crdnn = torch.jit.script(self._crdnn)
    ts_int8_crdnn = torch.jit.script(model_quant)


if __name__ == "__main__":
  unittest.main()
