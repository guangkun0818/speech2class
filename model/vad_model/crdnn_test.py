# Author: Xiaoyue Yang
# Email: 609946862@qq.com
# Created on 2023.02.12
""" Unitest of Crdnn """

import glog
import unittest
import torch
from parameterized import parameterized
from model.vad_model.crdnn import Conv1dCnnBlock, Conv2dCnnBlock, CnnBlockConfig
from model.vad_model.crdnn import DnnBlock, DnnBlockConfig
# from model, vad model,crdnn import LstmRNNBlock, GruRNNElock, RNNBlockconfig
# from model.vad model, crdnn import CRDNN, CRDNNConfig


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
        self._cnn_block_1d = Conv1dCnnBlock(config=CnnBlockConfig(
            **cnn_1d_config))

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
        self._cnn_block_2d = Conv2dCnnBlock(config=CnnBlockConfig(
            **cnn_2d_config))

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
        self.assertListEqual(self._cldnn._kernel_configs, [(1, 3), (1, 5),
                                                           (1, 3)])
        self.assertListEqual(self._cnn_block_2d._kernel_configs,
                             [(3, 3), (5, 3), (3, 3)])

    @parameterized.expand([(101,), (156,), (271,)])
    def test_cnn_block_1d_forward(self, feat_len):
        # Unitest of Cnn Block forward with Conv1d. default feats_dim = 64
        feats = torch.rand(4, feat_len, self._cnn_block_1d._input_dim)
        output = self._cnn_block_1d(feats)
        self.assertEqual(output.shape[1], feat_len)
        self.assertEqual(output.shape[2],
                         self._cnn_block_1d._out_channels_config[-1])
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
            output, cache = self._cnn_block_1d.inference(
                feats[:, frame_id:frame_id + 1, :], cache)
            stream_output.append(output)
        stream_output = torch.concat(stream_output, dim=1)
        self.assertEqual(stream_output.shape[1], feat_len)

        # with maximum abs diff of precision with 2e-7
        glog.info(torch.max(stream_output - non_steam_output))
        self.assertTrue(
            torch.allclose(non_steam_output,
                           stream_output,
                           rtol=3e-5,
                           atol=3e-7))

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
            output, cache = self._cnn_block_2d.inference(
                feats[:, frame_id:frame_id + 1, :], cache)
            stream_output.append(output)
        stream_output = torch.concat(stream_output, dim=1)
        self.assertEqual(stream_output.shape[1], feat_len)

        # with maximum abs diff of precision with 2e-7
        glog.info(torch.max(stream_output - non_steam_output))
        self.assertTrue(
            torch.allclose(non_steam_output,
                           stream_output,
                           rtol=3e-5,
                           atol=3e-7))

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
            output, cache = self._cldnn.inference(
                feats[:, frame_id:frame_id + 1, :], cache)
            stream_output.append(output)
        stream_output = torch.concat(stream_output, dim=1)
        self.assertEqual(stream_output.shape[1], feat_len)

        # with maximum abs diff of precision with 2e-7
        glog.info(torch.max(stream_output - non_steam_output))
        self.assertTrue(
            torch.allclose(non_steam_output,
                           stream_output,
                           rtol=3e-5,
                           atol=3e-7))


class TestDnnBlock(unittest.TestCase):
    """ Unittest of DnnBLock """

    def setUp(self) -> None:
        config = {"num_layers": 4, "hidden_dim": 64}
        self._dnn_block = DnnBlock(_input_dim=320,
                                   config=DnnBlockConfig(**config))

    @parameterized.expand([(101,), (156,), (271,)])
    def test_dnn_block_forward(self, feat_len):
        # Unittest of Dnn Block forward, unittest of inference will be ommitted
        # since shareing same graph with training.
        feats = torch.rand(4, feat_len, self._dnn_block._input_dim)
        output = self._dnn_block(feats)
        self.assertEqual(output.shape[1], feat_len)
        self.assertEqual(output.shape[-1], self._dnn_block._hidden_dim)


if __name__ == "__main__":
    unittest.main()
