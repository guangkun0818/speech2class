# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.03.04
""" Unittest of Crdnn Onnx export """

import torch
import unittest

import onnxruntime as ort

from model.vad_model.crdnn import CrdnnConfig, Crdnn
from model.vad_model.onnx_export.crdnn_onnx import CrdnnOnnxInit, CrdnnOnnxInference


class TestCrdnnOnnxExport(unittest.TestCase):
  """ Unittest of both CrdnnOnnxInit and CrdnnOnnxInference, 
        using tiny Crdnn setting, 277k in quant torchscript
    """

  def setUp(self) -> None:
    self._cache_nums = 3 + 2  # 3 layers Cnn, 2 for Rnn cache
    self._config = {
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

    self._crdnn_init = CrdnnOnnxInit(config=CrdnnConfig(**self._config))
    self._crdnn_infer = CrdnnOnnxInference(config=CrdnnConfig(**self._config))

    self._dummy_feats = torch.rand(1, 100, 64)
    self._dummy_cache = self._crdnn_init.initialize_cache()

  def test_crdnn_init_onnx_export(self):
    # Unittest of CrdnnOnnxInit export
    # Onnx export require args, use dummy_input 1 of it. However
    # dummy_input cant not be traced within onnx model, therefore
    # input_feed of OrtSession should be empty.
    output_names = ["cache_%d" % i for i in range(5)]
    torch.onnx.export(self._crdnn_init,
                      args=1,
                      f="test_logs/crdnn_init.onnx",
                      verbose=True,
                      input_names=None,
                      output_names=output_names)

    torch_output = self._crdnn_init(1)  # dummy_input

    ort_session = ort.InferenceSession("test_logs/crdnn_init.onnx")
    # Output of onnx were flattened as list[*cnn_cache, *rnn_cache]
    onnx_output = ort_session.run(None, input_feed={})

    # Cache check
    for th_out, onnx_out in zip([*torch_output[0], *torch_output[1]], onnx_output):
      self.assertTrue(torch.allclose(th_out, torch.Tensor(onnx_out)))

  def test_crdnn_infer_onnx_export(self):
    # Unittest of CrdnnOnnxInit export
    # Flatten feats + tuple(*caches) as the whole list with ["feats", "cache_0", ...]
    self._crdnn_infer.train(False)
    input_names = ["feats"] + ["cache_%d" % i for i in range(5)]
    output_names = ["logits"] + ["next_cache_%d" % i for i in range(5)]

    torch.onnx.export(self._crdnn_infer, (self._dummy_feats, self._dummy_cache),
                      "test_logs/crdnn_inference.onnx",
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names)

    torch_logits, torch_cache = self._crdnn_infer(self._dummy_feats, self._dummy_cache)

    ort_session = ort.InferenceSession("test_logs/crdnn_inference.onnx")

    # Output of onnx were flattened as list[*cnn_cache, *rnn_cache]
    dummy_cache = {}
    for i, cache in enumerate([*self._dummy_cache[0], *self._dummy_cache[1]]):
      dummy_cache["cache_{}".format(i)] = cache.numpy()

    # Outputs of Onnx: ["logits", next_cache_0, ...]
    onnx_output = ort_session.run(["logits"] + ["next_cache_%d" % i for i in range(5)],
                                  input_feed={
                                      "feats": self._dummy_feats.numpy(),
                                      **dummy_cache
                                  })

    for th_out, onnx_out in zip([torch_logits] + [*torch_cache[0], *torch_cache[1]], onnx_output):
      self.assertTrue(torch.allclose(th_out, torch.Tensor(onnx_out), atol=1e-7, rtol=1e-4))


if __name__ == "__main__":
  unittest.main()
