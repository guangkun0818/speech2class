# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.19
""" Inference of Both Vad and Vpr tasks """

import collections
import os
import sys
import gflags
import glob
import glog
import logging
import torch
import shutil
import yaml

import torch.nn.functional as F
import pytorch_lightning as pl

from enum import Enum, unique
from typing import List, Dict
from torch.utils.data import DataLoader

from dataset.dataset import VadTestDataset
from vpr_task import VprTask
from vad_task import VadTask
from tools.model_average import model_average

FLAGS = gflags.FLAGS

gflags.DEFINE_string("inference_config", "config/inference/infer.yaml",
                     "YAML configuration of inference.")


class VprInference(VprTask):
  """ TODO: Build Vpr Inference inherited from VprTask """
  ...


class VadInference(VadTask):
  """ Build Vad Inference inherited from VadTask """

  def __init__(self, task_config, infer_config) -> None:
    # Inference Initialize
    super(VadInference, self).__init__(config=task_config)

    # Export path of vad model, quantized torchscript and onnx
    self._export_quant_model = infer_config["export_quant_model"]
    self._export_onnx_model = infer_config["export_onnx_model"]
    self._model_export_path = infer_config["result_path"]

    self._test_data = infer_config["test_data"]
    # Load torchscript frontend
    self._frontend_model = os.path.join(task_config["task"]["export_path"],
                                        "frontend.script")

    self._post_process, self._post_process_config = self.post_process_factory(
        infer_config["post_process"])

    # Track down test acc
    self._total_metrics = {"acc": [], "far": [], "frr": []}

  @property
  def device(self):
    # Specify device for model export
    return next(self._vad_model.parameters()).device.type

  def export_quant_model(self):
    # Vad model quantization and export
    self._vad_model.vad_model.train(False)
    model_quant = self._vad_model.vad_model

    # Ops fuse
    for layer_id in range(self._vad_model.vad_model._cnn_blocks._num_layers):
      model_quant = torch.quantization.fuse_modules(model_quant, [
          "_cnn_blocks._cnn_layers.{}.0".format(layer_id),
          "_cnn_blocks._cnn_layers.{}.1".format(layer_id)
      ])

    # Dynamic quantize Linear and RNN layer
    model_quant = torch.quantization.quantize_dynamic(
        model_quant,  # the original model 
        {torch.nn.Linear, torch.nn.GRU, torch.nn.LSTM
        },  # a set of layers to dynamically quantize 
        dtype=torch.qint8)

    vad_model_int8 = torch.jit.script(model_quant)

    vad_model_int8.save(
        os.path.join(self._model_export_path,
                     "{}_int8.script".format(model_quant.__class__.__name__)))

  def export_onnx_model(self, chunk_size):
    # Cnn_cache: num_layers, Rnn_cache, 2.
    num_cache = self._vad_model.vad_model._cnn_blocks._num_layers + 2
    # Onnx vad export for on-device deploy
    self._vad_model.vad_model.train(False)
    model = self._vad_model.vad_model

    # Export cache_init model.
    model.forward = self._vad_model.vad_model.initialize_cache
    # Onnx will flatten original cache into Tuple[cache_0, cache_1, ...]
    output_names = ["cache_%d" % i for i in range(num_cache)]
    init_model_path = os.path.join(
        self._model_export_path,
        "{}_init.onnx".format(model.__class__.__name__))

    # Args is required by onnx export, set as empty for init
    torch.onnx.export(model,
                      args=(),
                      f=init_model_path,
                      verbose=True,
                      input_names=None,
                      output_names=output_names)

    # Export inference model
    model.forward = model.inference
    dummy_feats = torch.rand(1, chunk_size, 64)
    dummy_cache = model.initialize_cache()

    input_names = ["feats"] + ["cache_%d" % i for i in range(num_cache)]
    output_names = ["logits"] + ["next_cache_%d" % i for i in range(num_cache)]
    infer_model_path = os.path.join(
        self._model_export_path,
        "{}_inference.onnx".format(model.__class__.__name__))

    torch.onnx.export(model,
                      args=(dummy_feats, dummy_cache),
                      f=infer_model_path,
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names)

  def test_dataloader(self):
    """ Testdataloader Impl """
    dataset = VadTestDataset(self._test_data, self._frontend_model)
    # frontend.script conflict with num_works > 1. This happends when "ddp_spawn"
    # specified, which should replace with "ddp". However, MacBook seems only
    # ddp_spawn supported.
    dataloader = DataLoader(
        dataset=dataset, batch_size=1,
        num_workers=4)  # only support batch_size = 1 currently
    return dataloader

  def post_process_factory(self, config: Dict):
    # Return post_process method with given config like: ["sliding window"]
    # config should be consist with method name.
    return getattr(self, config["type"]), config["config"]

  def no_post_process(self, batch, dummy_conf=None):
    # No-post-processing, return with raw model output
    stream_output = []  # Streaming output pool

    feat = batch["feat"]
    cache = self._vad_model.initialize_cache()

    feat_len = batch["feat"].shape[1]
    # Inference in streaming mode, simulated
    for frame_id in range(0, feat_len, 1):
      # Simulate streaming inference
      logits, cache = self._vad_model.inference(
          feat[:, frame_id:frame_id + 1, :], cache)
      stream_output.append(logits)
    predictions = torch.concat(stream_output, dim=1)

    return predictions

  def sliding_window(self,
                     batch,
                     window_size=10,
                     speech_start_thres=0.5,
                     speech_end_thres=0.9) -> torch.Tensor:
    """ Inference with sliding_window post process
            Args:
                batch: Raw inference results from vad model.
                window_size: the length of sliding window
                speech_start_thres: threshold to detect speech start when 
                                    has_speech_start is false.
                speech_end_thres: threshold to detect speech start when 
                                    has_speech_start is true.
            return:
                Frame-level smoothed prediction.
        """
    post_process_output = []

    has_speech_start = False  # Has speech started yet? work as switch
    ring_buffer = collections.deque(maxlen=window_size)

    feat = batch["feat"]
    cache = self._vad_model.initialize_cache()

    feat_len = batch["feat"].shape[1]
    for frame_id in range(0, feat_len, 1):
      # Simulate streaming inference
      logits, cache = self._vad_model.inference(
          feat[:, frame_id:frame_id + 1, :], cache)

      pred_label = torch.argmax(logits, dim=-1, keepdim=True)
      post_process_output.append(torch.zeros(1, 1, 1))

      if len(ring_buffer) < window_size - 1:
        # Keep filling in window if not full
        ring_buffer.append(pred_label)
        continue

      if not has_speech_start:
        # If speech has not started yet, wait for the onset status to appear
        ring_buffer.append(pred_label)
        num_speech_fs = ring_buffer.count(1)
        if num_speech_fs > window_size * speech_start_thres:
          onset = frame_id - len(ring_buffer) + 1  # Front frame of the window
          has_speech_start = True
          ring_buffer.clear()
      else:
        # If speech has started, wait for the offset status to appear
        ring_buffer.append(pred_label)
        num_non_speech_fs = ring_buffer.count(0)
        if num_non_speech_fs > window_size * speech_end_thres:
          offset = frame_id  # End frame of the window
          has_speech_start = False
          ring_buffer.clear()
          for i in range(onset, offset):
            post_process_output[i] = torch.ones(1, 1, 1)

    if has_speech_start:
      # If there is no offset until the end of the audio
      # set the status from onset to the end of audio as speech(1)
      for i in range(onset, feat_len):
        post_process_output[i] = torch.ones(1, 1, 1)

    predictions = torch.concat(post_process_output, dim=1).squeeze(-1)
    predictions = F.one_hot(
        predictions.to(device=feat.device).to(dtype=torch.int64))

    return predictions

  def test_step(self, batch, batch_idx):
    # Return result with given post-process.
    predictions = self._post_process(batch, **self._post_process_config)

    batch_metrics = self._metric.vad_infer_metrics(predictions, batch["label"])
    glog.info("{}: acc: {:.3f} far: {:.3f} frr: {:.3f}".format(
        batch["utt"][0], batch_metrics["acc"] * 100, batch_metrics["far"] * 100,
        batch_metrics["frr"] * 100))

    for met_t in batch_metrics:
      assert met_t in self._total_metrics
      self._total_metrics[met_t].append(batch_metrics[met_t])

  def on_test_start(self) -> None:
    # If export_quant_model is specified, export torchscript model for deploy
    if self._export_quant_model:
      # Model should move to CPU for export
      if self.device == "cuda":
        self._vad_model.cpu()
        self.export_quant_model()
        self._vad_model.cuda()
      else:
        self.export_quant_model()

    # If export_onnx_model is specified, export onnx model for deploy
    if self._export_onnx_model["do_export"]:
      # Model should move to CPU for export
      if self.device == "cuda":
        self._vad_model.cpu()
        self.export_onnx_model(self._export_onnx_model["chunk_size"])
        self._vad_model.cuda()
      else:
        self.export_onnx_model(self._export_onnx_model["chunk_size"])

  def on_test_end(self) -> None:
    num_testcases = len(self._total_metrics["acc"])

    glog.info("Total ACC: {:.3f} FAR: {:.3f} FRR: {:.3f}".format(
        sum(self._total_metrics["acc"]) / num_testcases * 100,
        sum(self._total_metrics["far"]) / num_testcases * 100,
        sum(self._total_metrics["frr"]) / num_testcases * 100,
    ))


@unique
class InferFactory(Enum):
  """ Inference task Factory, build selected inference from config """
  VPR = VprInference
  VAD = VadInference


def inference():
  # ------ Inference set up and logging initialization ------
  FLAGS(sys.argv)
  with open(FLAGS.inference_config, 'r') as config_yaml:
    infer_config = yaml.load(config_yaml.read(), Loader=yaml.FullLoader)
    task_config = yaml.load(open(infer_config["task_config"], 'r').read(),
                            Loader=yaml.FullLoader)

  # Set up load and export path
  TASK_TYPE = task_config["task"]["type"]
  INFER_EXPORT_PATH = infer_config["result_path"]

  # Backup inference config and setup logging
  glog.info("{} inference setting up ....".format(TASK_TYPE))
  os.makedirs(INFER_EXPORT_PATH, exist_ok=True)
  handler = logging.FileHandler(os.path.join(INFER_EXPORT_PATH,
                                             "inference.log"))
  shutil.copyfile(
      FLAGS.inference_config,
      os.path.join(INFER_EXPORT_PATH, os.path.basename(FLAGS.inference_config)))
  glog.init()
  glog.logger.addHandler(handler)
  glog.info(infer_config)

  # ----- Inference -----
  if infer_config["chkpt_aver"] == True:
    model_average(
        os.path.join(task_config["task"]["export_path"], "checkpoints"))
    CHKPT_PATH = os.path.join(task_config["task"]["export_path"], "checkpoints",
                              "averaged.chkpt")
  else:
    assert infer_config[
        "chkpt_name"], "Please specify chkpt_name if chkpt_aver not applied."
    CHKPT_PATH = os.path.join(task_config["task"]["export_path"], "checkpoints",
                              infer_config["chkpt_name"])
  task_inference = InferFactory[TASK_TYPE].value.load_from_checkpoint(
      CHKPT_PATH, task_config=task_config,
      infer_config=infer_config)  # Build from InferFactory

  trainer = pl.Trainer(logger=False, **infer_config["trainer"])
  trainer.test(task_inference)


if __name__ == "__main__":
  inference()
