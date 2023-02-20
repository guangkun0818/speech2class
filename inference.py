# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.19
""" Inference of Both Vad and Vpr tasks """

import os
import sys
import gflags
import glob
import glog
import logging
import torch
import shutil
import yaml

import pytorch_lightning as pl

from enum import Enum, unique
from typing import List
from torch.utils.data import DataLoader

from dataset.dataset import VadTestDataset
from vpr_task import VprTask
from vad_task import VadTask

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

        self._test_data = infer_config["test_data"]
        # Load torchscript frontend
        self._frontend_model = os.path.join(task_config["task"]["export_path"],
                                            "frontend.script")
        # TODO: Impl post_processing
        self._post_process = self.post_process_factory(
            infer_config["post_process"])

        self._acc = []  # Track down test acc

    def post_process_factory(self, config: List[str]):
        # TODO: Return post_process method with given config like: ["sliding window"]
        ...

    def test_dataloader(self):
        """ Testdataloader Impl """
        dataset = VadTestDataset(self._test_data, self._frontend_model)
        # TODO: Fix frontend.script conflict with num_works > 1
        # This happends when "ddp_spawn" specified, which should replace with "ddp".
        # However, MacBook seems only ddp_spawn supported.
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1)  # only support batch_size = 1 currently
        return dataloader

    def test_step(self, batch, batch_idx):
        # Inference in streaming mode, simulated
        stream_output = []  # Streaming output pool

        feat = batch["feat"]
        cache = self._vad_model.initialize_cache()

        feat_len = batch["feat"].shape[1]
        for frame_id in range(0, feat_len, 1):
            # Simulate streaming inference
            logits, cache = self._vad_model.inference(
                feat[:, frame_id:frame_id + 1, :], cache)
            stream_output.append(logits)
        predictions = torch.concat(stream_output, dim=1)

        metrics = self._metric(predictions, batch["label"])
        glog.info(metrics)
        self._acc.append(metrics["acc"])

    def on_test_end(self) -> None:
        glog.info("Total Acc: {}".format(sum(self._acc) / len(self._acc)))


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
    CHKPT_PATH = os.path.join(task_config["task"]["export_path"], "checkpoints",
                              infer_config["chkpt_name"])
    INFER_EXPORT_PATH = infer_config["result_path"]

    # Backup inference config and setup logging
    glog.info("{} inference setting up ....".format(TASK_TYPE))
    os.makedirs(INFER_EXPORT_PATH, exist_ok=True)
    handler = logging.FileHandler(
        os.path.join(INFER_EXPORT_PATH, "inference.log"))
    shutil.copyfile(
        FLAGS.inference_config,
        os.path.join(INFER_EXPORT_PATH,
                     os.path.basename(FLAGS.inference_config)))
    glog.init()
    glog.logger.addHandler(handler)
    glog.info(infer_config)

    # ----- Inference -----
    task_inference = InferFactory[TASK_TYPE].value.load_from_checkpoint(
        CHKPT_PATH, task_config=task_config,
        infer_config=infer_config)  # Build from InferFactory

    trainer = pl.Trainer(logger=False, **infer_config["trainer"])
    trainer.test(task_inference)


if __name__ == "__main__":
    inference()
