# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.19
""" VPR and VAD Training Entrypoint """

import os
import sys
import torch
import yaml
import gflags
import glog
import logging
import random
import shutil
import pytorch_lightning as pl

from enum import Enum, unique
from pytorch_lightning import loggers as pl_loggers

import callbacks.callbacks as callbacks
from vpr_task import VprTask
from vad_task import VadTask

FLAGS = gflags.FLAGS

gflags.DEFINE_string("config_path", "config/training/resnet18.yaml", "YAML configuration of Task")


@unique
class TaskFactory(Enum):
  """ Task Factory, build selected task from config """
  VPR = VprTask
  VAD = VadTask


def run_task():
  torch.manual_seed(1234)
  random.seed(1234)  # For reproducibility

  # Task logging initialization
  FLAGS(sys.argv)
  with open(FLAGS.config_path, 'r') as config_yaml:
    config = yaml.load(config_yaml.read(), Loader=yaml.FullLoader)

  TASK_TYPE = config["task"]["type"]
  TASK_NAME = config["task"]["name"]
  TASK_EXPORT_PATH = config["task"]["export_path"]

  os.makedirs(TASK_EXPORT_PATH, exist_ok=True)
  handler = logging.FileHandler(os.path.join(TASK_EXPORT_PATH, "run.log"))
  glog.init()
  glog.logger.addHandler(handler)

  # Initialize Task
  glog.info("{} Task building....".format(TASK_TYPE))
  shutil.copyfile(FLAGS.config_path,
                  os.path.join(TASK_EXPORT_PATH, os.path.basename(FLAGS.config_path)))

  glog.info(config)

  # ----- Task Build -----
  # Setup pretrained-model if finetune/base_model is set
  if config["finetune"]["base_model"]:
    # If base_model of finetune is set, then finetune from base_model
    task = TaskFactory[TASK_TYPE].value.load_from_checkpoint(config["finetune"]["base_model"],
                                                             config=config)
  else:
    task = TaskFactory[TASK_TYPE].value(config)  # Build from TaskFactory

  # CallBacks function setup
  chkpt_filename = TASK_NAME + "-{epoch}-{val_loss:.2f}" + "-{%s:.2f}" % config["callbacks"][
      "model_chkpt_config"]["monitor"]

  chkpt_callback = callbacks.ModelCheckpoint(
      dirpath=os.path.join(TASK_EXPORT_PATH, "checkpoints"),
      filename=chkpt_filename,
      **config["callbacks"]["model_chkpt_config"])  # Callbacks save chkpt of model.

  lr_monitor = callbacks.LearningRateMonitor(logging_interval='step')
  callback_funcs = [chkpt_callback, lr_monitor]

  # Export frontend as torchscript for deployment
  frontend_save = callbacks.FrontendExport(save_dir=TASK_EXPORT_PATH)
  callback_funcs.append(frontend_save)

  tb_logger = pl_loggers.TensorBoardLogger(TASK_EXPORT_PATH)

  # Setup trainer
  trainer = pl.Trainer(**config["trainer"],
                       logger=tb_logger,
                       log_every_n_steps=2,
                       callbacks=callback_funcs)
  # If resume set, resume training from specific model.
  trainer.fit(task, ckpt_path=config["resume"])


if __name__ == "__main__":
  run_task()
