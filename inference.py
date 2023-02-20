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
    ...