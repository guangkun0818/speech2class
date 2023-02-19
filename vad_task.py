# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.19
""" VAD Training scripts """

import copy
import glog
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from dataset.frontend.frontend import EcapaFrontend, KaldiWaveFeature
from dataset.dataset import VadTrainDataset, VadEvalDataset, collate_fn
from model.vad_model.vad_model import VadModel
from model.loss.loss import Loss
from model.utils import Metric, MetricConfig
from optimizer.optim_setup import OptimSetup


class VadTask(pl.LightningModule):
    """ Build VAD task from yaml config """
    ...
