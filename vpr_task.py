# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.19
""" VPR Training scripts """

import copy
import glog
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from dataset.frontend.frontend import EcapaFrontend, KaldiWaveFeature
from dataset.dataset import VprTrainDataset, VprEvalDataset, collate_fn
from model.embedding_model.embedding_model import EmbeddingModel
from model.loss.loss import Loss
from model.utils import Metric, MetricConfig
from optimizer.optim_setup import OptimSetup