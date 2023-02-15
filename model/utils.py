# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.15
""" Utilities for training, metric specificly """

import dataclasses
import glog
import torch
import torch.nn as nn

from typing import List, Tuple


@dataclasses.dataclass
class MetricConfig:
    """ Config of metrics, supporting both Vad and Vpr tasks. """
    task: str = "VPR"
    top_ks: Tuple[int] = (1, 5)
