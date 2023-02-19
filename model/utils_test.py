# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.19
""" Unittest of utilities """

import glog
import torch
import unittest

from parameterized import parameterized
from model.utils import Metric, MetricConfig


class TestMetrics(unittest.TestCase):
    """ Unittest of Metrics """

    def setUp(self) -> None:
        vpr_config = {"task": "VPR", "top_ks": [1, 5]}
        self._vpr_metrics = Metric(config=MetricConfig(**vpr_config))
        vad_config = {"task": "VAD", "top_ks": None}
        self._vad_metrics = Metric(config=MetricConfig(**vad_config))

    @parameterized.expand([(128,), (5,)])
    def test_metric_topks_vpr_task(self, num_classes):
        # Metrics processing for vpr task
        preds = torch.rand(32, num_classes)
        labels = torch.randint(0, num_classes, (32,))
        glog.info("Predicts: {}".format(preds.shape))
        glog.info("Labels: {}".format(labels.shape))
        metrics = self._vpr_metrics(preds=preds, labels=labels)
        for key in metrics:
            glog.info(" {}: {}".format(key, metrics[key]))

    @parameterized.expand([(
        torch.Tensor([[1, 0]]),
        torch.Tensor([[[0.4498, 0.7822], [0.6885, 0.3769]]]),
        torch.Tensor([1]),
    ),
                           (
                               torch.Tensor([[0, 0, 0], [1, 1, 0]]),
                               torch.Tensor([[[0.8438,
                                               0.6637], [0.0040, 0.9138],
                                              [0.2288, 0.9692]],
                                             [[0.5117,
                                               0.8571], [0.3471, 0.5770],
                                              [0.4606, 0.1697]]]),
                               torch.Tensor([0.66666666666666]),
                           )])
    def test_metric_topks_vad_task(self, label, preds, acc_true_value):
        # Metrics of Vad task
        acc = self._vad_metrics._vad_accuarcy(preds, label)
        self.assertEqual(acc, acc_true_value)
        acc = self._vad_metrics(preds, label)
        self.assertDictEqual(acc, {"acc": acc_true_value})


if __name__ == "__main__":
    unittest.main()
