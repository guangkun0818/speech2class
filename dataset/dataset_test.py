# Author: Xiaoyue Yang
# Email: 609946862@qq.com
# Created on 2023.02.09
""" Unittest of dataset """

import glog
import unittest
import torch

from torch.utils.data import DataLoader
from dataset.dataset import VprTrainDataset, VprEvalDataset, collate_fn


class VprDatasetTest(unittest.TestCase):
    """ Unittest of Trainataset, Evaldataset for VPR Tasks """

    def setUp(self) -> None:
        self._config = {
            "train_data": "sample_data/vpr_eval_data.json",
            "eval_data": "sample_data/vpr_eval_data.json",
            "noise_data": "sample_data/noise_data.json",
            "batch_size": 32,
            "chunk_size": 350,
            "num_classes": 10,
            "feat_type": "fbank",
            "feat_config": {
                "num_mel_bins": 64,
                "frame_length": 25,
                "frame_shift": 10,
                "dither": 0.0,
                "samplerate": 16000
            },
            "add_noise_proportion": 1.0,
            "add_noise_config": {
                "min_snr_db": 10,
                "max_snr_db": 50,
                "max_gain_db": 300.0,
            }
        }

        self._train_dataset = VprTrainDataset(self._config)
        self._eval_dataset = VprEvalDataset(self._config)

    def test_vpr_train_dataset(self):
        # Unittest of train dataset
        glog.info("VprTrainDataset unittest...")

        count = 0
        dataloader = DataLoader(dataset=self._train_dataset,
                                batch_size=256,
                                num_workers=1,
                                collate_fn=collate_fn)

        for i, batch in enumerate(dataloader):
            count += 1
            glog.info("Feats size {}.".format(batch["feat"].shape))
            glog.info("Speak_ids size {}.".format(batch["label"]))

        glog.info("Total iter {} with batch size {}.".format(count, 256))

    def test_vpr_eval_dataset(self):
        # Unittest of eval dataset
        glog.info("VprEvalDataset unittest....")

        for batch_size in range(1, 11):
            count = 0
            dataloader = DataLoader(dataset=self._eval_dataset,
                                    batch_size=batch_size,
                                    num_workers=1,
                                    collate_fn=collate_fn)
            for i, batch in enumerate(dataloader):
                count += 1
                glog.info("Feats size {}.".format(batch["feat"].shape))
                glog.info("Speak_ids size {}.".format(batch["label"]))

            glog.info("Total iter {} with batch size {}.".format(
                count, batch_size))


if __name__ == "__main__":
    unittest.main()
