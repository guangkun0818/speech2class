# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.09
""" Make Torch Dataset for VPR and VAD """

import abc
import glog
import json
import random
import torchaudio
import torch
import numpy as np
import h5py

from torch.utils.data import Dataset
from torch.utils.data import Sampler

import dataset.frontend.data_augmentation as data_augmentation
from dataset.frontend.frontend import EcapaFrontend, KaldiWaveFeature


class BaseDataset(Dataset):
    """ Base Dataset to inherit for train, eval, test """

    def __init__(self,
                 dataset_json,
                 noiseset_json=None,
                 min_dur_filter=0.0) -> None:
        """ Args: Please refer to sample_data for JSON setting.
                dataset_json: JSON file of data
                noiseset_json: JSON file of noise data
                min_dur_filter: Filter dataset if shorter than given min_dur_filter
        """
        super(BaseDataset, self).__init__()
        self._total_duration = 0.0
        # Load json datainfos, _spk_label for one-hot encoding
        self._dataset, self._spk_label = self._make_dataset_from_json(
            dataset_json, min_dur_filter=min_dur_filter)

        # Load noise set if add noise applied
        self._noise_dataset = []
        if noiseset_json is not None:
            self._make_noiseset_from_json(noiseset_json)

    @property
    def num_classes(self):
        # For sanity check of config, only for vpr task.
        return len(self._spk_label)

    def _make_noiseset_from_json(self, noise_json):
        # Make Noise datapool for add noise data_augmentation
        with open(noise_json, 'r') as json_f:
            for line in json_f:
                data_infos = json.loads(line)
                self._noise_dataset.append(data_infos["noise_filepath"])

    def _make_dataset_from_json(self, json_file, min_dur_filter):
        # Make Dataset list from JSON file, both VAD and VPR Task. dataset.json
        # should provide with 'spk_id' field if VPR task is specified.
        datamap = []
        spk_dic = {}
        with open(json_file, 'r') as json_f:
            for line in json_f:
                data_infos = json.loads(line)
                if data_infos["duration"] > min_dur_filter:
                    datamap.append(data_infos)
                    self._total_duration += data_infos["duration"]
                    if "spk_id" in data_infos:
                        # for VAD Task compliance.
                        if data_infos["spk_id"] not in spk_dic:
                            spk_dic[data_infos["spk_id"]] = len(spk_dic)
        return datamap, spk_dic

    def _read_label_from_hdf5(self, label_str):
        # This is used for VAD task by reading label from HDF5 file
        with h5py.File(label_str, 'r') as label_f:
            label = torch.as_tensor(label_f["label"])
        return label

    def __len__(self):
        """ Overwrite __len__"""
        return len(self._dataset)

    @property
    def total_duration(self):
        return self._total_duration

    @abc.abstractmethod
    def __getitem__(self, index):
        """ Implement for train, eval, test specifically """
        pass


class VprTrainDataset(BaseDataset):
    """ TrainDataset with Data Augmentation """

    def __init__(self, config) -> None:
        super(VprTrainDataset,
              self).__init__(config["train_data"],
                             noiseset_json=config["noise_data"])

        # Sanity check of config
        glog.check_eq(config["num_classes"], self.num_classes)
        glog.info("Training dataset: {}h with {} entries from {} spks.".format(
            self.total_duration / 3600, len(self), self.num_classes))

        self._chunk_size = config["chunk_size"]

        # Data Augmentation, on-the-fly fashion
        # TODO:? Spec_Aug necessary?
        self._add_noise_proportion = config["add_noise_proportion"]
        self._add_noise_config = config["add_noise_config"]
        self._add_noise = data_augmentation.add_noise

        if config["feat_type"] == "fbank":
            self._frontend = KaldiWaveFeature(**config["feat_config"])
        elif config["feat_type"] == "ecapa":
            self._frontend = EcapaFrontend()
        else:
            NotImplementedError

    def _spilt_chunk(self, feat: torch.Tensor):
        # randomly extract fixed size of chunk from original feat.
        if feat.shape[-1] < self._chunk_size:
            # If audio feats lengths less than chunksize, should repeat itself
            # to meet the chunk_size requirement
            feat = feat.repeat(
                torch.div(
                    self._chunk_size, feat.shape[1], rounding_mode="floor") + 1,
                1)
        offset = random.randint(0, feat.shape[0] - self._chunk_size)
        feat = feat[offset:offset + self._chunk_size, :]

        return feat

    def __getitem__(self, index):
        """ Return:
                {"feat": Tensor.float,
                 "spk_id": Tensor.long}
        """
        data = self._dataset[index]

        assert "audio_filepath" in data
        assert "spk_id" in data
        pcm, _ = torchaudio.load(data["audio_filepath"], normalize=True)

        # Data Augmentation
        # Use add noise proportion control the augmentation ratio of all dataset
        need_noisify_aug = random.uniform(0, 1) < self._add_noise_proportion
        if need_noisify_aug:
            noise_pcm, _ = torchaudio.load(random.choice(self._noise_dataset),
                                           normalize=True)
            pcm = self._add_noise(pcm, noise_pcm, **self._add_noise_config)

        feat = self._frontend(pcm)
        feat = self._spilt_chunk(feat)

        return {
            "feat": feat,
            "label": torch.tensor(self._spk_label[data["spk_id"]])
        }


class VprEvalDataset(BaseDataset):
    """ EavlDataset with Data Augmentation """

    def __init__(self, config) -> None:
        super(VprEvalDataset, self).__init__(config["eval_data"])

        # Sanity check of config
        glog.info(
            "Evaluation dataset: {}h with {} entries from {} spks.".format(
                self.total_duration / 3600, len(self), self.num_classes))

        self._chunk_size = config["chunk_size"]
        if config["feat_type"] == "fbank":
            self._frontend = KaldiWaveFeature(**config["feat_config"])
        elif config["feat_type"] == "ecapa":
            self._frontend = EcapaFrontend()
        else:
            NotImplementedError

    def _spilt_chunk(self, feat: torch.Tensor):
        # randomly extract fixed size of chunk from original feat.
        if feat.shape[-1] < self._chunk_size:
            # If audio feats lengths less than chunksize, should repeat itself
            # to meet the chunk_size requirement
            feat = feat.repeat(
                torch.div(
                    self._chunk_size, feat.shape[1], rounding_mode="floor") + 1,
                1)
        offset = random.randint(0, feat.shape[0] - self._chunk_size)
        feat = feat[offset:offset + self._chunk_size, :]

        return feat

    def __getitem__(self, index):
        """ Return:
                {"feat": Tensor.float,
                 "spk_id": Tensor.long}
        """
        data = self._dataset[index]

        assert "audio_filepath" in data
        assert "spk_id" in data
        pcm, _ = torchaudio.load(data["audio_filepath"], normalize=True)

        feat = self._frontend(pcm)
        feat = self._spilt_chunk(feat)

        return {
            "feat": feat,
            "label": torch.tensor(self._spk_label[data["spk_id"]])
        }


class VadTrainDataset(BaseDataset):
    """ TrainDataset for VAD task, note that JSON of file input is required. 
        NOTE: Set num workers of DataLoader to get considerable performance
    """

    def __init__(self, config) -> None:
        super(VadTrainDataset,
              self).__init__(config["train_data"],
                             noiseset_json=config["noise_data"],
                             min_dur_filter=config["min_dur_filter"])

        glog.info("Training dataset: {}h with {} entries.".format(
            self.total_duration / 3600, len(self)))

        self._chunk_size = config["chunk_size"]

        # Data Augmentation, on-the-fly fashion
        self._add_noise_proportion = config["add_noise_proportion"]
        self._add_noise_config = config["add_noise_config"]
        self._add_noise = data_augmentation.add_noise

        if config["feat_type"] == "fbank":
            self._frontend = KaldiWaveFeature(**config["feat_config"])
        elif config["feat_type"] == "ecapa":
            self._frontend = EcapaFrontend()
        else:
            NotImplementedError

    def _spilt_chunk(self, feat, label):
        # Split chunks from raw feats.
        offset = random.randint(0, label.shape[0] - self._chunk_size)
        feat = feat[offset:offset + self._chunk_size, :]
        label = label[offset:offset + self._chunk_size]
        return feat, label

    def __getitem__(self, index):
        """ Return:
                {"feat": Tensor.float,
                 "label": Tensor.long}
        """
        data = self._dataset[index]

        assert "audio_filepath" in data
        assert "label" in data
        pcm, _ = torchaudio.load(data["audio_filepath"], normalize=True)

        # Data Augmentation
        # Use add noise proportion control the augmentation ratio of all dataset
        need_noisify_aug = random.uniform(0, 1) < self._add_noise_proportion
        if need_noisify_aug:
            noise_pcm, _ = torchaudio.load(random.choice(self._noise_dataset),
                                           normalize=True)
            pcm = self._add_noise(pcm, noise_pcm, **self._add_noise_config)

        feat = self._frontend(pcm)
        label_info = data["label"]
        if label_info.startswith("NS"):
            # Indicating this is a total noise file, so label as zeros.
            label = torch.zeros(feat.shape[1]).long()
        else:
            label = self._read_label_from_hdf5(label_info)
            # feats: (T, D); label: (T)
            # NOTE: Assertion to check frame length match between feats and label.
            # Label of vad dataset is strictly follow the framing strategy of acoustic
            # feature extraction, that is 25ms frame_size, 10ms frame_shift and
            # drop_last without padding (like dataloader drop last). Thus, if any
            # problem encountered by following length check, please review your frontend
            # config and vad dataset generating pipeline codes sequentially.
            glog.check_eq(feat.shape[0], label.shape[0])

        feat, label = self._spilt_chunk(feat, label)

        return {"feat": feat, "label": label}


class VadEvalDataset(BaseDataset):
    """ EvalDataset for VAD task. """

    def __init__(self, config) -> None:
        super(VadEvalDataset,
              self).__init__(config["eval_data"],
                             min_dur_filter=config["min_dur_filter"])

        glog.info("Evaluation dataset: {}h with {} entries.".format(
            self.total_duration / 3600, len(self)))

        self._chunk_size = config["chunk_size"]

        if config["feat_type"] == "fbank":
            self._frontend = KaldiWaveFeature(**config["feat_config"])
        elif config["feat_type"] == "ecapa":
            self._frontend = EcapaFrontend()
        else:
            NotImplementedError

    def _spilt_chunk(self, feat, label):
        # NOTE: Evaluation with chunk is prefered for batching and has
        # no downgrade on metric acc compared with chunk-free.
        offset = random.randint(0, label.shape[0] - self._chunk_size)
        feat = feat[offset:offset + self._chunk_size, :]
        label = label[offset:offset + self._chunk_size]
        return feat, label

    def __getitem__(self, index):
        """ Return:
                {"feat": Tensor.float,
                 "label": Tensor.long}
        """
        data = self._dataset[index]

        assert "audio_filepath" in data
        assert "label" in data
        pcm, _ = torchaudio.load(data["audio_filepath"], normalize=True)

        feat = self._frontend(pcm)

        label_info = data["label"]
        if label_info.startswith("NS"):
            # Indicating this is a total noise file, so label as zeros.
            label = torch.zeros(feat.shape[1]).long()
        else:
            label = self._read_label_from_hdf5(label_info)
            # feats: (T, D); label: (T)
            glog.check_eq(feat.shape[0], label.shape[0])

        feat, label = self._spilt_chunk(feat, label)

        return {"feat": feat, "label": label}


class VadTestDataset(BaseDataset):
    """ Test Dataset for vad task inference. TorchScript frontend will be
        utilized. Chunkization will be exclude and batch_size shall be 1.
    """

    def __init__(self, dataset_json, frontend) -> None:
        # Testset should not filter any of the data.
        super(VadTestDataset, self).__init__(dataset_json=dataset_json)
        glog.info("Test dataset: {}h with {} entries.".format(
            self.total_duration / 3600, len(self)))

        # Load Torchscript frontend to init feature extraction session
        self._frontend_sess = torch.jit.load(frontend)

    def __getitem__(self, index):
        """ Return:
                {"feat": Tensor.float,
                 "label": Tensor.long}
        """
        data = self._dataset[index]

        assert "audio_filepath" in data
        assert "label" in data
        pcm, _ = torchaudio.load(data["audio_filepath"], normalize=True)

        feat = self._frontend_sess(pcm)

        label_info = data["label"]
        if label_info.startswith("NS"):
            # Indicating this is a total noise file, so label as zeros.
            label = torch.zeros(feat.shape[1]).long()
        else:
            label = self._read_label_from_hdf5(label_info)
            # feats: (T, D); label: (T)
            glog.check_eq(feat.shape[0], label.shape[0])

        # Chunk split exluded.
        return {"feat": feat, "label": label}


def collate_fn(raw_batch):
    """ Batching right before output, implement for train, eval """
    batch_map = {"feat": [], "label": []}

    for data_slice in raw_batch:
        # Reorganize batch data as Map
        glog.check("feat" in data_slice.keys())
        glog.check("label" in data_slice.keys())

        batch_map["feat"].append(data_slice["feat"])
        batch_map["label"].append(data_slice["label"])
    # stack as a batch
    batch_map["feat"] = torch.stack(batch_map["feat"], dim=0)
    batch_map["label"] = torch.stack(batch_map["label"], dim=0).long()

    return batch_map
