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


class VprTask(pl.LightningModule):
  """ Build VPR task from yaml config """

  def __init__(self, config) -> None:
    """ Initalization of Task """

    super(VprTask, self).__init__()
    # Split configs of yaml
    self._dataset_config = config["dataset"]
    self._embedding_model_config = config["embedding_model"]
    self._loss_config = config["loss"]
    self._metric_config = config["metric"]
    self._optim_config = config["optim_setup"]

    # Build all components with given yaml config
    self._frontend = self._get_frontend(copy.deepcopy(config["dataset"]))
    self._embedding_model = EmbeddingModel(self._embedding_model_config)
    self._loss = Loss(self._loss_config)
    self._metric = Metric(config=MetricConfig(**self._metric_config["config"]))

  def _get_frontend(self, config):
    # Get Frontend from config to export frontend compute graph
    # Set dither as 0.0 when output frontend
    if config["feat_type"] == "fbank":
      config["feat_config"]["dither"] = 0.0
      return KaldiWaveFeature(**config["feat_config"])
    elif config["feat_type"] == "ecapa":
      return EcapaFrontend()
    else:
      raise ValueError("feat type not supported.")

  def forward(self):
    """ Forward of Model for inference, for model export """
    pass

  def train_dataloader(self):
    """ Config dataloader of training step """

    dataset = VprTrainDataset(self._dataset_config)
    dataloader = DataLoader(dataset=dataset,
                            shuffle=True,
                            collate_fn=collate_fn,
                            batch_size=self._dataset_config["batch_size"],
                            num_workers=self._dataset_config["num_workers"])
    return dataloader

  def val_dataloader(self):
    """ Config dataloader of evalutaion step """

    dataset = VprEvalDataset(self._dataset_config)
    dataloader = DataLoader(dataset=dataset,
                            collate_fn=collate_fn,
                            batch_size=self._dataset_config["batch_size"],
                            num_workers=self._dataset_config["num_workers"])
    return dataloader

  def training_step(self, batch, batch_idx):
    """ DataAgumentation would be compute every training step begins,
            so, off-to-go batch input would be
            {"feat": Tensor Float,
             "label": Tensor Long,}
        """

    embeddings = self._embedding_model(batch["feat"])

    # Organize batch as Loss API
    loss_input_batch = {
        "embeddings": embeddings,
        "labels": batch["label"],
    }
    loss = self._loss(loss_input_batch)
    if batch_idx % 100 == 0:
      glog.info(
          "Train (Epoch: {} / Local_steps: {} / Global_steps: {}) loss: {}".
          format(self.current_epoch, batch_idx, self.global_step, loss))

    self.log("train_loss", loss)
    return loss

  def validation_step(self, batch, batch_idx):
    """ DataAgumentation would be compute every training step begins, 
            data augmentation should be excluded during eval.
        """

    embeddings = self._embedding_model(batch["feat"])

    # Organize batch as Loss API
    loss_input_batch = {
        "embeddings": embeddings,
        "labels": batch["label"],
    }
    loss = self._loss(loss_input_batch)
    predictions = self._loss.predict(loss_input_batch)

    metrics = self._metric(preds=predictions, labels=batch["label"])
    if batch_idx % 100 == 0:
      glog.info(
          "Eval (Epoch: {} / Local_steps: {} / Global_steps: {}) loss: {} Metrics: {}"
          .format(self.current_epoch, batch_idx, self.global_step, loss,
                  metrics))

    self.log_dict({"val_loss": loss, **metrics}, sync_dist=True)

  def configure_optimizers(self):
    """ Optimizer configuration """
    Optimizer, LR_Scheduler = OptimSetup(self._optim_config)
    optimizer = Optimizer(self.parameters(),
                          **self._optim_config["optimizer"]["config"])
    lr_scheduler = LR_Scheduler(optimizer=optimizer,
                                **self._optim_config["lr_scheduler"]["config"])
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            **self._optim_config["lr_scheduler"]["step_config"]
        }
    }
