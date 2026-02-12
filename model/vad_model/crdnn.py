# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.12
""" Crdnn vad model impl, streaming ensured. """

import abc
import dataclasses
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List, Dict


@dataclasses.dataclass
class CnnBlockConfig:
  """ Config interface of CnnBlock, model should be built from this only """
  num_layers: int = 3
  conv_type: str = "conv1d"
  input_dim: int = 64
  in_channels_config: Tuple = (64, 128, 128)
  out_channels_config: Tuple = (128, 128, 64)
  kernel_configs: Tuple = (3, 5, 3)  # If conv2d, set as Tuple[(T, F)]
  stride_configs: Tuple = None  # If conv2d, set as Tuple[int] working on Freq where Time is always 1
  dilation_config: Tuple = (2, 1, 2)  # If conv2d, should set as Tuple[(T, F)]


class CnnBlockBase(nn.Module):
  """ NOTE: Base model of Cnn block, Conv1d and Conv2d type shall be inherited from 
        it. Basically this desgin is for torchscript export which should avoid data type 
        inconsistency between Conv1d and Conv2d. Cache strategy of inference should be 
        imp; ensuring streaming inference of model.
    """

  def __init__(self, config: CnnBlockConfig) -> None:
    super(CnnBlockBase, self).__init__()

    assert len(config.kernel_configs) == len(
        config.dilation_config) == config.num_layers
    assert len(config.in_channels_config) == len(
        config.out_channels_config) == config.num_layers

    self._num_layers = config.num_layers
    self._input_dim = config.input_dim
    self._in_channels_config = config.in_channels_config
    self._out_channels_config = config.out_channels_config
    self._kernel_configs = config.kernel_configs
    self._stride_configs = config.stride_configs
    self._dilation_config = config.dilation_config

    # Data normalization using BatchNorm1d
    self._normalization = nn.BatchNorm1d(num_features=self._input_dim)

    # Build cnn block by stacking cnn_layer
    self._cnn_layers = torch.nn.ModuleList([
        nn.Sequential(*self._make_cnn_layers(layer_id))
        for layer_id in range(self._num_layers)
    ])

  @abc.abstractmethod
  def _make_cnn_layers(self, layer_id):
    # Build Cnn layers, Conv1d with stride=1, padding="valid" and
    # Conv2d with stride=(1, stride), padding="valid" are mandatory
    # for causal forward and streaming mode inference impl.
    ...

  @property
  @abc.abstractmethod
  def cache_size(self):
    # Cache size will be applied in each layer within cnn_block. This size will be
    # left padding size with 0 of for each cnn layer during training, whereas
    # indicating buffer size of stored history infos during inference.
    ...

  @property
  @abc.abstractmethod
  def output_dim(self):
    # Output dim, specific for sequential Dnn block init.
    ...

  @abc.abstractmethod
  def _left_padding(self, feats: torch.Tensor, padding_size: int):
    # If Conv1d, Feats: (B, D, T) -> (B, D, T + padding_size)， padding left with 0.
    # If Conv2d, Feats: (B, C, T, D) -> (B, C, T + padding_size, D), padding left with 0.
    ...

  @abc.abstractmethod
  def forward(self, x: torch.Tensor):
    # Training graph
    ...

  @torch.jit.export
  @abc.abstractmethod
  def initialize_cnn_cache(self):
    # Initialize cache when inference start.
    ...

  @torch.jit.export
  @torch.inference_mode(mode=True)
  @abc.abstractmethod
  def inference(self, x: torch.Tensor, cache: List[torch.Tensor]):
    # Inference graph, Batch Size shall be 1.
    # Cache strategy applied in inference, supporting both streaming and
    # non-streaming inference. The initial cache setup please refer to
    # initialize_cnn_cache.
    ...


class Conv1dCnnBlock(CnnBlockBase):
  """ Conv1d Cnn block. Streaming inference ensured """

  def __init__(self, config: CnnBlockConfig) -> None:
    assert config.conv_type == "conv1d"
    assert config.in_channels_config[
        0] == config.input_dim, "If conv1d is set, in channels of 1st layer should be consist with input dim"
    super(Conv1dCnnBlock, self).__init__(config=config)

  def _make_cnn_layers(self, layer_id) -> List[nn.Module]:
    # Build Conv1d layers, stride won't work here, strides will strictly be 1.
    return [
        nn.Conv1d(in_channels=self._in_channels_config[layer_id],
                  out_channels=self._out_channels_config[layer_id],
                  kernel_size=self._kernel_configs[layer_id],
                  stride=1,
                  dilation=self._dilation_config[layer_id],
                  padding=0),
        nn.BatchNorm1d(num_features=self._out_channels_config[layer_id]),
        nn.LeakyReLU()
    ]

  @property
  def cache_size(self) -> List[int]:
    # Cache size of Conv1d Cnn block
    return [(self._kernel_configs[i] - 1) * self._dilation_config[i]
            for i in range(self._num_layers)]

  @property
  def output_dim(self) -> int:
    # Out_dim of Conv1d Cnn block
    return self._out_channels_config[-1]

  def _left_padding(self, feats: torch.Tensor,
                    padding_size: int) -> torch.Tensor:
    # Conv1d left padding
    # Feats: (B, D, T) -> (B, D, T + padding_size), padding left with 0
    batch_size = feats.shape[0]
    feats_dim = feats.shape[1]
    left_padding = torch.zeros(batch_size, feats_dim,
                               padding_size).to(feats.device)
    feats = torch.concat([left_padding, feats], dim=-1)
    return feats

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Conv1d training graph
    x = self._normalization(x.transpose(1, 2)).transpose(1, 2)
    x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T) for Conv1d

    for layer_id, cnn_layer in enumerate(self._cnn_layers):
      x = self._left_padding(x, padding_size=self.cache_size[layer_id])
      x = cnn_layer(x)

    x = x.transpose(1, 2)  # (B, D, T) -> (B, T, D)
    return x

  @torch.jit.export
  def initialize_cnn_cache(self) -> List[torch.Tensor]:
    # Cnn cache of Conv1d initialization
    cache = []
    for cache_size in zip(self._in_channels_config, self.cache_size):
      cache.append(torch.zeros(1, *cache_size))  # (1, D, cache_size)
    return cache

  @torch.jit.export
  @torch.inference_mode(mode=True)
  def inference(
      self, x: torch.Tensor,
      cache: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """ "feat": Tensor.Float (1, T, D),
            "cache": List[layer_0 cache(Tensor.Float), layer_1 cache(Tensor.Float), ...],
        """
    # Streaming inference of Conv1d
    next_cache = []
    x = self._normalization(x.transpose(1, 2)).transpose(1, 2)
    x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T) for Conv1d

    for layer_id, cnn_layer in enumerate(self._cnn_layers):
      # if Conv1d, Feats: (B, D, T) -> (B, D, T + padding_size)
      x = torch.concat([cache[layer_id].to(x.device), x], dim=2)
      cache_pos = x.shape[2] - self.cache_size[layer_id]
      next_cache.append(x[:, :, cache_pos:])
      x = cnn_layer(x)

    x = x.transpose(1, 2)  # (B, D, T) -> (B, T, D)

    return x, next_cache


class Conv2dCnnBlock(CnnBlockBase):
  """ Conv2d Cnn block. Streaming inference ensured """

  def __init__(self, config: CnnBlockConfig) -> None:
    assert config.conv_type == "conv2d"
    assert config.in_channels_config[
        0] == 1, "If conv2d is set, in_channels of 1st layer should 1."

    super(Conv2dCnnBlock, self).__init__(config=config)

  def _make_cnn_layers(self, layer_id) -> List[nn.Module]:
    # Build Conv2d layers.
    # stride should be (1, stride) with input shape (B, C, T, F)
    return [
        nn.Conv2d(in_channels=self._in_channels_config[layer_id],
                  out_channels=self._out_channels_config[layer_id],
                  kernel_size=self._kernel_configs[layer_id],
                  stride=(1, self._stride_configs[layer_id]),
                  dilation=self._dilation_config[layer_id],
                  padding=0),
        nn.BatchNorm2d(num_features=self._out_channels_config[layer_id]),
        nn.LeakyReLU()
    ]

  @property
  def cache_size(self) -> List[int]:
    # Given if Conv2d applied, kernel_size and dilation will be configed as (T, F)
    return [(self._kernel_configs[i][0] - 1) * self._dilation_config[i][0]
            for i in range(self._num_layers)]

  @property
  def output_dim(self) -> int:
    # Out dim of Conv2d Cnn block
    length = self._input_dim  # Init with input_dim
    for layer_id in range(self._num_layers):
      # Only pick Freq domain of (T, F) for out_dim compute
      dilation = self._dilation_config[layer_id][1]
      k_size = self._kernel_configs[layer_id][1]
      stride = self._stride_configs[layer_id]
      length = math.floor((length - dilation * (k_size - 1) - 1) / stride + 1)
    return length * self._out_channels_config[-1]

  def _left_padding(self, feats: torch.Tensor,
                    padding_size: int) -> torch.Tensor:
    # Conv2d left padding
    # Feats: (B, C, T, D) -> (B, C, T + padding_size, D), padding left with 0
    batch_size = feats.shape[0]
    channel_dim = feats.shape[1]
    feats_dim = feats.shape[-1]
    left_padding = torch.zeros(batch_size, channel_dim, padding_size,
                               feats_dim).to(feats.device)
    feats = torch.concat([left_padding, feats], dim=2)
    return feats

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Conv2d training graph
    x = self._normalization(x.transpose(1, 2)).transpose(1, 2)
    x = x.unsqueeze(1)  # (B, T, D) -> (B, C, T, D) for Conv2d

    for layer_id, cnn_layer in enumerate(self._cnn_layers):
      x = self._left_padding(x, padding_size=self.cache_size[layer_id])
      x = cnn_layer(x)

    # NOTE: Original design of reshape without transpose(C, T) will
    # lead to mismatch of computation between training and streaming
    # inference due to the permutaion of reshape disarange time_dim's
    # consistency. So, there you go. However this will sweep away the
    # rewards of 0.5% of eval acc from the extensive infos benefit from
    # this disarange. Worthy sacrifice for streaming anyway.
    x = x.transpose(1, 2).contiguous().reshape(
        x.shape[0], x.shape[2],
        -1)  # (B, C, T, D）-> (B, T, C, D) -> (B, T, C * D) - for Conv2d

    return x

  @torch.jit.export
  def initialize_cnn_cache(self) -> List[torch.Tensor]:
    # Cnn cache of Conv2d initialization
    cache = []
    length = self._input_dim  # Init with input_dim
    for layer_id in range(self._num_layers):
      # The first cache is (1, 1, T_cache_size, input_dim) where Batchsize = channels = 1
      cache.append(
          torch.zeros(1, self._in_channels_config[layer_id],
                      self.cache_size[layer_id], length))
      # Only pick Freq domain of (T, F) for out_dim compute
      dilation = self._dilation_config[layer_id][1]
      k_size = self._kernel_configs[layer_id][1]
      stride = self._stride_configs[layer_id]
      length = math.floor((length - dilation * (k_size - 1) - 1) / stride + 1)
    return cache

  @torch.jit.export
  @torch.inference_mode(mode=True)
  def inference(
      self, x: torch.Tensor,
      cache: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """ "feat": Tensor.Float (1, T, D),
            "cache": List[layer_0 cache(Tensor.Float), layer_1 cache(Tensor.Float), ...],
        """
    next_cache = []
    x = self._normalization(x.transpose(1, 2)).transpose(1, 2)
    x = x.unsqueeze(1)  # (B, T, D) -> (B, C, T, D). for Conv2d

    for layer_id, cnn_layer in enumerate(self._cnn_layers):
      # if Conv2d, Feats: (B, C, T, D) -> (B, C, T + padding size, D)
      x = torch.concat([cache[layer_id].to(x.device), x], dim=2)
      cache_pos = x.shape[2] - self.cache_size[layer_id]
      next_cache.append(x[:, :, cache_pos:, :])
      x = cnn_layer(x)

    x = x.transpose(1, 2).reshape(
        x.shape[0], x.shape[2],
        -1)  # (B, C, T, D) -> (B, T, C, D)->(B, T, C * D) for Conv2d

    return x, next_cache


@dataclasses.dataclass
class DnnBlockConfig:
  """ Dnn Block config of Crdnn """
  num_layers: int = 2
  hidden_dim: int = 64
  dropout_p: float = 0.15


class DnnBlock(nn.Module):
  """ Dnn Block Impl, naturally support streaming mode """

  def __init__(self, _input_dim: int, config: DnnBlockConfig):
    super(DnnBlock, self).__init__()
    # Initialization
    self._input_dim = _input_dim
    self._num_layers = config.num_layers
    self._hidden_dim = config.hidden_dim
    self._dropout_p = config.dropout_p

    self._dnn_layers = nn.Sequential(*self._make_dnn_layers())

  def _make_dnn_layers(self) -> List[nn.Module]:
    # Make dnn layer from config
    dnn_layers = []
    for layer_id in range(self._num_layers):
      if layer_id == 0:
        dnn_layers.append(
            nn.Sequential(nn.Linear(self._input_dim, self._hidden_dim),
                          nn.LeakyReLU(), nn.Dropout(p=self._dropout_p)))
      else:
        dnn_layers.append(
            nn.Sequential(nn.Linear(self._hidden_dim, self._hidden_dim),
                          nn.LeakyReLU(), nn.Dropout(p=self._dropout_p)))
    return dnn_layers

  @property
  def output_dim(self) -> int:
    return self._hidden_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Training graph
    return self._dnn_layers(x)

  @torch.jit.export
  @torch.inference_mode(mode=True)
  def inference(self, x: torch.Tensor) -> torch.Tensor:
    # Inference graph, cache-free
    return self._dnn_layers(x)


@dataclasses.dataclass
class RnnBlockConfig:
  """ Rnn Block config of Crdnn """
  rnn_type: str = "lstm"
  hidden_size: int = 128
  num_layers: int = 2
  batch_first: bool = True
  dropout: float = 0.0
  bidirectional: bool = False


class RnnBlockBase(nn.Module):
  """ NOTE: Base model of Rnn block, LSTM and GRU type shall inherit from 
        it. Basically this desgin is for torchscript export which should avoid 
        inconsistency of cache between LSTM and GRU, Tuple[h_0, c_0] and h_0
        respectivly. Rnn naturally support streaming mode with cache. 
    """

  def __init__(self, _input_dim: int, config: RnnBlockConfig):
    super(RnnBlockBase, self).__init__()
    # Initialization
    self._input_size = _input_dim
    self._hidden_size = config.hidden_size
    self._num_layers = config.num_layers
    self._batch_first = config.batch_first
    self._dropout = config.dropout
    self._bidirectional = config.bidirectional

    self._layernorm = nn.LayerNorm(self._input_size)

  def _make_rnn_layers(self, layer: nn.Module) -> nn.Module:
    # Build Rnn layers with specific type rnn
    rnn_layers = layer(input_size=self._input_size,
                       hidden_size=self._hidden_size,
                       num_layers=self._num_layers,
                       batch_first=self._batch_first,
                       dropout=self._dropout,
                       bidirectional=self._bidirectional)
    return rnn_layers

  @property
  def output_dim(self):
    return self._hidden_size

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Training graph
    x = self._layernorm(x)
    x, _ = self._rnn_layers(x)
    return x

  @torch.jit.export
  @abc.abstractmethod
  def initialize_rnn_cache(self):
    # Initialize rnn cache when streaming inference start. Batch size = 1 is mandatory.
    # GRU cache: Tuple[torch.Tensor, dummy_tensor]; LSTM cache: Tuple[torch.Tensor, torch.Tensor]
    ...

  @torch.jit.export
  @torch.inference_mode(mode=True)
  @abc.abstractmethod
  def inference(self, x: torch.Tensor, cache):
    """ "feat": (B, T, D)
            "cache": (h_O, c_0) for LSTM, (h_O, dummy_cache) for GRU
        """
    ...


class GruRnnBlock(RnnBlockBase):
  """ GRU Rnn block impl """

  def __init__(self, _input_dim: int, config: RnnBlockConfig):
    super(GruRnnBlock, self).__init__(_input_dim, config)
    assert config.rnn_type == "gru"
    self._rnn_type = config.rnn_type
    self._rnn_layers = self._make_rnn_layers(nn.GRU)

  @torch.jit.export
  def initialize_rnn_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
    # NOTE: Originally GRU cache is solely h_0. However considering of torchscript export,
    # which has strict data type indication of funcs, cache init shall be consist with LSTM.
    cache = (torch.zeros(self._num_layers, 1,
                         self._hidden_size), torch.zeros(0))  # h_0, dummy_cache
    return cache

  @torch.inference_mode(mode=True)
  def inference(self, x: torch.Tensor, cache: Tuple[torch.Tensor,
                                                    torch.Tensor]):
    """ "feat": (B, T, D)
            "cache": (h_0, dummy_cache) for GRU
        """
    # Streaming inference graph
    x = self._layernorm(x)
    output, next_cache = self._rnn_layers(x, cache[0].to(x.device))

    return output, (next_cache, cache[1])  # maintain the dummy cache


class LstmRnnBlock(RnnBlockBase):
  """ LSTM Rnn block impl """

  def __init__(self, _input_dim: int, config: RnnBlockConfig):
    super(LstmRnnBlock, self).__init__(_input_dim, config)
    assert config.rnn_type == "lstm"
    self._rnn_type = config.rnn_type
    self._rnn_layers = self._make_rnn_layers(nn.LSTM)

  @torch.jit.export
  def initialize_rnn_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
    # Initialize rnn cache when streaming inference start. Batch size = 1 is mandatory.
    # LSTM cache: Tuple[torch.Tensor, torch.Tensor]
    cache = (torch.zeros(self._num_layers, 1, self._hidden_size),
             torch.zeros(self._num_layers, 1, self._hidden_size))  # h_0, c_0
    return cache

  @torch.jit.export
  @torch.inference_mode(mode=True)
  def inference(self, x: torch.Tensor, cache: Tuple[torch.Tensor,
                                                    torch.Tensor]):
    """ "feat": (B, T, D)
            "cache": (h_0, c_0) for LSTM
        """
    # Streaming inference graph
    x = self._layernorm(x)
    output, next_cache = self._rnn_layers(
        x, (cache[0].to(x.device), cache[1].to(x.device)))

    return output, next_cache


@dataclasses.dataclass
class CrdnnConfig:
  """ Crdnn Model config interface """
  cnn_block_config: CnnBlockConfig = CnnBlockConfig()
  dnn_block_config: DnnBlockConfig = DnnBlockConfig()
  rnn_block_config: RnnBlockConfig = RnnBlockConfig()


class Crdnn(nn.Module):
  """ Vad model Crdnn impl, build with Cnn, Dnn, Rnn blocks """

  def __init__(self, config: CrdnnConfig):
    super(Crdnn, self).__init__()
    # Initialization
    # CnnBlock initialize
    if config.cnn_block_config["conv_type"] == "conv1d":
      self._cnn_blocks = Conv1dCnnBlock(config=CnnBlockConfig(
          **config.cnn_block_config))
    elif config.cnn_block_config["conv_type"] == "conv2d":
      self._cnn_blocks = Conv2dCnnBlock(config=CnnBlockConfig(
          **config.cnn_block_config))

    # DnnBlock initialize
    self._dnn_blocks = DnnBlock(
        _input_dim=self._cnn_blocks.output_dim,
        config=DnnBlockConfig(**config.dnn_block_config))

    # RnnBlock initialize
    if config.rnn_block_config["rnn_type"] == "gru":
      self._rnn_blocks = GruRnnBlock(
          _input_dim=self._dnn_blocks.output_dim,
          config=RnnBlockConfig(**config.rnn_block_config))
    elif config.rnn_block_config["rnn_type"] == "lstm":
      self._rnn_blocks = LstmRnnBlock(
          _input_dim=self._dnn_blocks.output_dim,
          config=RnnBlockConfig(**config.rnn_block_config))

    # Output layer initialize, linearly transform into dim = 2
    self._output_layer = nn.Linear(in_features=self._rnn_blocks.output_dim,
                                   out_features=2)
    self._softmax = nn.Softmax(dim=-1)

  @torch.jit.export
  def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
    # Compute softmax logits.
    return self._softmax(x)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Training Graph
    x = self._cnn_blocks(x)
    x = self._dnn_blocks(x)
    x = self._rnn_blocks(x)
    x = self._output_layer(x)  # output: (B, T, 2)
    return x

  @torch.jit.export
  def initialize_cache(self):
    # Initialize cache when inference start, Batch size = 1
    # DNN does not need cache.
    cnn_cache = self._cnn_blocks.initialize_cnn_cache()
    rnn_cache = self._rnn_blocks.initialize_rnn_cache()

    return (cnn_cache, rnn_cache)

  @torch.jit.export
  @torch.inference_mode(mode=True)
  def inference(self, x: torch.Tensor, cache: Tuple[List[torch.Tensor],
                                                    Tuple[torch.Tensor,
                                                          torch.Tensor]]):
    # Streaming inference impl by using inference interface of all blocks.
    cnn_cache = cache[0]  # cnn_cache
    rnn_cache = cache[1]  # rnn_cache

    x, cnn_cache = self._cnn_blocks.inference(x, cnn_cache)
    x = self._dnn_blocks.inference(x)
    x, rnn_cache = self._rnn_blocks.inference(x, rnn_cache)
    x = self._output_layer(x)
    logits = self._softmax(x)

    return logits, (cnn_cache, rnn_cache)  # Cache for next chunk or frame input
