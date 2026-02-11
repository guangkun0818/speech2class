# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.13
""" Implement of ResNet embedding model
    https://arxiv.org/pdf/1512.03385.pdf
"""

import dataclasses
import glog
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
  """ Basic Block used for ResNet18 nad ResNet34 """

  EXPANSION = 1

  def __init__(self, in_channels, out_channels, stride=1):
    super(BasicBlock, self).__init__()

    # Residual function of basic block
    # Block infos refered from paper:
    # (Conv2d(3 * 3, 64),
    #  Conv2d(3 * 3, 64))
    self._residual_function = nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=3,
                  stride=stride,
                  padding=1,
                  bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels,
                  out_channels=out_channels * self.EXPANSION,
                  kernel_size=3,
                  padding=1,
                  bias=False), nn.BatchNorm2d(out_channels * self.EXPANSION))

    # Shortcut
    self._shortcut = nn.Sequential()

    # The shortcut output dimension is not the same with residual
    # function, using 1 * 1 conv to fix the mismatch
    if stride != 1 or in_channels != self.EXPANSION * out_channels:
      self._shortcut = nn.Sequential(
          nn.Conv2d(in_channels=in_channels,
                    out_channels=self.EXPANSION * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(out_channels * self.EXPANSION))

  def forward(self, x):
    # Training graph
    # Input shape: (Batch, channel, height, width)
    return nn.ReLU(inplace=True)(self._residual_function(x) + self._shortcut(x))


class BottleNeckBlock(nn.Module):
  """ Bottleneck block used for ResNet50 and more """

  EXPANSION = 4

  def __init__(self, in_channels, out_channels, stride=1) -> None:
    super(BottleNeckBlock, self).__init__()

    # Residual function of bottleneck block
    # Block infos refered from paper:
    # (Conv2d(1 * 1, 64),
    #  Conv2d(3 * 3, 64)
    #  Conv2d(1 * 1, 256))
    self._residual_function = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels,
                  out_channels=out_channels,
                  kernel_size=3,
                  stride=stride,
                  padding=1,
                  bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels,
                  out_channels=out_channels * self.EXPANSION,
                  kernel_size=1,
                  bias=False), nn.BatchNorm2d(out_channels * self.EXPANSION))

    # Shortcut
    self._shortcut = nn.Sequential()

    # The shortcut output dimension is not the same with residual
    # function, using 1 * 1 conv to fix the mismatch
    if stride != 1 or in_channels != self.EXPANSION * out_channels:
      self._shortcut = nn.Sequential(
          nn.Conv2d(in_channels=in_channels,
                    out_channels=self.EXPANSION * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(out_channels * self.EXPANSION))

  def forward(self, x):
    # Training graph
    # Input shape: (Batch, channel, height, width)
    return nn.ReLU(inplace=True)(self._residual_function(x) + self._shortcut(x))


class StatisticPooling(nn.Module):
  """ Statistic pooling of X-Vector """

  def __init__(self, input_dim):
    super(StatisticPooling, self).__init__()

    self._input_dim = input_dim
    self._output_dim = self._input_dim * 2

  def _statistic_std(self, x: torch.Tensor, mean_x: torch.Tensor, dim=-1, unbiased=False, eps=1e-8):
    # Add epsilon in sqrt function to gain more numerically stable
    var = torch.sum((x - mean_x.unsqueeze(-1))**2, dim=dim)
    if unbiased:
      length = x.shape[dim] - 1
    else:
      length = x.shape[dim]
    return torch.sqrt(var / length + eps)

  @property
  def output_dim(self):
    # Public property for embedding layer config
    return self._output_dim

  def forward(self, x: torch.Tensor):
    # Training Graph
    # Compress the timestep dimension
    mean_x = torch.mean(x, -1, False)
    # This might cause NaN `std_x = torch.std(x, -1, False)`
    std_x = self._statistic_std(x, mean_x, dim=-1, unbiased=False)
    output = torch.cat([mean_x, std_x], dim=-1)
    output = output.view(-1, self.output_dim)
    return output


class _Fc_layer(nn.Module):
  """ Fully Connect layer """

  def __init__(self, input_dim, output_dim):
    super(_Fc_layer, self).__init__()
    # TODO: add LeakyreLU to get more nonlinearity?
    self._fc = nn.Linear(in_features=input_dim, out_features=output_dim)
    self._leaky_relu = nn.LeakyReLU(0.1, inplace=True)
    self._batchnorm1d = nn.BatchNorm1d(output_dim, momentum=0.5)

  def forward(self, x: torch.Tensor):
    # Training Graph
    output = self._fc(x)
    output = self._leaky_relu(output)
    output = output.unsqueeze(2)
    output = self._batchnorm1d(output)
    output = output.squeeze(2)
    return output


class EmbeddingLayer(nn.Module):
  """ Process embedding from pooling layer output """

  def __init__(self, input_dim, embedding_dims=(256,)):
    super(EmbeddingLayer, self).__init__()

    # Embedding layer might have more than 1 layer, but
    # only the last layer output is speaker embedding
    self._embedding_dims = embedding_dims
    self._input_dims = [input_dim] + self._embedding_dims[:-1]

    # Build FC layers with config of embedding_dims
    self._fc_layers = []
    for i in range(len(self._embedding_dims)):
      self._fc_layers.append(
          nn.Sequential(_Fc_layer(input_dim=self._input_dims[i],
                                  output_dim=self._embedding_dims[i])))
    self._fc_layers = nn.Sequential(*self._fc_layers)

  def forward(self, x):
    # Training Graph
    # Output is final speaker embedding
    output = self._fc_layers(x)
    return output


@dataclasses.dataclass
class ResNetConfig:
  """ Config of ResNet, future implemented embedding_model should config itself 
        in this fashion.

        The default config is ResNet18
        ResNet34:
            block: nn.Module = BasicBlock
            num_block: tuple = (3, 4, 6, 3)
        ResNet50:
            block: nn.Module = BottleNeckBlock
            num_block: tuple = (3, 4, 6, 3)
        ResNet101:
            block: nn.Module = BottleNeckBlock
            num_block: tuple = (3, 4, 23, 3)
        ResNet152:
            block: nn.Module = BottleNeckBlock
            num_block: tuple = (3, 4, 36, 3)
    """
  feats_dim: int = 80
  block_type: str = "BasicBlock"
  num_blocks: tuple = (2, 2, 2, 2)
  in_channels: int = 32
  embedding_dims: tuple = (256)


class ResNet(nn.Module):
  """ Interface compatible of all type ResNet models """

  def __init__(self, config: ResNetConfig):
    super(ResNet, self).__init__()

    self._blocks_pool = {"BasicBlock": BasicBlock, "BottleNeckBlock": BottleNeckBlock}

    self._feats_dim = config.feats_dim
    self._block_type = self._blocks_pool[config.block_type]
    self._num_blocks = config.num_blocks
    self._in_channels = config.in_channels
    self._embedding_dims = config.embedding_dims

    # Subsampling layer before ResNet blocks
    self._conv_sub_sampling = nn.Sequential(
        nn.Conv2d(1, self._in_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(self._in_channels), nn.ReLU(inplace=True))

    # Different with original paper, stride of first ResNet block is 1
    self._resnet_block_1 = self._make_layer(self._block_type, config.in_channels,
                                            self._num_blocks[0], 1)
    self._resnet_block_2 = self._make_layer(self._block_type, config.in_channels * 2,
                                            self._num_blocks[1], 2)
    self._resnet_block_3 = self._make_layer(self._block_type, config.in_channels * 4,
                                            self._num_blocks[2], 2)
    self._resnet_block_4 = self._make_layer(self._block_type, config.in_channels * 8,
                                            self._num_blocks[3], 2)

    # Compute output feats dim for pooling layer config
    self._output_dim = config.in_channels * 8 * (
        (config.feats_dim - 1) // 8 + 1) * self._block_type.EXPANSION

    # Compress timestep dimension infomation as embedding
    self._pooling_layer = StatisticPooling(input_dim=self._output_dim)
    self._embedding_layer = EmbeddingLayer(input_dim=self._pooling_layer.output_dim,
                                           embedding_dims=self._embedding_dims)

  @property
  def output_dim(self):
    # Public property for pooling layer config
    return self._output_dim

  def _make_layer(self, block_type, out_channels, num_blocks, stride):
    """ Make ResNet layer (which is not "layer' as neuron network layer, e.g. Conv2d).
            One layer may contain more than one residual block.
            Args:
                block: Block type, BasicBlock or BottleNeckBlock 
                out_channels: Output depth channel number of this layer 
                num blocks: The number of blocks per layer 
                stride: The stride of the first block of this layer
            Return:
                Return a ResNet layer
        """
    # We have num block blocks per layer, the stride of first
    # block could be 1 or 2, other blocks would always be 1
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []

    for stride in strides:
      layers.append(block_type(self._in_channels, out_channels, stride))
      self._in_channels = out_channels * block_type.EXPANSION

    return nn.Sequential(*layers)

  def forward(self, x: torch.Tensor):
    # Training graph
    # Input shape: (Batch size, Seq_len, feats_dim)

    output = x.unsqueeze(1)
    output = output.transpose(2, 3)
    output = self._conv_sub_sampling(output)
    output = self._resnet_block_1(output)
    output = self._resnet_block_2(output)
    output = self._resnet_block_3(output)
    output = self._resnet_block_4(output)
    # Output shape: (Batch_size, out_channels, feats_dim, seq_len),
    # where seq_len represent the orginal seq_len position

    output = self._pooling_layer(output)
    output = self._embedding_layer(output)
    # Output shape: (Batch_size, embedding_ dim)
    # Final; output of speaker embedding

    return output

  @torch.inference_mode(mode=True)
  def inference(self, x: torch.Tensor):
    # Training graph
    # Input shape: (Batch size, Seq_len, feats_dim)

    output = x.unsqueeze(1)
    output = output.transpose(2, 3)
    output = self._conv_sub_sampling(output)
    output = self._resnet_block_1(output)
    output = self._resnet_block_2(output)
    output = self._resnet_block_3(output)
    output = self._resnet_block_4(output)
    # Output shape: (Batch_size, out_channels, feats_dim, seq_len),
    # where seq_len represent the orginal seq_len position

    output = self._pooling_layer(output)
    output = self._embedding_layer(output)
    # Output shape: (Batch_size, embedding_ dim)
    # Final; output of speaker embedding

    return output
