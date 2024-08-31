// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.02.25
// Frontend to extract feature from TorchScript model, exclude
// WavReader for thread-safety.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "frontend/wav.h"
#include "glog/logging.h"
#include "torch/script.h"
#include "torch/torch.h"

namespace vad_rt {
namespace frontend {

using TorchModule = torch::jit::script::Module;
using Tensor = torch::Tensor;

class FeatureExtractor {
 public:
  // Instance should be constructed with frontend script path
  FeatureExtractor(const std::string& frontend_path);

  // Extract feats from given chunk of pcms
  void ExtractPcms(const std::vector<float>& pcms, Tensor& feats);

  const std::shared_ptr<TorchModule> FrontendPtr() const { return extractor_; }

  const float FrameShift() const { return this->frame_shift_; };

  const float FrameLength() const { return this->frame_length_; }

  const int SampleRate() const { return this->sample_rate_; }

  // Num of samples within frame.
  const int FrameSamples() const {
    return static_cast<int>(this->FrameLength() * this->SampleRate() / 1000);
  }

  // Num of samples within frame shift.
  const int FrameShiftSamples() const {
    return static_cast<int>(this->FrameShift() * this->SampleRate() / 1000);
  }

 private:
  // Load saved frontend torchscript model;
  void LoadModel(const std::string& frontend_path);

  std::shared_ptr<TorchModule> extractor_ = nullptr;

  float frame_shift_ = 10;   // Frame shift of frontend, in ms.
  float frame_length_ = 25;  // Frame length of frontend, in ms.
  int sample_rate_ = 16000;  // Sample rate of frontend
};

}  // namespace frontend
}  // namespace vad_rt