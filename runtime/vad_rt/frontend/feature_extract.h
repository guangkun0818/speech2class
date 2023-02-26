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

 private:
  // Load saved frontend torchscript model;
  void LoadModel(const std::string& frontend_path);

  std::shared_ptr<TorchModule> extractor_ = nullptr;
};

}  // namespace frontend
}  // namespace vad_rt