// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.02.25
// Frontend to extract feature from TorchScript model, exclude
// WavReader for thread-safety.

#include "frontend/feature_extract.h"

namespace vad_rt {
namespace frontend {

FeatureExtractor::FeatureExtractor(const std::string& frontend_path) {
  CHECK(!frontend_path.empty());
  LoadModel(frontend_path);
}

void FeatureExtractor::LoadModel(const std::string& frontend_path) {
  // Read saved TorchScript frontend as feature extractor.
  // Frontend Script require 1.12.1 libtorch for ops support.
  TorchModule model = torch::jit::load(frontend_path);
  extractor_ = std::make_shared<TorchModule>(std::move(model));
}

void FeatureExtractor::ExtractPcms(const std::vector<float>& pcms,
                                   Tensor& feats) {
  // Load vector based pcms into tensor.
  Tensor pcms_t = torch::from_blob(const_cast<float*>(pcms.data()), pcms.size(),
                                   torch::kFloat32);

  pcms_t = std::move(torch::div(
      pcms_t.unsqueeze(0),
      32768));  // torchaudio normalize=True, divide int16_t abs of limits 32768

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> input_pcms;
  input_pcms.push_back(std::move(pcms_t));

  // Execute the model and turn its output into a tensor.
  feats = extractor_->forward(input_pcms).toTensor();
}

}  // namespace frontend
}  // namespace vad_rt