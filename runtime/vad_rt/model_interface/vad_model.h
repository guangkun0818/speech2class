// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.02.25
// Vad model interface supporting streaming inference,
// thread-safety ensured.

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "torch/script.h"

namespace vad_rt {
namespace model_interface {

using TorchModule = torch::jit::script::Module;
using Tensor = torch::Tensor;

class VadModel {
 public:
  VadModel(const std::string& model_path, float threshold) {
    TorchModule model = torch::jit::load(model_path);
    vad_model_ = std::make_shared<TorchModule>(std::move(model));
    threshold_ = threshold;  // Threshold to determine is speech.
  }

  // Initallze vad model cache when VAD Session start.
  auto InitCache() {
    torch::jit::IValue cache = vad_model_->run_method("initialize_cache");
    return cache;
  }

  // Output of VadModel. Support non-streaming and streaming with
  // any chunksize even frame-level.
  void Inference(Tensor& feats, torch::jit::IValue& cache, Tensor& logits) {
    auto output = vad_model_->run_method("inference", feats, cache).toTuple();
    logits = output->elements()[0].toTensor();  // Logits.
    cache = output->elements()[1].toTuple();    // Cache for next frame.
  }

  inline int IsSpeech(Tensor& logits) {
    // Logits (1, 1, 2) of one frame, where index=1 of last dim of indicating
    // speech if greater than threshold.
    CHECK_EQ(logits.dim(), 3);
    CHECK_EQ(logits.size(0), 1);
    CHECK_EQ(logits.size(1), 1);
    CHECK_EQ(logits.size(2), 2);
    auto accessor =
        logits
            .accessor</*type=float*/ float, /*dim=3*/ 3>();  // More efficient?

    return accessor[0][0][1] > threshold_;  // Speech index=1
  }

 private:
  std::shared_ptr<TorchModule> vad_model_;
  float threshold_;
};

}  // namespace model_interface
}  // namespace vad_rt
