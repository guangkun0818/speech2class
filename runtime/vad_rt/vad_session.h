// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.02.25
// Vad session impl. Audio and vad session should be
// one-to-one correspondence. thread-safety ensured.

#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "frontend/feature_extract.h"
#include "glog/logging.h"
#include "model_interface/vad_model.h"
#include "torch/torch.h"

namespace vad_rt {

using Tensor = torch::Tensor;

struct VadSessionOpts {
  std::string frontend_path;   // frontend.script path
  std::string vad_model_path;  // vad_model.script path
  float threshold;  // Threshold of logits determining speech within vad model.
  size_t res_cache_capacity;  // Cache size of final Result of Vad Session for
                              // post-process

  // TODO: More configs.
};

struct VadResource {
  std::shared_ptr<model_interface::VadModel> vad_model;  // VadModel
  std::shared_ptr<frontend::FeatureExtractor>
      feats_extract;  // feature_pipeline
  // Build Vad Resource from inputs
  VadResource(const std::string& frontend_path,
              const std::string& vad_model_path,
              float threshold /*speech threshold*/) {
    vad_model = std::make_shared<model_interface::VadModel>(
        std::move(model_interface::VadModel(vad_model_path, threshold)));
    feats_extract = std::make_shared<frontend::FeatureExtractor>(
        std::move(frontend::FeatureExtractor(frontend_path)));
  }
};

struct SessionState {
  torch::jit::IValue model_cache;  // Vad model cache for last input
  // 25ms frame size, 18ms frameshift, therefore buffer 15ms
  // for framing accuracy during streaming inference.
  std::vector<float> pcm_buffer;
  float timestamp;
  std::queue<int> results_cache;
};

class VadSession {
 public:
  explicit VadSession(const std::shared_ptr<VadSessionOpts>& opts,
                      const std::shared_ptr<VadResource>& resource);
  VadSession(const VadSession&) = delete;

  void SessionStart();

  // When session ends, export all cached results out
  void FinalizeSession();

  /*
    The Whole process would be like:
      ProcessPcms (pcms):
      PostProcess();
      FinalizeResults ();
      GetResults ();
    Output SessionState.final_result, could be empty.
  */
  void Process(const std::vector<float>& pcms);

  // Interface to get results
  const std::vector<int> GetResults() { return std::move(this->results_); }

 protected:
  // Reset SessionState.
  void Reset();

  // Post processing of cached-result;
  void PostProcess();

  // Process results if there is final result ready for output
  void FinalizeResults();

  std::shared_ptr<model_interface::VadModel> vad_model_;
  std::shared_ptr<frontend::FeatureExtractor> feat_pipeline_;
  std::unique_ptr<SessionState> session_state_;

 private:
  // Process input pcms.
  void ProcessPcms(const std::vector<float>& pcms);

  std::vector<int> results_;
  int res_cache_capacity_;
};

}  // namespace vad_rt
