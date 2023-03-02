// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.02.25
// Vad session impl. Audio and vad session should be
// one-to-one correspondence. thread-safety ensured.

#include <deque>
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
  std::string frontend_path;     // frontend.script path
  std::string vad_model_path;    // vad_model.script path
  float speech_thres;            // Threshold of logits determining speech.
  bool do_post_process = false;  // Specify whether do post-process
  size_t window_size = -1;       // Check-window size of post-process
  float switch_thres = -1;       // Speech start end transit threshold

  // TODO: More configs.
};

struct VadResource {
  std::shared_ptr<model_interface::VadModel> vad_model;  // VadModel
  std::shared_ptr<frontend::FeatureExtractor>
      feats_extract;  // feature_pipeline
  // Build Vad Resource from inputs
  VadResource(const std::string& frontend_path,
              const std::string& vad_model_path,
              float speech_thres /*speech threshold*/) {
    vad_model = std::make_shared<model_interface::VadModel>(
        std::move(model_interface::VadModel(vad_model_path, speech_thres)));
    feats_extract = std::make_shared<frontend::FeatureExtractor>(
        std::move(frontend::FeatureExtractor(frontend_path)));
  }
};

struct SessionState {
  // Vad model cache for last input
  torch::jit::IValue model_cache;
  // 25ms frame size, 18ms frameshift, therefore buffer 15ms
  // for framing accuracy during streaming inference.
  std::vector<float> pcm_buffer;
  float timestamp;
  std::queue<int> results_cache;

  // ----- Post processing -----
  // Sliding window to determin speech onset and offset
  std::deque<int> check_window;
  size_t speech_f_count;  // Count speech frame within check_window
  bool has_speech_start;  // Indication speech start
};

class VadSession {
 public:
  /*
    The Whole process would be like:
      SessionStart();
      for (;;) {
        Process(pcms) {
          ProcessPcms(pcms):
          PostProcess();
        };
        GetResults();
      };
      FinalizeSession();
    Output SessionState.final_result, could be empty.
  */
  explicit VadSession(const std::shared_ptr<VadSessionOpts>& opts,
                      const std::shared_ptr<VadResource>& resource);
  VadSession(const VadSession&) = delete;

  void SessionStart();

  // When session ends, export all cached results out
  void FinalizeSession();

  // Entrypoint of vad process
  void Process(const std::vector<float>& pcms);

  // Interface to get results
  const std::vector<int> GetResults() { return std::move(this->results_); }

 protected:
  // Reset SessionState.
  void Reset();

  // Post processing of cached-result;
  void PostProcess();

  bool do_post_process_;
  std::shared_ptr<model_interface::VadModel> vad_model_;
  std::shared_ptr<frontend::FeatureExtractor> feat_pipeline_;
  std::unique_ptr<SessionState> session_state_;

 private:
  // Process input pcms.
  void ProcessPcms(const std::vector<float>& pcms);

  // No-post-process impl
  void NoPostProcess();

  // Sliding window post-processing
  void SlidingWindow();

  // Update CheckWindow with every raw output of vad model, used inn
  // SlidingWindow
  void UpdateCheckWindow(const int& is_speech);

  std::vector<int> results_;

  // Config related with post-process
  size_t window_size_;
  float switch_thres_;
};

}  // namespace vad_rt
