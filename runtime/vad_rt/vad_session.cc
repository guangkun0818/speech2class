// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.02.25
// Vad session impl. Audio and vad session should be
// one-to-one correspondence. thread-safety ensured.

#include "vad_session.h"

#include <algorithm>

namespace vad_rt {

VadSession::VadSession(const std::shared_ptr<VadSessionOpts>& opts,
                       const std::shared_ptr<VadResource>& resource) {
  vad_model_ = resource->vad_model;
  feat_pipeline_ = resource->feats_extract;
  session_state_ = std::make_unique<SessionState>();
  res_cache_capacity_ = opts->res_cache_capacity;
}

void VadSession::Reset() {
  // Reinit model cache.
  session_state_->model_cache = vad_model_->InitCache();
  // Clear Pcm buffer.
  session_state_->pcm_buffer.clear();
  // Reset Timstamp tracking
  session_state_->timestamp = 0.0;
  // Clear results_cache.
  while (!session_state_->results_cache.empty()) {
    session_state_->results_cache.pop();
  }
}

void VadSession::SessionStart() {
  // Reset Session states, clear final_result and results_cache,
  Reset();
}

void VadSession::FinalizeSession() {
  // Export all cached results out, Reset Session.
  while (!session_state_->results_cache.empty()) {
    this->results_.push_back(
        static_cast<bool>(session_state_->results_cache.front()));
    session_state_->results_cache.pop();
  }
  Reset();
}

void VadSession::ProcessPcms(const std::vector<float>& pcms) {
  std::vector<float> pcms_ready;
  pcms_ready.insert(
      pcms_ready.end(), session_state_->pcm_buffer.begin(),
      session_state_->pcm_buffer.end());  // Left padding pcm_buffer
  session_state_->pcm_buffer.clear();

  // Concat pcm buffer with input pcms, update pcm_buffer.
  pcms_ready.insert(pcms_ready.end(), pcms.begin(), pcms.end());

  // Framing strategy:
  // bound--------------------bound
  //   |----25ms----|-10ms-|    |
  //   |      |----25ms----|    |
  //   |             |----25ms--|--|
  // --------------------------------
  // The last frame will be discard when frame shift beyond the bound of given
  // pcms, therefore, pcm_buffer not only need to cache 240 samples (25ms -
  // 10ms) but also residual of discarded pcms.
  int resi_size = (pcms_ready.size() - 400) % 160;
  int offset =
      std::min<int>(pcms_ready.size(), /*15ms + residual*/ 240 + resi_size);

  session_state_->pcm_buffer.insert(session_state_->pcm_buffer.end(),
                                    pcms_ready.end() - offset,
                                    pcms_ready.end());
  if (pcms_ready.size() < 400) {
    // 16000 sample rate, 25ms for frame, 400 samples.
    LOG(INFO) << "Pcm less than 400 samples, override process";
    return;
  }

  Tensor feats;
  feat_pipeline_->ExtractPcms(pcms_ready,
                              feats);  // Feats Extract (seq_len, feat_dim)
  Tensor logits;
  feats.unsqueeze_(0);  // (1, seqlen, feat _dim)
  vad_model_->Inference(feats, session_state_->model_cache,
                        logits);  // Vad model Inference.

  // Cache raw output of vad model.
  int frame_len = logits.size(1);
  for (int i = 0; i < frame_len; i++) {
    auto frame_logit = logits.slice(1, i, i + 1, 1);  // (1, T, 2) -> (1, 1, 2)
    session_state_->results_cache.push(vad_model_->IsSpeech(frame_logit));
  }
}

void VadSession::PostProcess() { /* TODO(xiaoyue.yang@transsion.com) Impl
                                    post-process */
}

void VadSession::FinalizeResults() {
  // Pop out cached result if it reach predefined capacity of result_buffer.

  // This means the Vad session usually won't output result of current input
  // pcms immediately. Because most post-process of vad systems require infos
  // of both history and future to get final decision.
  while (session_state_->results_cache.size() >= res_cache_capacity_) {
    this->results_.push_back(
        static_cast<bool>(session_state_->results_cache.front()));
    session_state_->results_cache.pop();
  }
}

void VadSession::Process(const std::vector<float>& pcms) {
  ProcessPcms(pcms);
  PostProcess();
  FinalizeResults();
}

}  // namespace vad_rt
