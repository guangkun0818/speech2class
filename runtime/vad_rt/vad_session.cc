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
  do_post_process_ = opts->do_post_process;
  window_size_ = opts->window_size;
  speech_start_thres_ = opts->speech_start_thres;
  speech_end_thres_ = opts->speech_end_thres;

  LOG_IF(INFO, do_post_process_)
      << "Window size: " << window_size_
      << " Speech start threshold: " << speech_start_thres_
      << " Speech end threshold: " << speech_end_thres_;
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
  // Clear Check Window
  while (!session_state_->check_window.empty()) {
    session_state_->check_window.pop_front();
  }
  // Set speech not start
  session_state_->has_speech_start = false;
  // Reset speech_frames_count
  session_state_->speech_f_count = 0;
}

void VadSession::SessionStart() {
  // Reset Session states, clear final_result and results_cache,
  Reset();
}

void VadSession::FinalizeSession() {
  // Export all cached results out, Reset Session.
  while (!session_state_->check_window.empty()) {
    if (session_state_->has_speech_start) {
      // Speech has't end yet, still export as speech
      this->results_.push_back(1);
    } else {
      // Speech has ended, export as non-speech
      this->results_.push_back(0);
    }
    session_state_->check_window.pop_front();
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
  int resi_size = (pcms_ready.size() - this->feat_pipeline_->FrameSamples()) %
                  this->feat_pipeline_->FrameShiftSamples();
  int frame_res = this->feat_pipeline_->FrameSamples() -
                  this->feat_pipeline_->FrameShiftSamples();
  int offset = std::min<int>(pcms_ready.size(),
                             /*15ms + residual*/ frame_res + resi_size);

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

void VadSession::NoPostProcess() {
  // No-post-processing
  while (!session_state_->results_cache.empty()) {
    this->results_.push_back(session_state_->results_cache
                                 .front());  // Directly move to final result
    session_state_->results_cache.pop();
  }
}

void VadSession::SlidingWindow() {
  // ----- Sliding Window post-processing -----

  // This means the Vad session usually won't output result of current input
  // pcms immediately. Because most post-process of vad systems require infos
  // of both history and future to get final decision.
  // ---------------------------------------------------------------------
  // NOTE: Demo config of post-processing as follow, considering of Asr task as
  // down-streaming task, Vad system should encourage speech easily start and
  // hardly end covering enough infos for Asr by setting lower
  // speech_start_thres and higher speech_end_thres respectively.
  // "post_process": {
  //    "do_post_process": true,
  //    "window_size": 30,
  //    "speech_start_thres": 0.5,
  //    "speech_end_thres": 0.9
  // }
  // ----------------------------------------------------------------------
  // Sliding-Window check speech state from last time first, then update
  // check_window states with current raw_output from vad_model.
  while (!session_state_->results_cache.empty()) {
    if (session_state_->has_speech_start == false &&
        session_state_->check_window.size() == this->window_size_) {
      // If speech has not started yet and the num of speech frames within
      // check_window reach speech start threshold (speech_start_thres_ *
      // window_size), then start of speech and final result of all frames
      // within check_window determined. Export the whole check_window as
      // speech(result=1). This will regard several non-speech frames within
      // check_window as speech.
      if (session_state_->speech_f_count >
          this->speech_start_thres_ * this->window_size_) {
        // Speech start detected, export whole check_window as speech
        session_state_->has_speech_start = true;
        for (int i = 0; i < session_state_->check_window.size(); i++) {
          this->results_.push_back(1);
        }
        session_state_->check_window.clear();  // Clear check_window
        session_state_->speech_f_count = 0;    // Clear speech frames count

      } else {
        this->results_.push_back(
            0);  // Speech has not started, non-speech maintained.
      }
    } else if (session_state_->has_speech_start == true &&
               session_state_->check_window.size() == this->window_size_) {
      int non_speech_f_count =
          session_state_->check_window.size() - session_state_->speech_f_count;
      // If speech has started and the num of non-speech frames within
      // check_window reach speech end threshold (speech_end_thres *
      // window_size), then end of speech and final result of all frames within
      // check_window determined. Export the whole check_window as
      // speech(result=1). This will still regard several non-speech frames
      // within check_window as speech.
      if (non_speech_f_count > this->speech_end_thres_ * this->window_size_) {
        // Speech end detected, export whole check_window as speech
        session_state_->has_speech_start = false;
        for (int i = 0; i < session_state_->check_window.size(); i++) {
          this->results_.push_back(
              1);  // Export to final result, speech starting to end
        }
        session_state_->check_window.clear();  // Clear check_window
        session_state_->speech_f_count = 0;    // Clear speech frames count

      } else {
        this->results_.push_back(1);  // Final result, speech has not ended.
      }
    }

    // Update sliding-window states with current raw_output from vad pipeline.
    int raw_is_speech = session_state_->results_cache.front();
    session_state_->results_cache.pop();
    UpdateCheckWindow(raw_is_speech);
  }
}

void VadSession::UpdateCheckWindow(const int& is_speech) {
  // Push cache results into check_window.
  if (session_state_->check_window.size() < this->window_size_) {
    session_state_->check_window.push_back(is_speech);
    // If raw output is speech, is_speech = 1, update speech_f_count
    session_state_->speech_f_count += is_speech;
  } else {
    // Update speech_f_count if check_window reach its capacity
    session_state_->speech_f_count -= session_state_->check_window.front();
    session_state_->check_window.pop_front();
    session_state_->check_window.push_back(is_speech);
    session_state_->speech_f_count += is_speech;
  }
}

void VadSession::PostProcess() {
  // Post-process interface
  if (this->do_post_process_) {
    SlidingWindow();
  } else {
    NoPostProcess();
  }
}

void VadSession::Process(const std::vector<float>& pcms) {
  ProcessPcms(pcms);
  PostProcess();
}

}  // namespace vad_rt
