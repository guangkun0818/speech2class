// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.02.25
// Unittest of Vad Session.

#include "vad_session.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace vad_rt;

// ------------ Unittest of SlidingWindow -----------------
class TestVadSessionSlidingWindow : public ::testing::Test {
 protected:
  void SetUp() {
    opts = std::make_shared<VadSessionOpts>();
    opts->frontend_path = "sample_data/model/demo_task/frontend.script";
    opts->vad_model_path = "sample_data/model/crdnn_int8.script";
    opts->speech_thres = 0.5;  // speech > threshold

    opts->do_post_process = true;  // Post-process specified
    opts->window_size = 30;
    opts->speech_start_thres = 0.5;
    opts->speech_end_thres = 0.9;

    vad_resource = std::make_shared<VadResource>(VadResource(
        opts->frontend_path, opts->vad_model_path, opts->speech_thres));
    vad_session = std::move(
        std::unique_ptr<VadSession>(new VadSession(opts, vad_resource)));
  }

  std::shared_ptr<VadResource> vad_resource;
  std::shared_ptr<VadSessionOpts> opts;
  std::unique_ptr<VadSession> vad_session;
};

TEST_F(TestVadSessionSlidingWindow, TestSessionPipelineNonStreaming) {
  std::vector<int> result;

  Tensor sample_wav = torch::rand({1, 1600000}).contiguous();
  std::vector<float> sample_pcms(
      sample_wav.data_ptr<float>(),
      sample_wav.data_ptr<float>() + sample_wav.numel());
  vad_session->SessionStart();
  vad_session->Process(sample_pcms);
  vad_session->FinalizeSession();
  result = vad_session->GetResults();
  EXPECT_EQ(result.size(), 9998);
}

TEST_F(TestVadSessionSlidingWindow, TestSessionStreaming) {
  std::vector<int> result;
  std::vector<int> chunk_result;

  Tensor sample_wav = torch::rand({1, 16000}).contiguous();
  std::vector<float> sample_pcms(
      sample_wav.data_ptr<float>(),
      sample_wav.data_ptr<float>() + sample_wav.numel());

  vad_session->SessionStart();
  for (int i = 0; i < 100; i++) {
    // Simulated streaming session
    vad_session->Process(sample_pcms);
    chunk_result = vad_session->GetResults();
    result.insert(result.end(), chunk_result.begin(), chunk_result.end());
  }

  vad_session->FinalizeSession();
  chunk_result = vad_session->GetResults();
  result.insert(result.end(), chunk_result.begin(), chunk_result.end());

  EXPECT_EQ(result.size(), 9998);
}

TEST_F(TestVadSessionSlidingWindow, TestSessionStreamingWithResidual) {
  std::vector<int> result;
  std::vector<int> chunk_result;
  // 16123 samples, must have residual.
  Tensor sample_wav = torch::rand({1, 16123}).contiguous();
  std::vector<float> sample_pcms(
      sample_wav.data_ptr<float>(),
      sample_wav.data_ptr<float>() + sample_wav.numel());

  vad_session->SessionStart();
  for (int i = 0; i < 100; i++) {
    // Simulated streaming session
    vad_session->Process(sample_pcms);
    chunk_result = vad_session->GetResults();
    result.insert(result.end(), chunk_result.begin(), chunk_result.end());
  }

  vad_session->FinalizeSession();
  chunk_result = vad_session->GetResults();
  result.insert(result.end(), chunk_result.begin(), chunk_result.end());

  EXPECT_EQ(result.size(), 10075);
}

// ------------ Unittest of NoPostProcess -----------------
class TestVadSessionNoPostProcess : public ::testing::Test {
 protected:
  void SetUp() {
    opts = std::make_shared<VadSessionOpts>();
    opts->frontend_path = "sample_data/model/demo_task/frontend.script";
    opts->vad_model_path = "sample_data/model/crdnn_int8.script";
    opts->speech_thres = 0.5;  // speech > threshold

    opts->do_post_process = false;  // No Post-process specified

    vad_resource = std::make_shared<VadResource>(VadResource(
        opts->frontend_path, opts->vad_model_path, opts->speech_thres));
    vad_session = std::move(
        std::unique_ptr<VadSession>(new VadSession(opts, vad_resource)));
  }

  std::shared_ptr<VadResource> vad_resource;
  std::shared_ptr<VadSessionOpts> opts;
  std::unique_ptr<VadSession> vad_session;
};

TEST_F(TestVadSessionNoPostProcess, TestSessionPipelineNonStreaming) {
  std::vector<int> result;

  Tensor sample_wav = torch::rand({1, 1600000}).contiguous();
  std::vector<float> sample_pcms(
      sample_wav.data_ptr<float>(),
      sample_wav.data_ptr<float>() + sample_wav.numel());
  vad_session->SessionStart();
  vad_session->Process(sample_pcms);
  vad_session->FinalizeSession();
  result = vad_session->GetResults();
  EXPECT_EQ(result.size(), 9998);
}

TEST_F(TestVadSessionNoPostProcess, TestSessionStreaming) {
  std::vector<int> result;
  std::vector<int> chunk_result;

  Tensor sample_wav = torch::rand({1, 16000}).contiguous();
  std::vector<float> sample_pcms(
      sample_wav.data_ptr<float>(),
      sample_wav.data_ptr<float>() + sample_wav.numel());

  vad_session->SessionStart();
  for (int i = 0; i < 100; i++) {
    // Simulated streaming session
    vad_session->Process(sample_pcms);
    chunk_result = vad_session->GetResults();
    result.insert(result.end(), chunk_result.begin(), chunk_result.end());
  }

  vad_session->FinalizeSession();
  chunk_result = vad_session->GetResults();
  result.insert(result.end(), chunk_result.begin(), chunk_result.end());

  EXPECT_EQ(result.size(), 9998);
}

TEST_F(TestVadSessionNoPostProcess, TestSessionStreamingWithResidual) {
  std::vector<int> result;
  std::vector<int> chunk_result;
  // 16123 samples, must have residual.
  Tensor sample_wav = torch::rand({1, 16123}).contiguous();
  std::vector<float> sample_pcms(
      sample_wav.data_ptr<float>(),
      sample_wav.data_ptr<float>() + sample_wav.numel());

  vad_session->SessionStart();
  for (int i = 0; i < 100; i++) {
    // Simulated streaming session
    vad_session->Process(sample_pcms);
    chunk_result = vad_session->GetResults();
    result.insert(result.end(), chunk_result.begin(), chunk_result.end());
  }

  vad_session->FinalizeSession();
  chunk_result = vad_session->GetResults();
  result.insert(result.end(), chunk_result.begin(), chunk_result.end());

  EXPECT_EQ(result.size(), 10075);
}