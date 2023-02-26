// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.02.25
// Unittest of Vad Session.

#include "vad_session.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace vad_rt;

class TestVadSession : public ::testing::Test {
 protected:
  void SetUp() {
    opts = std::make_shared<VadSessionOpts>();
    opts->frontend_path = "sample_data/model/demo_task/frontend.script";
    opts->vad_model_path = "sample_data/model/crdnn_int8.script";

    // Result cache capacity, mainly for postprocess, if no post-proces applied,
    // set as 1
    opts->res_cache_capacity = 1;
    opts->threshold = 0.5;  // speech > threshold
    vad_resource = std::make_shared<VadResource>(VadResource(
        opts->frontend_path, opts->vad_model_path, opts->threshold));
    vad_session = std::move(
        std::unique_ptr<VadSession>(new VadSession(opts, vad_resource)));
  }

  std::shared_ptr<VadResource> vad_resource;
  std::shared_ptr<VadSessionOpts> opts;
  std::unique_ptr<VadSession> vad_session;
};

TEST_F(TestVadSession, TestSessionPipelineNoPostProcess) {
  std::vector<int> result;

  Tensor sample_wav = torch::rand({1, 1600000}).contiguous();
  std::vector<float> sample_pcms(
      sample_wav.data_ptr<float>(),
      sample_wav.data_ptr<float>() + sample_wav.numel());
  vad_session->SessionStart();
  vad_session->Process(sample_pcms);
  result = vad_session->GetResults();
  EXPECT_EQ(result.size(), 9998);
}

TEST_F(TestVadSession, TestSessionStreaming) {
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