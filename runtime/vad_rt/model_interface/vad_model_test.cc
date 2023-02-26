// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.02.25
// Unittest of Vad model.

#include "model_interface/vad_model.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace vad_rt::model_interface;

class TestVadModel : public ::testing::Test {
 protected:
  void SetUp() {
    std::string model_path = "sample_data/model/crdnn_int8.script";
    vad_model_ =
        std::make_shared<VadModel>(std::move(VadModel(model_path, 0.8)));
  }

  std::shared_ptr<VadModel> vad_model_;
};

TEST_F(TestVadModel, InitCacheTest) { auto cache = vad_model_->InitCache(); }

TEST_F(TestVadModel, TestChunkStreamInference) {
  // Chunk size = 100
  auto cache = vad_model_->InitCache();  // Init Cache
  Tensor feats = torch::rand({1, 100, 64});
  Tensor logits;

  for (int i = 1; i < 100; i++) {
    // Cache will update when frame inputs.
    vad_model_->Inference(feats, cache, logits);
  }
}

TEST_F(TestVadModel, TestFramelevelStreamInference) {
  // Frame level streaming
  auto cache = vad_model_->InitCache();  // Init Cache
  Tensor feats = torch::rand({1, 1, 64});
  Tensor logits;

  for (int i = 1; i < 10000; i++) {
    // Cache will update when frame inputs.
    vad_model_->Inference(feats, cache, logits);
  }
}

TEST_F(TestVadModel, TestIsSpeech) {
  Tensor logits = torch::tensor({{{0.1, 0.9}}}, {torch::kFloat32});
  bool is_speech = vad_model_->IsSpeech(logits);
  EXPECT_TRUE(is_speech);  // 0.9 > threshold 0.8

  logits = torch::tensor({{{0.4, 0.6}}}, {torch::kFloat32});
  is_speech = vad_model_->IsSpeech(logits);
  EXPECT_FALSE(is_speech);  // 0.6 < threshold 0.8
}