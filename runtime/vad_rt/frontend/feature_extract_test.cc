// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.02.25
// Unittest of frontend.

#include "frontend/feature_extract.h"

#include "gtest/gtest.h"

using namespace vad_rt::frontend;

class FrontendTest : public ::testing::Test {
 protected:
  void SetUp() {
    feat_extractor_ = std::make_shared<FeatureExtractor>(std::move(
        FeatureExtractor("sample_data/model/demo_task/frontend.script")));
    wav_reader_ = std::make_shared<WavReader>();
  }
  std::shared_ptr<FeatureExtractor> feat_extractor_;
  std::shared_ptr<WavReader> wav_reader_;
};

TEST_F(FrontendTest, ExtractorBuild) {
  // Frontend Script require 1.12.1 libtorch for ops support.
  EXPECT_NE(feat_extractor_->FrontendPtr(), nullptr);
  EXPECT_EQ(feat_extractor_->FrameShift(), 10);
  EXPECT_EQ(feat_extractor_->FrameLength(), 25);
  EXPECT_EQ(feat_extractor_->SampleRate(), 16000);
  EXPECT_EQ(feat_extractor_->FrameSamples(), 400);
  EXPECT_EQ(feat_extractor_->NumChannel(), 1);
  EXPECT_EQ(feat_extractor_->BitsPerSample(), 16);
}

TEST_F(FrontendTest, FeatureExtract) {
  // Frontend Script require 1.12.1 libtorch for ops support.
  EXPECT_NE(feat_extractor_->FrontendPtr(), nullptr);

  std::string audio_path = "sample_data/data/wavs/251-136532-0007.wav";
  wav_reader_->Open(audio_path);
  std::vector<float> pcms(wav_reader_->data(),
                          wav_reader_->data() + wav_reader_->num_samples());

  torch::Tensor feats;
  feat_extractor_->ExtractPcms(pcms, feats);
  // Expected feat size
  c10::IntArrayRef expect_size({559, 64});
  EXPECT_EQ(feats.sizes(), expect_size);

  pcms.clear();
  audio_path = "sample_data/data/wavs/652-130726-0030.wav";
  wav_reader_->Open(audio_path);
  pcms.assign(wav_reader_->data(),
              wav_reader_->data() + wav_reader_->num_samples());
  feat_extractor_->ExtractPcms(pcms, feats);
  // Expected feat size
  expect_size = {283, 64};
  EXPECT_EQ(feats.sizes(), expect_size);
}