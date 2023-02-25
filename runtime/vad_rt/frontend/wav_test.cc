// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.02.25
// Unittest of Wav IO.

#include "frontend/wav.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

TEST(DummyTest, DummyAssertion) {
  EXPECT_EQ(1, 1);
  EXPECT_STREQ("test", "test");
}

TEST(WavIOTest, WavRead) {
  EXPECT_EQ(1, 1);
  std::string audio_path = "sample_data/data/wavs/251-136532-0007.wav";
  std::shared_ptr<vad_rt::frontend::WavReader> wav_reader(
      new vad_rt::frontend::WavReader());

  // Load Wav from audio_path.
  wav_reader->Open(audio_path);
  EXPECT_EQ(1, wav_reader->num_channel());
  EXPECT_EQ(16000, wav_reader->sample_rate());
  EXPECT_EQ(16, wav_reader->bits_per_sample());
  EXPECT_EQ(89680, wav_reader->num_samples());
  std::vector<float> pcms(wav_reader->data(),
                          wav_reader->data() + wav_reader->num_samples());
  EXPECT_EQ(pcms.size(), wav_reader->num_samples());

  // Load Wav from another audio_path.
  audio_path = "sample_data/data/wavs/652-130726-0030.wav";
  wav_reader->Open(audio_path);
  EXPECT_EQ(45520, wav_reader->num_samples());
  pcms.assign(wav_reader->data(),
              wav_reader->data() + wav_reader->num_samples());
  EXPECT_EQ(pcms.size(), wav_reader->num_samples());
}

TEST(WavIOTest, WavWrite) {
  EXPECT_EQ(1, 1);  // TODO: Impl Write test
}