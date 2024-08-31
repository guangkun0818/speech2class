// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.02.25
// Build Vad session, designed for algo offline test.

#include <experimental/filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "utils/json.h"
#include "utils/thread_pool.h"
#include "vad_session.h"

DEFINE_string(session_conf, "runtime/config/example.json",
              "Config of Vad session.");
DEFINE_string(dataset_json, "runtime/config/test_data.json",
              "Dataset.json, Test dataset.");
DEFINE_string(export_path, "test_logs/engine_test/",
              "Path to export vad results.");
DEFINE_int32(num_thread, 4, "Num of threads of offline vad session.");

// Global resources for threading vad inference.
namespace fs = std::experimental::filesystem;
std::shared_ptr<vad_rt::VadSessionOpts> opts;
std::shared_ptr<vad_rt::VadResource> vad_resource;
std::mutex g_mutex;  // Beware that lock thread-non-safety parts

namespace vad_rt {

using utils::json::JSON;

void LoadConf(const std::string& session_conf, JSON& conf) {
  std::ifstream sess_conf_f(session_conf);
  std::string line, conf_infos;
  while (std::getline(sess_conf_f, line)) {
    conf_infos += line;
  }
  conf = JSON::Load(conf_infos);
  CHECK(conf.hasKey("frontend") && conf.hasKey("vad_model"));
  CHECK(conf.hasKey("speech_thres"));
  CHECK(conf.hasKey("post_process"));
  CHECK(conf["post_process"].hasKey("do_post_process") &&
        conf["post_process"].hasKey("window_size") &&
        conf["post_process"].hasKey("speech_start_thres") &&
        conf["post_process"].hasKey("speech_end_thres"));
}

void LoadOpts(JSON& conf, const std::shared_ptr<VadSessionOpts>& opts) {
  opts->frontend_path = conf["frontend"].ToString();
  CHECK(fs::exists(opts->frontend_path));
  opts->vad_model_path = conf["vad_model"].ToString();
  CHECK(fs::exists(opts->vad_model_path));
  opts->speech_thres = conf["speech_thres"].ToFloat();

  if (conf["post_process"]["do_post_process"].ToBool()) {
    opts->do_post_process = conf["post_process"]["do_post_process"].ToBool();
    opts->window_size = conf["post_process"]["window_size"].ToInt();
    opts->speech_start_thres =
        conf["post_process"]["speech_start_thres"].ToFloat();
    opts->speech_end_thres = conf["post_process"]["speech_end_thres"].ToFloat();
  }
}

/*
  Vad Session for offline test.
*/
class VadSessionOffline : public VadSession {
 public:
  explicit VadSessionOffline(const std::shared_ptr<VadSessionOpts>& opts,
                             const std::shared_ptr<VadResource>& resource)
      : VadSession(opts, resource) {
    wav_reader_ = std::make_unique<frontend::WavReader>();
    LOG_IF(INFO, this->do_post_process_)
        << "Offline Vad session built, Post-process will be applied.";
    LOG_IF(INFO, !this->do_post_process_)
        << "Offline Vad session built, No Post-process specified.";
  }

  // Vad process through audio file.
  void ProcessAudioFile(const std::string& audio_path) {
    wav_reader_->Open(audio_path);
    std::vector<float> pcms;
    pcms.assign(wav_reader_->data(),
                wav_reader_->data() + wav_reader_->num_samples());

    LOG(INFO) << audio_path << " processing....";
    this->Process(pcms);
    LOG(INFO) << audio_path << " Done.";
  }

  void ExportProcessedAudio(const std::string& input_audio_path,
                            const std::string& export_audio_path,
                            const std::vector<int>& vad_results) {
    // Read origin wave data.
    wav_reader_->Open(input_audio_path);
    LOG(INFO) << "Origin duration: "
              << float(wav_reader_->num_samples()) / wav_reader_->sample_rate()
              << "s.";

    float* export_data = new float[wav_reader_->num_samples()];
    int start = 0;
    int end = start + this->feat_pipeline_->FrameSamples();
    int num_export_samples = 0;

    // Read sample with vad frame-level result = 1
    for (auto frame_res : vad_results) {
      if (frame_res) {
        memcpy(/*export_pcm.end()=*/export_data + num_export_samples,
               /*frame.start()=*/wav_reader_->data() + start,
               /*num_frame_length=*/(end - start) * sizeof(float));
        num_export_samples += (end - start);
      }
      start = std::max(end, start + this->feat_pipeline_->FrameShiftSamples());
      end += this->feat_pipeline_->FrameShiftSamples();
    }

    auto wav_writer = frontend::WavWriter(
        export_data, num_export_samples, this->feat_pipeline_->NumChannel(),
        this->feat_pipeline_->SampleRate(),
        this->feat_pipeline_->BitsPerSample());
    wav_writer.Write(export_audio_path);

    LOG(INFO) << "Duration after Vad: "
              << float(num_export_samples) / this->feat_pipeline_->SampleRate()
              << "s.";
    LOG(INFO) << "Export audio processed by VAD to " << export_audio_path;
    delete export_data;  // Free ptr.
  }

 private:
  std::unique_ptr<frontend::WavReader> wav_reader_;
};

/*
  Thread entrypoint of vad inference.
*/
void VadInference(const std::pair<std::string, std::string>& wav_scp,
                  fs::path& export_path) {
  std::unique_ptr<VadSessionOffline> vad_sess(
      new VadSessionOffline(opts, vad_resource));
  LOG(INFO) << "Offline Vad session built.";

  std::vector<int> vad_result;
  vad_sess->SessionStart();
  vad_sess->ProcessAudioFile(wav_scp.second);
  vad_sess->FinalizeSession();
  vad_result = vad_sess->GetResults();

  // Export vad processed audio.
  auto export_audio = export_path / fs::path(wav_scp.first + "_processed.wav");
  vad_sess->ExportProcessedAudio(wav_scp.second, export_audio, vad_result);

  // Save results with torch::save as Tensor for metrics compute.
  Tensor vad_result_t =
      torch::from_blob(vad_result.data(), vad_result.size(), torch::kInt32);

  // Result = "utt.vad"
  fs::path save_path = export_path / fs::path(wav_scp.first + ".vad");
  auto bytes = torch::jit::pickle_save(vad_result_t);
  std::ofstream vad_res_f(save_path, std::ios::out | std::ios::binary);
  vad_res_f.write(bytes.data(), bytes.size());
  vad_res_f.close();
}

}  // namespace vad_rt

int main(int argc, char** argv) {
  FLAGS_logtostderr = 1;
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "Session Conf: " << FLAGS_session_conf;
  vad_rt::utils::json::JSON conf;
  vad_rt::LoadConf(FLAGS_session_conf, conf);

  // Load Vad opts.
  opts = std::make_shared<vad_rt::VadSessionOpts>();
  vad_rt::LoadOpts(conf, opts);

  // Load Vad resources.
  vad_resource =
      std::make_shared<vad_rt::VadResource>(std::move(vad_rt::VadResource(
          conf["frontend"].ToString(), conf["vad_model"].ToString(),
          conf["speech_thres"].ToFloat())));

  // Build vad session.
  fs::path export_path = FLAGS_export_path;
  vad_rt::utils::ThreadPool thread_pool(FLAGS_num_thread);
  std::ifstream datamap(FLAGS_dataset_json);
  std::string line;
  while (std::getline(datamap, line)) {
    conf = vad_rt::JSON::Load(line);
    CHECK(conf.hasKey("utt"));
    CHECK(conf.hasKey("audio_filepath"));
    LOG(INFO) << conf["utt"].ToString() << " utterance loaded.";
    auto wav_scp = std::make_pair(conf["utt"].ToString(),
                                  conf["audio_filepath"].ToString());

    thread_pool.enqueue(vad_rt::VadInference, wav_scp, export_path);
  }

  return 0;
}