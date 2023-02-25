enable_testing()

include_directories(${TORCH_INCLUDE_DIRS})

# Unittest of Wav IO
add_executable(wav_test wav_test.cc)
target_link_libraries(wav_test gtest_main glog)
add_test(
  NAME wav_test
  COMMAND wav_test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/..)
