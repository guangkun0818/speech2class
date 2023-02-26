enable_testing()

include_directories(${TORCH_INCLUDE_DIRS})

# frontend lib
add_library(frontend_shared SHARED feature_extract.cc) # shared
add_library(frontend_static STATIC feature_extract.cc) # static

target_link_libraries(frontend_shared ${TORCH_LIBRARIES} glog)
target_link_libraries(frontend_static ${TORCH_LIBRARIES} glog)

# Unittest of Wav IO
add_executable(wav_test wav_test.cc)
target_link_libraries(wav_test gtest_main glog)
add_test(
  NAME wav_test
  COMMAND wav_test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/..)

# Unittest of FeatureExtractor
add_executable(frontend_test feature_extract_test.cc)
target_link_libraries(frontend_test frontend_static gtest_main)
add_test(
  NAME frontend_test
  COMMAND frontend_test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/..)
