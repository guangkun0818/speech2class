enable_testing()

include_directories(${TORCH_INCLUDE_DIRS})

# vad_model lib
add_library(vadmodel_shared SHARED vad_model.cc) # shared
add_library(vadmodel_static STATIC vad_model.cc) # static

target_link_libraries(vadmodel_shared ${TORCH_LIBRARIES} glog)
target_link_libraries(vadmodel_static ${TORCH_LIBRARIES} glog)

# Unittest of vad_model
add_executable(vadmodel_test vad_model_test.cc)
target_link_libraries(vadmodel_test vadmodel_static gtest_main glog)
add_test(
  NAME vadmodel_test
  COMMAND vadmodel_test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/..)
