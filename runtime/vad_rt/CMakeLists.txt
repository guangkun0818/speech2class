enable_testing()

add_subdirectory(frontend)
add_subdirectory(model_interface)

# vad_session lib
add_library(vadsess_shared SHARED vad_session.cc) # shared
add_library(vadsess_static STATIC vad_session.cc) # static

target_link_libraries(vadsess_shared frontend_shared vadmodel_shared
                      ${TORCH_LIBRARIES} glog)
target_link_libraries(vadsess_static frontend_static vadmodel_static
                      ${TORCH_LIBRARIES} glog)

# Unittest of vad_session
add_executable(vadsess_test vad_session_test.cc)
target_link_libraries(vadsess_test vadsess_static gtest_main)
add_test(
  NAME vadsess_test
  COMMAND vadsess_test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/..)

# Build offline vad session
add_executable(build_vad_sess build_vad_sess.cc utils/thread_pool.h
                              utils/json.h)
target_link_libraries(build_vad_sess vadsess_static glog gflags)
