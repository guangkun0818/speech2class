cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(
  vad_runtime
  LANGUAGES CXX C
  VERSION 0.1)

# libTorch requires C++14 at least
set(CMAKE_CXX_STANDARD 17)

include(CTest)

include(FetchContent)
set(FETCHCONTENT_QUIET off)
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/3rd_party")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(gflags)
include(glog)
include(googletest)
include(libtorch)

# Delare header include root directory of project
include_directories(vad_rt)

add_subdirectory(vad_rt)
