cmake_minimum_required(VERSION 3.22)
project(bm4d-gpu)

# Add support for C++14
set(CMAKE_CXX_STANDARD 14)

option(BUILD_CLI "Build cli (requires OpenCV)" ON)

# BM4D library
find_package(CUDA REQUIRED)

include_directories(include)

set(BM4D-H
    include/bm4d-gpu/bm4d.h
    include/bm4d-gpu/helper_cuda.h
    include/bm4d-gpu/helper_string.h
    include/bm4d-gpu/kernels.cuh
    include/bm4d-gpu/parameters.h
    include/bm4d-gpu/stopwatch.hpp
    include/bm4d-gpu/utils.h)

set(BM4D-CXX src/bm4d.cpp)
set(BM4d-CU src/kernels.cu)

cuda_add_library(bm4d-gpu-lib ${BM4D-CXX} ${BM4d-CU} ${BM4D-H})

# BM4D CLI
if(BUILD_CLI)
  include_directories(${OpenCV_INCLUDE_DIRS})
  find_package(OpenCV REQUIRED core video videoio highgui)

  set(CLI-CXX src/allreader.cpp src/main.cpp)
  set(CLI-H include/bm4d-gpu/allreader.h)
  cuda_add_executable(bm4d-gpu ${CLI-CXX} ${CLI-H})
  target_link_libraries(bm4d-gpu bm4d-gpu-lib ${OpenCV_LIBRARIES})
endif()
