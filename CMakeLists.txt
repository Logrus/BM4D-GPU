cmake_minimum_required(VERSION 2.8)
project(bm4d-gpu)

## Add support for C++14
set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED core video videoio highgui)

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CXX src/main.cpp
        src/bm4d.cpp
        src/allreader.cpp)
set(CU  src/kernels.cu)
set(H   include/bm4d-gpu/allreader.h
        include/bm4d-gpu/bm4d.h
        include/bm4d-gpu/helper_cuda.h
        include/bm4d-gpu/helper_string.h
        include/bm4d-gpu/kernels.cuh
        include/bm4d-gpu/parameters.h
        include/bm4d-gpu/stopwatch.hpp)
        
cuda_add_executable(bm4d-gpu ${CXX} ${CU} ${H})
target_link_libraries(bm4d-gpu ${OpenCV_LIBRARIES})
