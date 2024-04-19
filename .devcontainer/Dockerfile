FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libopencv-dev \
    python3-opencv
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    git \
    ninja-build
