FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    gdb \
    cmake \
    git \
    libopencv-dev \
    ninja-build \
    python3-opencv \
    python3-pip \
    python3-setuptools \
    python3-wheel


RUN python3 -m pip install --upgrade pip && \
    pip install cmake_format
