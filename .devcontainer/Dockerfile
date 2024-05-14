ARG BASE_IMAGE
FROM $BASE_IMAGE

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    gdb \
    git \
    libopencv-dev \
    ninja-build \
    python3-opencv \
    python3-pip \
    python3-setuptools \
    python3-wheel

RUN python3 -m pip install --upgrade pip && \
    pip install cmake_format

ARG GROUP_NAME
ARG USER_NAME
RUN groupadd -r $GROUP_NAME && useradd -r -g $GROUP_NAME $USER_NAME

USER $USER_NAME