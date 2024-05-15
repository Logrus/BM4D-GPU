#!/usr/bin/env bash

SCRIPT_DIR=`dirname "$0"`

# The base image and host system should have the same CUDA version
# This script tries to extract the CUDA version from the nvcc --version output
# and then replaces the placeholder in the Dockerfile.template with the actual version.

# If it fails to fuilfil its purpose,
# you can always manually replace the placeholder with the actual version.

nvcc_output=$(nvcc --version 2>&1)

# Extract the release version using grep and sed
CUDA_VERSION=$(echo "$nvcc_output" | grep -oP 'release \K\d+\.\d+(\.\d+)?')

if [[ $(echo $CUDA_VERSION | awk -F'.' '{print NF}') -eq 2  ]]; then
    CUDA_VERSION="${CUDA_VERSION}.0"
fi

BASE_IMAGE="nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04"

echo "Detected cuda version: $CUDA_VERSION"

echo "Using base image: ${BASE_IMAGE}"

docker build --build-arg BASE_IMAGE=${BASE_IMAGE} \
             --build-arg USER_NAME=${USER} \
             --build-arg GROUP_NAME=${USER} \
             -t bm4d-gpu-devenv \
             -f $SCRIPT_DIR/Dockerfile . 