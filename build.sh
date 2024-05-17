#!/bin/bash

set -e
set -x

if [ -d build ]; then
    rm -rf build
fi

if [ "$1" == "debug" ]; then
    echo "Building in debug mode"
    cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Debug && \
    cmake --build build
else
    echo "Building in release mode"
    cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build
fi