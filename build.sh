#!/bin/bash

set -e
set -x

cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release && \
cmake --build build