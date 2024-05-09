#!/bin/bash

set -e
set -x

cmake -B build -GNinja && \
cmake --build build