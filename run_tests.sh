#!/bin/bash

cmake -B build -GNinja -DBUILD_TESTS=ON && \
cmake --build build && \
ctest -V --test-dir build/tests