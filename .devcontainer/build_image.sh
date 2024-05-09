#!/usr/bin/env bash

SCRIPT_DIR=`dirname "$0"`

docker build -f $SCRIPT_DIR/Dockerfile . -t bm4d-gpu-devenv