#!/usr/bin/env bash

SCRIPT_DIR=`dirname "$0"`

${SCRIPT_DIR}/dockerfile_from_template.sh

docker build -f $SCRIPT_DIR/Dockerfile . -t bm4d-gpu-devenv