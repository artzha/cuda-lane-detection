#!/bin/bash
USER=$(whoami)

docker run -it --net=host \
    --gpus all \
    -v $(pwd):/root/cuda-lane-detection \
    cudalanedet