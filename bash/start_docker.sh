#!/bin/bash
USER=$(whoami)

# docker run -it --net=host \
#     --gpus all \
#     -v $(pwd):/root/cuda-lane-detection \
#     cudalanedet

# docker run -it --net=host \
#     -v $(pwd):/root/cuda-lane-detection \
#     cudalanedet


# NVIDIA ORIN SET UP
docker run --net=host --runtime=nvidia -it --rm \
	-v /usr/local/cuda:/usr/local/cuda \
        -v $(pwd):/root/cuda-lane-detection \
        cudalanedet
