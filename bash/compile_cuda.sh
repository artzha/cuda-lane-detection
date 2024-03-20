#!/bin/bash
# Compiles CUDA and executables
nvcc -gencode arch=compute_75,code=sm_75 -I/usr/include/opencv4 -L/usr/local/cuda -L/usr/lib/ *cu *.cpp -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -o houghtransform