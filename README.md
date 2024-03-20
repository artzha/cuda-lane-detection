# CUDA Lane Detection
CUDA Implementation of a Hough Transform based Lane Detection algorithm.

<img src="https://i.ibb.co/5RM4n5F/Picture1.png" alt="Picture1" border="0">

## Building Docker Container
Run the following commands to build and run the docker container. This will automatically install the openCV CUDA, and ROS Noetic dependencies.

```Builds the docker container
bash bash/build_docker.sh
```

```Runs the docker container
bash bash/run_docker.sh
```

# In the Docker Container

You will need to compile and run the program within the docker container using the following command.

## Compiling
The program can be compiled on the Linux lab machines using the following command:

```
bash bash/compile_cuda.sh
```

## Running

The Lane Detection program requires two positional arguments. The `inputVideo` which is a path to the input video and the `outputVideo` which is the path at which the result video is stored.

Additionally, we can add either a `--cuda` flag to use the CUDA implementation or a `--seq` flag to use the sequential implementation. 

Therefore, in order to run it for the test video provided in the repository we can use the following command.

```
./LaneDetection.out ./test-video.mp4 ./result-video.avi --cuda
```
