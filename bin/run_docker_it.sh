#!/bin/bash
#############
## CL Args
#############
working_dir=${1:-`pwd`}
# interactively run docker. Note that we have the v option for binding our project directory to a /project directory
# un docker image. Downloaded or modified data within docker image that resides within /project directory will persist
# on host OS location, and will be visible on host OS
#docker run -v $working_dir:/project -p 8888:8888 -p 2200:22 -it whitleyo/microbiome_ubuntu20:latest
docker run -v $working_dir:/project -p 8888:8888 -p 2200:22 -it microbiome_ubuntu20:latest
