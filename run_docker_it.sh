#!/bin/bash
# interactively run docker. Note that we have the v option for binding our project directory to a /project directory
# un docker image. Downloaded or modified data within docker image that resides within /project directory will persist
# on host OS location, and will be visible on host OS
docker run -v /home/owenwhitley/projects/microbiome:/project -p 8888:8888 -it microbiome_ubuntu20 
