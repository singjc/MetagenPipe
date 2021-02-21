# Medicine By Design Microbiome Project

## Overview:

Explore and attempt to improve upon classification of healthy and nonhealthy patients based on microbiome species frequencies.

## folders:

* data: where data goes
* data/raw: where raw downloaded data goes
* data/preprocessed: where any data that was produced by our own manipulation goes

* src: where all preprocessing + analysis scripts/notebooks go

## files:
* Dockerfile: makes docker image for this project
* build_image.sh: runs command to build image
* run_docker_it.sh: run an interactive session in Docker container. Note you might have to change some filepaths in this file for it to work on your machine
