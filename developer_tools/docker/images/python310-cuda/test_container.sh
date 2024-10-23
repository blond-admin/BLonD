#!/bin/bash

# Exit immediately if a command fails
set -e

docker build -t python311-cuda-image . # builds docker-image
docker run --rm -it python311-cuda-image # opens shell to interactively work with the container
