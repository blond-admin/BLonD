#!/bin/bash

# Exit immediately if a command fails
set -e

docker build -t python310-cuda-image . # builds docker-image
docker run --rm -it python310-cuda-image # opens shell to interactively work with the container
