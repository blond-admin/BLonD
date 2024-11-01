#!/bin/bash

# Exit immediately if a command fails
set -e

# Login to docker if not already
# docker login gitlab-registry.cern.ch  # must have been executed already in order to work

# Define variables
IMAGE_NAME="gitlab-registry.cern.ch/blond/blond/python310"
TAG=${1:-latest}

echo "Starting container locally: $IMAGE_NAME:$TAG"
docker run -it "$IMAGE_NAME:$TAG" # opens shell to interactively work with the container
