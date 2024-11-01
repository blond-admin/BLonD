#!/bin/bash

# Exit immediately if a command fails
set -e

# Define variables
IMAGE_NAME="gitlab-registry.cern.ch/blond/blond/python310"
TAG=${1:-latest}


# Push the container to gitlab
echo "Pushing Docker image to: $IMAGE_NAME:$TAG"
docker push "$IMAGE_NAME:$TAG"