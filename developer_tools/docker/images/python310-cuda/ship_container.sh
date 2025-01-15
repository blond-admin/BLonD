#!/bin/bash

# Exit immediately if a command fails
set -e

# Login to docker if not already
# docker login gitlab-registry.cern.ch  # must have been executed already in order to work

# Define variables
IMAGE_NAME="gitlab-registry.cern.ch/blond/blond/python310"
TAG=${1:-latest}

# docker login gitlab-registry.cern.ch  # must have been executed already in order to work
# Builds the container
echo "Building Docker image: $IMAGE_NAME:$TAG"
docker build -t "$IMAGE_NAME:$TAG" .

# Push the container to gitlab
echo "Pushing Docker image to: $IMAGE_NAME:$TAG"
docker push "$IMAGE_NAME:$TAG"

echo "Docker image $IMAGE_NAME:$TAG successfully pushed."