#!/bin/bash

# Exit immediately if a command fails
set -e

# Login to docker if not already
# docker login gitlab-registry.cern.ch  # must have been executed already in order to work

# Define variables
IMAGE_NAME="gitlab-registry.cern.ch/blond/blond/python310"
TAG=${1:-latest}
#############################
# Builds the container
#############################
# requires read api token !
# 1. obtained token at https://gitlab.cern.ch/-/user_settings/personal_access_tokens
# 2. export GITLAB_READ_TOKEN=<YOUR-TOKEN>
# 3. Build your docker container
echo "Building Docker image: $IMAGE_NAME:$TAG"
docker build -t "$IMAGE_NAME:$TAG" . --secret id=gitlab_token,src=<(echo $GITLAB_READ_TOKEN)
#############################
