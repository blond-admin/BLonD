#!/bin/bash

# Exit immediately if a command fails
set -e

# builds docker-image
# requires read api token !
# 1. obtained token at https://gitlab.cern.ch/-/user_settings/personal_access_tokens
# 2. export GITLAB_READ_TOKEN=<YOUR-TOKEN>
# 3. Build your docker container
docker build -t python310-cuda-image . --secret id=gitlab_token,src=<(echo $GITLAB_READ_TOKEN)
docker run --rm -it python310-cuda-image # opens shell to interactively work with the container
