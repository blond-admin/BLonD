# docker login gitlab-registry.cern.ch  # must have been executed already in order to work
docker build -t gitlab-registry.cern.ch/blond/blond/python310  .  # builds the container
docker push gitlab-registry.cern.ch/blond/blond/python310  # uploads the container to gitlab