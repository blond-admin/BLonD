FROM python:3.10

# Create a non-root user
RUN useradd -m non_root

# Switch to root to install packages
USER root

# Set the environment variable to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

#######################################################################################
# Obtaining CERN rf-noise-cpp
#######################################################################################
# requires read api token !
# 1. obtained token at https://gitlab.cern.ch/-/user_settings/personal_access_tokens
# 2. export GITLAB_READ_TOKEN=<YOUR-TOKEN>
# 3. Build your docker container
RUN --mount=type=secret,id=gitlab_token \
    GITLAB_READ_TOKEN=$(cat /run/secrets/gitlab_token) && \
    git clone https://oauth2:$GITLAB_READ_TOKEN@gitlab.cern.ch/be-rf-cs/Tools-and-libs/rf-noise-cpp.git/ /usr/local/rf-noise-cpp
RUN apt-get update && \
    apt-get install -y libboost-all-dev
ENV RF_NOISE_DIR=/usr/local/rf-noise-cpp
#######################################################################################

RUN apt-get update && \
    apt-get install -y gcc g++  build-essential mpich libmpich-dev
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit && \
    rm -rf /var/lib/apt/lists/*
# Make sure NVCC is accessible
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH


RUN bash -c 'python3 -m pip install --upgrade pip&& \
       python3 -m pip install cupy-cuda12x  pytest_cov mpi4py'

# Switch to the non-root user
USER non_root

# Set the working directory (optional, adjust as needed)
WORKDIR /home/non_root

# Set the entrypoint
ENTRYPOINT ["/bin/bash"]