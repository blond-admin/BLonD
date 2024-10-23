FROM python:3.10
CMD ["bash"]

# Set the environment variable to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

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


