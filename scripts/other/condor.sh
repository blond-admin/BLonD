#!/bin/bash

echo "HOME = $HOME"
source $HOME/.bashrc
# echo "BLONDHOME = $BLONDHOME"
# cd $BLONDHOME

# export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="./:$PYTHONPATH"

# echo "PATH = $PATH"
echo "PYTHONPATH = $PYTHONPATH"

gcc --version
mpirun --version
nvcc --version
nvidia-smi

# if [ -z "$FFTWDIR" ]; then
# 	python blond/compile.py -p --with-fftw-omp --with-fftw-lib=$FFTWDIR/lib --with-fftw-header=$FFTWDIR/include;
# else
# 	python blond/compile.py -p --with-fftw-omp;
# fi

$@
