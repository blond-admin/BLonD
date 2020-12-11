#!/bin/bash

echo "HOME = $HOME"
echo "BLONDHOME = $BLONDHOME"

source $HOME/.bashrc
cd $BLONDHOME

echo "PATH = $PATH"
echo "PYTHONPATH = $PYTHONPATH"

which mpirun
which python

if [ -z "$FFTWDIR" ]; then
	python blond/compile.py -p --with-fftw-omp --with-fftw-lib=$FFTWDIR/lib --with-fftw-header=$FFTWDIR/include;
else
	python blond/compile.py -p --with-fftw-omp;
fi

$@
