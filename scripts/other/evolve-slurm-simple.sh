#!/bin/bash -l

####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################


which python
gcc --version
mpirun --version
nvcc --version
nvidia-smi

if [ x$SLURM_CPUS_PER_TASK == x ]; then
  export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

export CUDA_VISIBLE_DEVICES=0,1

source $HOME/.bashrc

cd $BLONDHOME

$@
