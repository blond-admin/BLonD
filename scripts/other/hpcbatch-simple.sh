#!/bin/bash

####################################
# HPCbatch slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################


# module load mpi/mvapich2/2.3

source $HOME/.bashrc

which python
gcc --version
mpirun --version

if [ x$SLURM_CPUS_PER_TASK == x ]; then
  export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi


$@
