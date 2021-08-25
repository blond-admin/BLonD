#!/bin/bash

####################################
# HPCbatch slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################


# module load mpi/mvapich2/2.3

# source $HOME/.bashrc

module purge
module load mpi/mvapich2/2.3
module load compiler/gcc7

export PYTHONPATH="$HOME/install/:$HOME/git/pymodules/:$HOME/scripts/:$PYTHONPATH"
export PATH="$HOME/install/miniconda3-mvapich2.3/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/install/lib:$LD_LIBRARY_PATH"
export MV2_HOMOGENEOUS_CLUSTER=1
export MV2_ENABLE_AFFINITY="0"


which python
gcc --version
mpirun --version

if [ x$SLURM_CPUS_PER_TASK == x ]; then
  export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi


$@
