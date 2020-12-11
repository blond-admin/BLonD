#!/bin/bash -l

####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################

#SBATCH --partition=gpu    # Partition
#SBATCH --account=pa200702    # Replace with your system project
#SBATCH --gres=gpu:2		# For srun, allow access to 2 GPUs
#SBATCH --exclude=gpu21,gpu22,gpu23,gpu24,gpu25,gpu26,gpu34


which python
gcc --version
mpirun --version
nvcc --version
nvidia-smi
export PYTHONPATH="/users/pa20/kiliakis/panos/BLonD_staff/BLonD:/users/pa20/kiliakis/install/pymodules"

if [ x$SLURM_CPUS_PER_TASK == x ]; then
  export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

source $HOME/.bashrc

cd $HOME/panos/BLonD_staff/BLonD

$@
