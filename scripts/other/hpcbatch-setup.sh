#!/bin/bash -l

####################################
# HPCBATCH slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################

#SBATCH --job-name=compile    # Job name
#SBATCH --output=compile.out # Stdout (%j expands to jobId)
#SBATCH --error=compile.err # Stderr (%j expands to jobId)
#SBATCH --ntasks=1     # Number of tasks(processes)
#SBATCH --nodes=1     # Number of nodes requested
#SBATCH --ntasks-per-node=1     # Tasks per node
#SBATCH --cpus-per-task=1     # Threads per task
#SBATCH --time=00:05:00   # walltime
#SBATCH --mem=56G   # memory per NODE
#SBATCH --partition=inf-short    # Partition
#SBATCH --export=ALL

if [ x$SLURM_CPUS_PER_TASK == x ]; then
  export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi


# module purge		# clean up loaded modules 
## LOAD MODULES ##
source $HOME/.bashrc


INSTALL_DIR=$HOME/install

python blond/compile.py --with-fftw --with-fftw-threads --with-fftw-lib=$INSTALL_DIR/lib/ --with-fftw-header=$INSTALL_DIR/include/ -p
