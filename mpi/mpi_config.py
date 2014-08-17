'''
**Module to set up MPI parallelisation used in the longitudinal tracker.**

:Authors: **Helga Timko**
'''

from mpi4py import MPI
import warnings


class MPI_Config(object):
    '''
    *Configuring MPI*
    '''
        
    def __init__(self):
        
        try:   
            self.mpi_comm = MPI.COMM_WORLD
            self.mpi_size = self.mpi_comm.Get_size()
            self.mpi_rank = self.mpi_comm.Get_rank()
            if self.mpi_size == 1:
                self.mpi_comm = None
                warnings.warn('Running on single CPU in serial mode!')                
        except:
            self.mpi_comm = None
            warnings.warn('Running on single CPU in serial mode!')

