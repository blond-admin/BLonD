'''
Functions related to running MPI simulations.

@author: Konstantinos Iliakis
@date: 01.01.2020
'''

import logging
import os
import socket
import sys
from functools import wraps

import numpy as np
from mpi4py import MPI

from ..utils import bmath as bm

WORKER = None


def mpiprint(*args, all=False):
    """Masks default print function, so that the worker id is also printed

    Args:
        all (bool, optional): _description_. Defaults to False.
    """
    if WORKER.is_master or all:
        print(f'[{WORKER.rank}]', *args)


def master_wrap(func):
    """Wrap function to be executed only by the master worker.

    Args:
        func (_type_): _description_

    Returns:
        _type_: _description_
    """
    @wraps(func)
    def wrap(*args, **kwargs):
        if WORKER.is_master:
            return func(*args, **kwargs)
        return None
    return wrap


def sequential_wrap(func, beam, split_args={}, gather_args={}):
    """Wrap a function to make it run in sequential mode.
    When in sequential mode, all the beam coordinates are gathered before executing
    the passed function, and re-splitted afterwards.

    Args:
        func (_type_): _description_
        beam (_type_): _description_
        split_args (dict, optional): _description_. Defaults to {}.
        gather_args (dict, optional): _description_. Defaults to {}.

    Returns:
        _type_: _description_
    """
    @wraps(func)
    def wrap(*args, **kw):
        beam.gather(**gather_args)
        if WORKER.is_master:
            result = func(*args, **kw)
        else:
            result = None
        beam.split(**split_args)
        return result
    return wrap


class Worker:
    """Stores information accessed by each MPI worker. Also contains all needed MPI methods.
    """
    def __init__(self):
        """Constructor
        """
        self.log = False
        self.logger = None
        # Global inter-communicator
        self.intercomm = MPI.COMM_WORLD
        self.rank = self.intercomm.rank
        self.workers = self.intercomm.size

        # Get hostname
        self.hostname = MPI.Get_processor_name()

        # Get host IP
        self.hostip = socket.gethostbyname(self.hostname)
        # Create communicator with processes on the same host
        color = np.dot(np.array(self.hostip.split('.'), int)
                       [1:], [1, 256, 256**2])
        self.nodecomm = self.intercomm.Split(color, self.rank)
        self.noderank = self.nodecomm.rank
        self.nodeworkers = self.nodecomm.size

        # Assign default values for GPUs
        self.gpu_id = -1
        self.has_gpu = False

    def assign_gpus(self, num_gpus=0):
        """Assign GPUs to workers

        Args:
            num_gpus (int, optional): _description_. Defaults to 0.
        """
        if self.noderank < num_gpus:
            self.has_gpu = True
            self.gpu_id = self.noderank

    def init_log(self, log, logdir):
        """Initialize the logs

        Args:
            log (_type_): _description_
            logdir (_type_): _description_
        """
        self.log = log
        self.logger = MPILog(rank=self.rank, log_dir=logdir)
        if not self.log:
            self.logger.disable()

    @property
    def is_master(self):
        """Return true if this worker is the master

        Returns:
            _type_: _description_
        """
        return self.rank == 0

    # Define the begin and size numbers in order to split a variable of length size
    def gather(self, var):
        """Gather a vector on the master

        Args:
            var (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.log:
            self.logger.debug('gather')

        # First I need to know the total size
        counts = np.zeros(self.workers, dtype=int)
        sendbuf = np.array([len(var)], dtype=int)
        self.intercomm.Gather(sendbuf, counts, root=0)
        total_size = np.sum(counts)

        if self.is_master:
            # counts = [size // self.workers + 1 if i < size % self.workers
            #           else size // self.workers for i in range(self.workers)]
            displs = np.append([0], np.cumsum(counts[:-1]))
            sendbuf = np.copy(var)
            recvbuf = np.resize(var, total_size)

            self.intercomm.Gatherv(sendbuf,
                                   [recvbuf, counts, displs, recvbuf.dtype.char], root=0)
            return recvbuf
        else:
            recvbuf = None
            self.intercomm.Gatherv(var, recvbuf, root=0)
            return var

    # All workers gather the variable var (from all workers)

    def allgather(self, var):
        """Gather vector to all workers.

        Args:
            var (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.log:
            self.logger.debug('allgather')

        # One first gather to collect all the sizes
        counts = np.zeros(self.workers, dtype=int)
        sendbuf = np.array([len(var)], dtype=int)
        self.intercomm.Allgather(sendbuf, counts)

        total_size = np.sum(counts)
        # counts = [size // self.workers + 1 if i < size % self.workers
        #           else size // self.workers for i in range(self.workers)]
        displs = np.append([0], np.cumsum(counts[:-1]))
        sendbuf = np.copy(var)
        recvbuf = np.resize(var, total_size)

        self.intercomm.Allgatherv(sendbuf,
                                  [recvbuf, counts, displs, recvbuf.dtype.char])
        return recvbuf

    def scatter(self, var):
        """Scatter vector from master to all workers.

        Args:
            var (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.log:
            self.logger.debug('scatter')

        # First broadcast the total_size from the master
        total_size = int(self.intercomm.bcast(len(var), root=0))

        # Then calculate the counts (size for each worker)
        counts = [total_size // self.workers + 1 if i < total_size % self.workers
                  else total_size // self.workers for i in range(self.workers)]

        if self.is_master:
            displs = np.append([0], np.cumsum(counts[:-1]))
            recvbuf = np.empty(counts[WORKER.rank], dtype=var.dtype.char)
            self.intercomm.Scatterv([var, counts, displs, var.dtype.char],
                                    recvbuf, root=0)
        else:
            sendbuf = None
            recvbuf = np.empty(counts[WORKER.rank], dtype=var.dtype.char)
            self.intercomm.Scatterv(sendbuf, recvbuf, root=0)

        return recvbuf

    def broadcast(self, var, root=0):
        """Broadcast array to all workers.

        Args:
            var (_type_): _description_
            root (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        if self.log:
            self.logger.debug('broadcast')

        # First broadcast the size and dtype from the master
        # recvbuf = self.intercomm.bcast([len(var), var.dtype.char], root=0)
        # size, dtype = recvbuf[0], recvbuf[1]

        if self.is_master:
            recvbuf = self.intercomm.bcast(var, root=root)
        else:
            recvbuf = None
            recvbuf = self.intercomm.bcast(recvbuf, root=root)

        return recvbuf

    def reduce(self, sendbuf, recvbuf=None, dtype=np.uint32, operator='custom_sum',
               comm=None):
        """Reduce array to master.

        Args:
            sendbuf (_type_): _description_
            recvbuf (_type_, optional): _description_. Defaults to None.
            dtype (_type_, optional): _description_. Defaults to np.uint32.
            operator (str, optional): _description_. Defaults to 'custom_sum'.
            comm (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if comm is None:
            comm = self.intercomm
        # supported ops:
        # sum, mean, std, max, min, prod, custom_sum
        if self.log:
            self.logger.debug('reduce')
        operator = operator.lower()
        if operator in ['sum', 'custom_sum']:
            mpi_op = MPI.SUM
        elif operator == 'max':
            mpi_op = MPI.MAX
        elif operator == 'min':
            mpi_op = MPI.MIN
        elif operator == 'prod':
            mpi_op = MPI.PROD
        elif operator in ['mean', 'avg']:
            mpi_op = MPI.SUM
        elif operator == 'std':
            recvbuf = self.gather(sendbuf)
            if WORKER.is_master:
                assert len(recvbuf) == 3 * self.workers
                totals = np.sum((recvbuf[2::3] - 1) * recvbuf[1::3]**2 +
                                recvbuf[2::3] * (recvbuf[1::3] - bm.mean(recvbuf[0::3]))**2)
                return np.array([np.sqrt(totals / (np.sum(recvbuf[2::3]) - 1))])
            else:
                return np.array([sendbuf[1]])

        if WORKER.is_master:
            if (recvbuf is None) or (sendbuf is recvbuf):
                comm.Reduce(MPI.IN_PLACE, sendbuf, op=mpi_op, root=0)
                recvbuf = sendbuf
            else:
                comm.Reduce(sendbuf, recvbuf, op=mpi_op, root=0)

            if operator in ['mean', 'avg']:
                return recvbuf / self.workers
            return recvbuf
        else:
            recvbuf = None
            comm.Reduce(sendbuf, recvbuf, op=mpi_op, root=0)
            return sendbuf

    def allreduce(self, sendbuf, recvbuf=None, dtype=np.uint32, operator='sum',
                  comm=None):
        """Reduce array to all workers.

        Args:
            sendbuf (_type_): _description_
            recvbuf (_type_, optional): _description_. Defaults to None.
            dtype (_type_, optional): _description_. Defaults to np.uint32.
            operator (str, optional): _description_. Defaults to 'sum'.
            comm (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if comm is None:
            comm = self.intercomm

        # supported ops:
        # sum, mean, std, max, min, prod, custom_sum
        if self.log:
            self.logger.debug('allreduce')
        operator = operator.lower()

        if operator in ['sum', 'custom_sum']:
            mpi_op = MPI.SUM
        elif operator == 'max':
            mpi_op = MPI.MAX
        elif operator == 'min':
            mpi_op = MPI.MIN
        elif operator == 'prod':
            mpi_op = MPI.PROD
        elif operator in ['mean', 'avg']:
            mpi_op = MPI.SUM
        elif operator == 'std':
            recvbuf = self.allgather(sendbuf)
            assert len(recvbuf) == 3 * self.workers
            totals = np.sum((recvbuf[2::3] - 1) * recvbuf[1::3]**2 +
                            recvbuf[2::3] * (recvbuf[1::3] - bm.mean(recvbuf[::3]))**2)
            return np.array([np.sqrt(totals / (np.sum(recvbuf[2::3]) - 1))])

        if (recvbuf is None) or (sendbuf is recvbuf):
            comm.Allreduce(MPI.IN_PLACE, sendbuf, op=mpi_op)
            recvbuf = sendbuf
        else:
            comm.Allreduce(sendbuf, recvbuf, op=mpi_op)

        if operator in ['mean', 'avg']:
            return recvbuf / self.workers
        return recvbuf

    def sync(self):
        """Synchronize all workers.
        """
        if self.log:
            self.logger.debug('sync')
        self.intercomm.Barrier()

    def finalize(self):
        """Leave MPI.
        """
        if self.log:
            self.logger.debug('finalize')
        if not self.is_master:
            sys.exit(0)

    def greet(self):
        """Print greeting message
        """
        if self.log:
            self.logger.debug('greet')
        print(f'[{self.rank}]@{self.hostname}: Hello World!')

    def print_version(self):
        """Print MPI version.
        """
        if self.log:
            self.logger.debug('version')
        print(f'[{self.rank}] Library: {MPI.get_vendor()}')


class MPILog:
    """Class to log messages coming from other classes. Messages contain
    {Time stamp} {Class name} {Log level} {Message}. Errors, warnings and info
    are logged into the console. To disable logging, call Logger().disable()
    Parameters
    ----------
    debug : bool
        Log DEBUG messages in 'debug.log'; default is False
    """

    def __init__(self, rank=0, log_dir='./logs'):

        # Root logger on DEBUG level
        self.disabled = False
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.WARNING)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        log_name = log_dir + f'/worker-{rank:03d}.log'
        # Console handler on INFO level
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.INFO)
        log_format = logging.Formatter(
            "%(asctime)s %(name)-25s %(levelname)-9s %(message)s")
        # console_handler.setFormatter(log_format)
        # self.root_logger.addHandler(console_handler)

        self.file_handler = logging.FileHandler(log_name, mode='w')
        self.file_handler.setLevel(logging.WARNING)
        self.file_handler.setFormatter(log_format)
        self.root_logger.addHandler(self.file_handler)
        logging.info("Initialized")
        # if debug == True:
        #     logging.debug("Logger in debug mode")

    def disable(self):
        """Disables all logging."""

        logging.info("Disable logging")
        # logging.disable(level=logging.NOTSET)
        # self.root_logger.setLevel(logging.NOTSET)
        # self.file_handler.setLevel(logging.NOTSET)
        self.root_logger.disabled = True
        self.disabled = True

    def debug(self, string):
        """Set logging to debug verbosity

        Args:
            string (_type_): _description_
        """
        if not self.disabled:
            logging.debug(string)

    def info(self, string):
        """Set logging to info verbosity

        Args:
            string (_type_): _description_
        """
        if not self.disabled:
            logging.info(string)

    def critical(self, string):
        """Set logging to critical verbosity

        Args:
            string (_type_): _description_
        """
        if not self.disabled:
            logging.critical(string)


if WORKER is None:
    WORKER = Worker()
