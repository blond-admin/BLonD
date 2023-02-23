import sys
import os
from mpi4py import MPI
import numpy as np
import logging
from functools import wraps
import socket


from ..utils import bmath as bm

worker = None


def mpiprint(*args, all=False):
    if worker.isMaster or all:
        print('[{}]'.format(worker.rank), *args)


def master_wrap(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if worker.isMaster:
            return f(*args, **kwargs)
        else:
            return None
    return wrap


def sequential_wrap(f, beam, split_args={}, gather_args={}):
    @wraps(f)
    def wrap(*args, **kw):
        beam.gather(**gather_args)
        if worker.isMaster:
            result = f(*args, **kw)
        else:
            result = None
        beam.split(**split_args)
        return result
    return wrap


class Worker:
    def __init__(self):
        self.log = False

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
        self.hasGPU = False

    def assignGPUs(self, num_gpus=0):
        if self.noderank < num_gpus:
            self.hasGPU = True
            self.gpu_id = self.noderank

    def initLog(self, log, logdir):
        self.log = log
        self.logger = MPILog(rank=self.rank, log_dir=logdir)
        if not self.log:
            self.logger.disable()


    def __del__(self):
        pass

    @property
    def isMaster(self):
        return self.rank == 0

    # Define the begin and size numbers in order to split a variable of length size
    def gather(self, var):
        if self.log:
            self.logger.debug('gather')

        # First I need to know the total size
        counts = np.zeros(self.workers, dtype=int)
        sendbuf = np.array([len(var)], dtype=int)
        self.intercomm.Gather(sendbuf, counts, root=0)
        total_size = np.sum(counts)

        if self.isMaster:
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
        if self.log:
            self.logger.debug('scatter')

        # First broadcast the total_size from the master
        total_size = int(self.intercomm.bcast(len(var), root=0))

        # Then calculate the counts (size for each worker)
        counts = [total_size // self.workers + 1 if i < total_size % self.workers
                  else total_size // self.workers for i in range(self.workers)]
        
        if self.isMaster:
            displs = np.append([0], np.cumsum(counts[:-1]))
            recvbuf = np.empty(counts[worker.rank], dtype=var.dtype.char)
            self.intercomm.Scatterv([var, counts, displs, var.dtype.char],
                                    recvbuf, root=0)
        else:
            sendbuf = None
            recvbuf = np.empty(counts[worker.rank], dtype=var.dtype.char)
            self.intercomm.Scatterv(sendbuf, recvbuf, root=0)

        return recvbuf

    def broadcast(self, var, root=0):
        if self.log:
            self.logger.debug('broadcast')

        # First broadcast the size and dtype from the master
        # recvbuf = self.intercomm.bcast([len(var), var.dtype.char], root=0)
        # size, dtype = recvbuf[0], recvbuf[1]

        if self.isMaster:   
            recvbuf = self.intercomm.bcast(var, root=root)
        else:
            recvbuf = None
            recvbuf = self.intercomm.bcast(recvbuf, root=root)

        return recvbuf


    def reduce(self, sendbuf, recvbuf=None, dtype=np.uint32, operator='custom_sum',
               comm=None):
        if comm is None:
            comm = self.intercomm
        # supported ops:
        # sum, mean, std, max, min, prod, custom_sum
        if self.log:
            self.logger.debug('reduce')
        operator = operator.lower()
        if operator == 'custom_sum':
            dtype = sendbuf.dtype.name
            if dtype == 'int16':
                op = add_op_int16
            elif dtype == 'int32':
                op = add_op_int32
            elif dtype == 'int64':
                op = add_op_int64
            elif dtype == 'uint16':
                op = add_op_uint16
            elif dtype == 'uint32':
                op = add_op_uint32
            elif dtype == 'uint64':
                op = add_op_uint64
            elif dtype == 'float32':
                op = add_op_float32
            elif dtype == 'float64':
                op = add_op_float64
            else:
                print('Error: Not recognized dtype:{}'.format(dtype))
                exit(-1)
        elif operator == 'sum':
            op = MPI.SUM
        elif operator == 'max':
            op = MPI.MAX
        elif operator == 'min':
            op = MPI.MIN
        elif operator == 'prod':
            op = MPI.PROD
        elif operator in ['mean', 'avg']:
            op = MPI.SUM
        elif operator == 'std':
            recvbuf = self.gather(sendbuf)
            if worker.isMaster:
                assert len(recvbuf) == 3 * self.workers
                totals = np.sum((recvbuf[2::3] - 1) * recvbuf[1::3]**2 +
                                recvbuf[2::3] * (recvbuf[1::3] - bm.mean(recvbuf[0::3]))**2)
                return np.array([np.sqrt(totals / (np.sum(recvbuf[2::3]) - 1))])
            else:
                return np.array([sendbuf[1]])

        if worker.isMaster:
            if (recvbuf is None) or (sendbuf is recvbuf):
                comm.Reduce(MPI.IN_PLACE, sendbuf, op=op, root=0)
                recvbuf = sendbuf
            else:
                comm.Reduce(sendbuf, recvbuf, op=op, root=0)

            if operator in ['mean', 'avg']:
                return recvbuf / self.workers
            else:
                return recvbuf
        else:
            recvbuf = None
            comm.Reduce(sendbuf, recvbuf, op=op, root=0)
            return sendbuf


    def allreduce(self, sendbuf, recvbuf=None, dtype=np.uint32, operator='sum',
                  comm=None):
        if comm is None:
            comm = self.intercomm

        # supported ops:
        # sum, mean, std, max, min, prod, custom_sum
        if self.log:
            self.logger.debug('allreduce')
        operator = operator.lower()
        if operator == 'custom_sum':
            dtype = sendbuf.dtype.name
            if dtype == 'int16':
                op = add_op_int16
            elif dtype == 'int32':
                op = add_op_int32
            elif dtype == 'int64':
                op = add_op_int64
            elif dtype == 'uint16':
                op = add_op_uint16
            elif dtype == 'uint32':
                op = add_op_uint32
            elif dtype == 'uint64':
                op = add_op_uint64
            elif dtype == 'float32':
                op = add_op_float32
            elif dtype == 'float64':
                op = add_op_float64
            else:
                print('Error: Not recognized dtype:{}'.format(dtype))
                exit(-1)
        elif operator == 'sum':
            op = MPI.SUM
        elif operator == 'max':
            op = MPI.MAX
        elif operator == 'min':
            op = MPI.MIN
        elif operator == 'prod':
            op = MPI.PROD
        elif operator in ['mean', 'avg']:
            op = MPI.SUM
        elif operator == 'std':
            recvbuf = self.allgather(sendbuf)
            assert len(recvbuf) == 3 * self.workers
            totals = np.sum((recvbuf[2::3] - 1) * recvbuf[1::3]**2 +
                            recvbuf[2::3] * (recvbuf[1::3] - bm.mean(recvbuf[::3]))**2)
            return np.array([np.sqrt(totals / (np.sum(recvbuf[2::3]) - 1))])

        if (recvbuf is None) or (sendbuf is recvbuf):
            comm.Allreduce(MPI.IN_PLACE, sendbuf, op=op)
            recvbuf = sendbuf
        else:
            comm.Allreduce(sendbuf, recvbuf, op=op)

        if operator in ['mean', 'avg']:

            return recvbuf / self.workers
        else:
            return recvbuf

    def sync(self):
        if self.log:
            self.logger.debug('sync')
        self.intercomm.Barrier()


    def finalize(self):
        if self.log:
            self.logger.debug('finalize')
        if not self.isMaster:
            sys.exit(0)

    def greet(self):
        if self.log:
            self.logger.debug('greet')
        print('[{}]@{}: Hello World!'.format(self.rank, self.hostname))

    def print_version(self):
        if self.log:
            self.logger.debug('version')
        print('[{}] Library: {}'.format(self.rank, MPI.get_vendor()))



class MPILog(object):
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

        log_name = log_dir+'/worker-%.3d.log' % rank
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
        if self.disabled == False:
            logging.debug(string)

    def info(self, string):
        if self.disabled == False:
            logging.info(string)

    def critical(self, string):
        if self.disabled == False:
            logging.critical(string)


if worker is None:
    worker = Worker()


def c_add_float32(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.float32)
    y = np.frombuffer(ymem, dtype=np.float32)
    bm.add(y, x, inplace=True)


add_op_float32 = MPI.Op.Create(c_add_float32, commute=True)


def c_add_float64(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.float64)
    y = np.frombuffer(ymem, dtype=np.float64)
    bm.add(y, x, inplace=True)


add_op_float64 = MPI.Op.Create(c_add_float64, commute=True)


def c_add_uint16(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.uint16)
    y = np.frombuffer(ymem, dtype=np.uint16)
    bm.add(y, x, inplace=True)


add_op_uint16 = MPI.Op.Create(c_add_uint16, commute=True)


def c_add_uint32(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.uint32)
    y = np.frombuffer(ymem, dtype=np.uint32)
    bm.add(y, x, inplace=True)


add_op_uint32 = MPI.Op.Create(c_add_uint32, commute=True)


def c_add_uint64(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.uint64)
    y = np.frombuffer(ymem, dtype=np.uint64)
    bm.add(y, x, inplace=True)


add_op_uint64 = MPI.Op.Create(c_add_uint64, commute=True)


def c_add_int16(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.int16)
    y = np.frombuffer(ymem, dtype=np.int16)
    bm.add(y, x, inplace=True)


add_op_int16 = MPI.Op.Create(c_add_int16, commute=True)


def c_add_int32(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.int32)
    y = np.frombuffer(ymem, dtype=np.int32)
    bm.add(y, x, inplace=True)


add_op_int32 = MPI.Op.Create(c_add_int32, commute=True)


def c_add_int64(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.int64)
    y = np.frombuffer(ymem, dtype=np.int64)
    bm.add(y, x, inplace=True)


add_op_int64 = MPI.Op.Create(c_add_int64, commute=True)
