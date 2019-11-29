import sys
import os
from mpi4py import MPI
import numpy as np
import logging
from functools import wraps
from ..utils import bmath as bm

worker = None


def print_wrap(f):
    @wraps(f)
    def wrap(*args):
        msg = '[{}] '.format(worker.rank) + ' '.join([str(a) for a in args])
        if worker.isMaster:
            worker.logger.debug(msg)
            return f('[{}]'.format(worker.rank), *args)
        else:
            return worker.logger.debug(msg)
    return wrap


mpiprint = print_wrap(print)


class Worker:
    def __init__(self, args={}):
        # args = parse()
        self.indices = {}
        self.intracomm = MPI.COMM_WORLD
        self.rank = self.intracomm.rank

        # self.intercomm = MPI.COMM_WORLD.Split(self.rank == 0, self.rank)
        # self.intercomm = self.intercomm.Create_intercomm(0, MPI.COMM_WORLD, 1)

        self.workers = self.intracomm.size

        self.hostname = MPI.Get_processor_name()
        self.log = args.get('log', False)
        # self.trace = args.get('trace', False)

        if self.log:
            self.logger = MPILog(
                rank=self.rank, log_dir=args.get('logdir', './logs'))
        else:
            self.logger = MPILog(rank=self.rank)
            self.logger.disable()

        # if self.trace:
        #     mpiprof.mode = 'tracing'
        #     mpiprof.init(logfile=args.get('tracefile', 'trace'))

    def __del__(self):
        pass
        # if self.trace:
        # mpiprof.finalize()

    @property
    def isMaster(self):
        return self.rank == 0

    # Define the begin and size numbers in order to split a variable of length size
    def split(self, size):
        self.logger.debug('split')
        counts = [size // self.workers + 1 if i < size % self.workers
                  else size // self.workers for i in range(self.workers)]
        displs = np.append([0], np.cumsum(counts[:-1])).astype(int)

        return displs[self.rank], counts[self.rank]

    # args are the buffers to fill with the gathered values
    # e.g. (comm, beam.dt, beam.dE)
    def gather(self, var, size):
        self.logger.debug('gather')
        if self.isMaster:
            counts = [size // self.workers + 1 if i < size % self.workers
                      else size // self.workers for i in range(self.workers)]
            displs = np.append([0], np.cumsum(counts[:-1]))
            sendbuf = np.copy(var)
            recvbuf = np.resize(var, np.sum(counts))

            self.intracomm.Gatherv(sendbuf,
                                   [recvbuf, counts, displs, recvbuf.dtype.char], root=0)
            return recvbuf
        else:
            recvbuf = None
            self.intracomm.Gatherv(var, recvbuf, root=0)
            return var

    def allgather(self, var, size):
        self.logger.debug('allgather')

        counts = [size // self.workers + 1 if i < size % self.workers
                  else size // self.workers for i in range(self.workers)]
        displs = np.append([0], np.cumsum(counts[:-1]))
        sendbuf = np.copy(var)
        recvbuf = np.resize(var, np.sum(counts))

        self.intracomm.Allgatherv(sendbuf,
                                  [recvbuf, counts, displs, recvbuf.dtype.char])
        return recvbuf

    def scatter(self, var, size):
        self.logger.debug('scatter')
        if self.isMaster:
            counts = [size // self.workers + 1 if i < size % self.workers
                      else size // self.workers for i in range(self.workers)]
            displs = np.append([0], np.cumsum(counts[:-1]))
            # sendbuf = np.copy(var)
            recvbuf = np.empty(counts[worker.rank], dtype=var.dtype.char)
            self.intracomm.Scatterv([var, counts, displs, var.dtype.char],
                                    recvbuf, root=0)
        else:
            counts = [size // self.workers + 1 if i < size % self.workers
                      else size // self.workers for i in range(self.workers)]
            sendbuf = None
            recvbuf = np.empty(counts[worker.rank], dtype=var.dtype.char)
            self.intracomm.Scatterv(sendbuf, recvbuf, root=0)

        return recvbuf

    def allreduce(self, sendbuf, recvbuf=None):
        self.logger.debug('allreduce')
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

        if (recvbuf is None) or (sendbuf is recvbuf):
            self.intracomm.Allreduce(MPI.IN_PLACE, sendbuf, op=op)
        else:
            self.intracomm.Allreduce(sendbuf, recvbuf, op=op)

    def sync(self):
        self.logger.debug('sync')
        self.intracomm.Barrier()

    def finalize(self):
        self.logger.debug('finalize')
        if not self.isMaster:
            sys.exit(0)

    def greet(self):
        self.logger.debug('greet')
        print('[{}]@{}: Hello World!'.format(self.rank, self.hostname))

    def print_version(self):
        self.logger.debug('version')
        # print('[{}] Library version: {}'.format(self.rank, MPI.Get_library_version()))
        # print('[{}] Version: {}'.format(self.rank,MPI.Get_version()))
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
        self.root_logger.setLevel(logging.DEBUG)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if rank < 0:
            log_name = log_dir+'/master.log'
        else:
            log_name = log_dir+'/worker-%.3d.log' % rank
        # Console handler on INFO level
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.INFO)
        log_format = logging.Formatter(
            "%(asctime)s %(name)-25s %(levelname)-9s %(message)s")
        # console_handler.setFormatter(log_format)
        # self.root_logger.addHandler(console_handler)

        self.file_handler = logging.FileHandler(log_name, mode='w')
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(log_format)
        self.root_logger.addHandler(self.file_handler)
        logging.debug("Initialized")
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


if worker is None:
    worker = Worker()
