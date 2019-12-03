import sys
import os
import numpy as np
import logging
from functools import wraps
from ..utils import bmath as bm
from mpi4py import MPI

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

    def __init__(self, args={}):
        # args = parse()
        # self.indices = {}
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


    def gather(self, var):
        self.logger.debug('gather')

        # First I need to know the total size
        counts = np.zeros(self.workers, dtype=int)
        sendbuf = np.array([len(var)], dtype=int)
        self.intracomm.Gather(sendbuf, counts, root=0)
        total_size = np.sum(counts)

        if self.isMaster:
            # counts = [size // self.workers + 1 if i < size % self.workers
            #           else size // self.workers for i in range(self.workers)]
            displs = np.append([0], np.cumsum(counts[:-1]))
            sendbuf = np.copy(var)
            recvbuf = np.resize(var, total_size)

            self.intracomm.Gatherv(sendbuf,
                                   [recvbuf, counts, displs, recvbuf.dtype.char], root=0)
            return recvbuf
        else:
            recvbuf = None
            self.intracomm.Gatherv(var, recvbuf, root=0)
            return var

    # All workers gather the variable var (from all workers)
    def allgather(self, var):
        self.logger.debug('allgather')

        # One first gather to collect all the sizes
        counts = np.zeros(self.workers, dtype=int)
        sendbuf = np.array([len(var)], dtype=int)
        self.intracomm.Allgather(sendbuf, counts)

        total_size = np.sum(counts)
        # counts = [size // self.workers + 1 if i < size % self.workers
        #           else size // self.workers for i in range(self.workers)]
        displs = np.append([0], np.cumsum(counts[:-1]))
        sendbuf = np.copy(var)
        recvbuf = np.resize(var, total_size)

        self.intracomm.Allgatherv(sendbuf,
                                  [recvbuf, counts, displs, recvbuf.dtype.char])
        return recvbuf

    def scatter(self, var):
        self.logger.debug('scatter')

        # First broadcast the total_size from the master
        total_size = int(self.intracomm.bcast(len(var), root=0))

        # Then calculate the counts (size for each worker)
        counts = [total_size // self.workers + 1 if i < total_size % self.workers
                  else total_size // self.workers for i in range(self.workers)]
        
        if self.isMaster:
            displs = np.append([0], np.cumsum(counts[:-1]))
            recvbuf = np.empty(counts[worker.rank], dtype=var.dtype.char)
            self.intracomm.Scatterv([var, counts, displs, var.dtype.char],
                                    recvbuf, root=0)
        else:
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

    # class sequential_context:
    #     class SkipWithBlock(Exception):
    #         pass

    #     def __init__(self, beam, split_args={}, gather_args={}):
    #         self.beam = beam
    #         self.split_args = split_args
    #         self.gather_args = gather_args

    #     def __enter__(self):
    #         self.beam.split(**self.split_args)
    #         if not worker.isMaster:
    #             # Do some magic
    #             sys.settrace(lambda *args, **keys: None)
    #             frame = sys._getframe(1)
    #             frame.f_trace = self.trace

    #     def trace(self, fname, event, arg):
    #         raise self.SkipWithBlock()

    #     def __exit__(self, type, value, traceback):
    #         self.beam.gather(**self.gather_args)
    #         if type is None:
    #             return
    #         if issubclass(type, self.SkipWithBlock):
    #             return True


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
