import sys
import os
from mpi4py import MPI
import numpy as np
import logging
from functools import wraps
import socket

try:
    from pyprof import timing
    # from pyprof import mpiprof
except ImportError:
    from ..utils import profile_mock as timing
    mpiprof = timing

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
    @timing.timeit(key='serial:init')
    # @mpiprof.traceit(key='serial:init')
    def __init__(self):
        self.start_turn = 100
        self.start_interval = 500
        self.indices = {}
        self.interval = 500
        self.coefficients = {'particles': [0], 'times': [0.],
                             'intra_particles': [0], 'intra_times': [0],
                             'intra_load': None}
        self.taskparallelism = False

        # Global inter-communicator
        self.intercomm = MPI.COMM_WORLD
        self.rank = self.intercomm.rank
        self.workers = self.intercomm.size

        # Setup TP intracomm
        self.hostname = MPI.Get_processor_name()
        self.hostip = socket.gethostbyname(self.hostname)

        # Create communicator with processes on the same host
        color = np.dot(np.array(self.hostip.split('.'), int)
                       [1:], [1, 256, 256**2])
        self.nodecomm = self.intercomm.Split(color, self.rank)
        self.noderank = self.nodecomm.rank
        self.nodeworkers = self.nodecomm.size

        # Break the hostcomm in neighboring pairs
        self.intracomm = self.nodecomm.Split(self.noderank//2, self.noderank)
        self.intraworkers = self.intracomm.size
        self.intrarank = self.intracomm.rank
        # tempcomm.Free()
        self.log = False
        self.trace = False

        # Assign default values
        self.gpucomm = None
        self.gpucommworkers = 0
        self.gpucommrank = 0
        self.gpu_id = -1
        self.hasGPU = False

    def assignGPUs(self, num_gpus=0):
        # Here goes the gpu assignment
        if num_gpus > 0:
            # Divide all workers into almost equal sized groups
            split_groups = np.array_split(np.arange(self.nodeworkers), num_gpus)

            # Find in which group this worker belongs
            mygroup = 0
            for i, a in enumerate(split_groups):
                if self.noderank in a:
                    mygroup = i
                    break

            # Save the group, it will be used to get access to the specific gpu
            self.gpu_id = mygroup

            # Create a communicator per group
            self.gpucomm = self.nodecomm.Split(mygroup, self.noderank)
            self.gpucommworkers = self.gpucomm.size
            self.gpucommrank = self.gpucomm.rank

            # If you are the first in the group, you get access to the gpu
            if self.gpucommrank == 0:
                self.hasGPU = True
            else:
                self.hasGPU = False

    def initLog(self, log, logdir):
        self.log = log
        self.logger = MPILog(rank=self.rank, log_dir=logdir)
        if not self.log:
            self.logger.disable()

    def initTrace(self, trace, tracefile):
        self.trace = trace
        if self.trace:
            mpiprof.mode = 'tracing'
            mpiprof.init(logfile=tracefile)

    def __del__(self):
        pass
        # if self.trace:
        # mpiprof.finalize()

    @property
    def isMaster(self):
        return self.rank == 0

    @property
    def isFirst(self):
        return (self.intrarank == 0) or (self.taskparallelism is False)

    @property
    def isLast(self):
        return (self.intrarank == self.intraworkers-1) or (self.taskparallelism is False)

    # Define the begin and size numbers in order to split a variable of length size

    @timing.timeit(key='comm:gather')
    # @mpiprof.traceit(key='comm:gather')
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

    @timing.timeit(key='comm:allgather')
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

    @timing.timeit(key='comm:scatter')
    # @mpiprof.traceit(key='comm:scatter')
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

    @timing.timeit(key='comm:broadcast')
    # @mpiprof.traceit(key='comm:scatter')
    def broadcast(self, var, root=0):
        if self.log:
            self.logger.debug('broadcast')

        if self.gpucommrank == root:
            recvbuf = self.gpucomm.bcast(var, root=root)
        else:
            recvbuf = None
            recvbuf = self.gpucomm.bcast(recvbuf, root=root)

        return recvbuf

    # @timing.timeit(key='comm:broadcast_reverse')
    # # @mpiprof.traceit(key='comm:scatter')
    # def broadcast_reverse(self, var):

    #     if self.log:
    #         self.logger.debug('broadcast_reverse')
    #     if self.gpucommrank == 0:
    #         recvbuf = None
    #         recvbuf = self.gpucomm.bcast(recvbuf, root=1)
    #     else:
    #         recvbuf = self.gpucomm.bcast(var, root=1)

    #     return recvbuf

    @timing.timeit(key='comm:reduce')
    # @mpiprof.traceit(key='comm:reduce')
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

    @timing.timeit(key='comm:allreduce')
    # @mpiprof.traceit(key='comm:allreduce')
    def allreduce(self, sendbuf, recvbuf=None, dtype=np.uint32, operator='custom_sum',
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

    @timing.timeit(key='serial:sync')
    # @mpiprof.traceit(key='serial:sync')
    def sync(self):
        if self.log:
            self.logger.debug('sync')
        self.intercomm.Barrier()

    @timing.timeit(key='serial:intraSync')
    # @mpiprof.traceit(key='serial:intraSync')
    def intraSync(self):
        if self.log:
            self.logger.debug('intraSync')
        self.intracomm.Barrier()

    @timing.timeit(key='serial:gpuSync')
    # @mpiprof.traceit(key='serial:gpuSync')
    def gpuSync(self):
        if self.log:
            self.logger.debug('gpuSync')
        self.gpucomm.Barrier()

    @timing.timeit(key='serial:finalize')
    # @mpiprof.traceit(key='serial:finalize')
    def finalize(self):
        if self.log:
            self.logger.debug('finalize')
        if not self.isMaster:
            sys.exit(0)

    @timing.timeit(key='comm:sendrecv')
    # @mpiprof.traceit(key='comm:sendrecv')
    def sendrecv(self, sendbuf, recvbuf):
        if self.log:
            self.logger.debug('sendrecv')
        if self.isFirst and not self.isLast:
            self.intracomm.Sendrecv(sendbuf, dest=self.intraworkers-1, sendtag=0,
                                    recvbuf=recvbuf, source=self.intraworkers-1,
                                    recvtag=1)
        elif self.isLast and not self.isFirst:
            self.intracomm.Sendrecv(recvbuf, dest=0, sendtag=1,
                                    recvbuf=sendbuf, source=0, recvtag=0)

    @timing.timeit(key='comm:redistribute')
    # @mpiprof.traceit(key='comm:redistribute')
    def redistribute(self, turn, beam, tcomp, tconst):

        # I calc the total particles
        # and the max time
        nodeparticles = self.allreduce(np.array([beam.n_macroparticles]),
                                       operator='sum', comm=self.nodecomm)[0]

        nodetcomp = self.allreduce(np.array([tcomp]), operator='max',
                                   comm=self.nodecomm)[0]

        self.coefficients['particles'].append(nodeparticles)
        self.coefficients['times'].append(nodetcomp)

        # Keep only the last values
        self.coefficients['particles'] = self.coefficients['particles'][-self.dlb['coeffs_keep']:]
        self.coefficients['times'] = self.coefficients['times'][-self.dlb['coeffs_keep']:]

        # We pass weights to the polyfit
        # The weight function I am using is:
        # e(-x/T), where x is the abs(distance) from the last
        # datapoint, and T is the decay coefficient
        ncoeffs = len(self.coefficients['times'])
        weights = np.exp(-(ncoeffs - 1 - np.arange(ncoeffs))/self.dlb['decay'])
        # We model the runtime as latency * particles + c
        # where latency = p[1] and c = p[0]

        p = np.polyfit(
            self.coefficients['particles'],
            self.coefficients['times'],
            deg=1,
            w=weights)
        latency = p[0]
        tconst += p[1]

        sendbuf = np.array(
            [latency, tconst, nodeparticles, self.coefficients['intra_load']],
            dtype=float)
        recvbuf = np.empty(len(sendbuf) * self.workers, dtype=float)
        self.intercomm.Allgather(sendbuf, recvbuf)

        latencies = recvbuf[::len(sendbuf)]
        ctimes = recvbuf[1::len(sendbuf)]
        Pi_old = recvbuf[2::len(sendbuf)] / self.nodeworkers
        intra_loads = recvbuf[3::len(sendbuf)]

        # avgt = np.mean(synctimes)
        # Need to normalize this to the number of nodeworkers
        P = np.sum(Pi_old)

        sum1 = np.sum(ctimes/latencies)
        sum2 = np.sum(1./latencies)
        Pi = (P + sum1 - ctimes * sum2)/(latencies * sum2)

        # For the scheme to work I need that avgt > ctimes, if not
        # it means that a machine will be assigned negative number fo particles
        # I need to put a lower bound on the number of particles that
        # a machine can get, example 10% of the total/n_workers
        Pi = np.maximum(Pi, 0.1 * P / self.workers)

        dPi = np.rint(Pi_old - Pi)
        # I distribute the node particles to each node according to their
        # intra loads
        dPi = dPi * intra_loads

        for i in range(len(dPi)):
            # Here I update the delta particle, assigning to each worker
            # dPi[i] = self.coefficients['intra_load'] * dPi[i]
            if dPi[i] < 0 and -dPi[i] > Pi[i]:
                dPi[i] = -Pi[i]
            elif dPi[i] > Pi[i]:
                dPi[i] = Pi[i]

        # Need better definition of the cutoff
        # Maybe as a percentage of the number of particles
        # Let's say that each transaction has to be at least
        # 1% of total/n_workers
        transactions = calc_transactions(
            dPi, cutoff=self.dlb['cutoff'] * P / self.workers)[self.rank]
        if dPi[self.rank] > 0 and len(transactions) > 0:
            reqs = []
            tot_to_send = np.sum([t[1] for t in transactions])
            i = beam.n_macroparticles - tot_to_send
            for t in transactions:
                # I need to send t[1] particles to t[0]
                # buf[:t[1]] de, then dt, then id
                buf = np.empty(3*t[1], dtype=float)
                buf[0:t[1]] = beam.dE[i:i+t[1]]
                buf[t[1]:2*t[1]] = beam.dt[i:i+t[1]]
                buf[2*t[1]:3*t[1]] = beam.id[i:i+t[1]]
                i += t[1]
                # self.logger.critical(
                #     '[{}]: Sending {} parts to {}.'.format(self.rank, t[1], t[0]))
                reqs.append(self.intercomm.Isend(buf, t[0]))
            # Then I need to resize local beam.dt and beam.dE, also
            # beam.n_macroparticles
            beam.dE = beam.dE[:beam.n_macroparticles-tot_to_send]
            beam.dt = beam.dt[:beam.n_macroparticles-tot_to_send]
            beam.id = beam.id[:beam.n_macroparticles-tot_to_send]
            beam.n_macroparticles -= tot_to_send
            for req in reqs:
                req.Wait()
            # req[0].Waitall(req)
        elif dPi[self.rank] < 0 and len(transactions) > 0:
            reqs = []
            recvbuf = []
            for t in transactions:
                # I need to receive t[1] particles from t[0]
                # The buffer contains: de, dt, id
                buf = np.empty(3*t[1], float)
                recvbuf.append(buf)
                # self.logger.critical(
                #     '[{}]: Receiving {} parts from {}.'.format(self.rank, t[1], t[0]))
                reqs.append(self.intercomm.Irecv(buf, t[0]))
            for req in reqs:
                req.Wait()
            # req[0].Waitall(req)
            # Then I need to resize local beam.dt and beam.dE, also
            # beam.n_macroparticles
            tot_to_recv = np.sum([t[1] for t in transactions])
            beam.dE = np.resize(
                beam.dE, beam.n_macroparticles + tot_to_recv)
            beam.dt = np.resize(
                beam.dt, beam.n_macroparticles + tot_to_recv)
            beam.id = np.resize(
                beam.id, beam.n_macroparticles + tot_to_recv)
            i = beam.n_macroparticles
            for buf, t in zip(recvbuf, transactions):
                beam.dE[i:i+t[1]] = buf[0:t[1]]
                beam.dt[i:i+t[1]] = buf[t[1]:2*t[1]]
                beam.id[i:i+t[1]] = buf[2*t[1]:3*t[1]]
                i += t[1]
            beam.n_macroparticles += tot_to_recv

        if np.sum(np.abs(dPi))/2 < 1e-4 * P:
            self.interval = min(2*self.interval, 4000)
            return self.interval
        else:
            self.interval = self.start_interval
            return self.start_turn

    '''
    @timing.timeit(key='comm:redistribute')
    # @mpiprof.traceit(key='comm:redistribute')
    def redistribute(self, turn, beam, tcomp, tconst):
        self.coefficients['particles'].append(beam.n_macroparticles)
        self.coefficients['times'].append(tcomp)
        # Keep only the last values
        self.coefficients['particles'] = self.coefficients['particles'][-self.dlb['coeffs_keep']:]
        self.coefficients['times'] = self.coefficients['times'][-self.dlb['coeffs_keep']:]
        # We pass weights to the polyfit
        # The weight function I am using is:
        # e(-x/T), where x is the abs(distance) from the last
        # datapoint, and T is the decay coefficient
        ncoeffs = len(self.coefficients['times'])
        weights = np.exp(-(ncoeffs - 1 - np.arange(ncoeffs))/self.dlb['decay'])
        # We model the runtime as latency * particles + c
        # where latency = p[1] and c = p[0]
        p = np.polyfit(
            self.coefficients['particles'],
            self.coefficients['times'],
            deg=1,
            w=weights)
        latency = p[0]
        tconst += p[1]
        sendbuf = np.array(
            [latency, tconst, beam.n_macroparticles], dtype=float)
        recvbuf = np.empty(len(sendbuf) * self.workers, dtype=float)
        self.intercomm.Allgather(sendbuf, recvbuf)
        latencies = recvbuf[::3]
        ctimes = recvbuf[1::3]
        Pi_old = recvbuf[2::3]
        # avgt = np.mean(synctimes)
        P = np.sum(Pi_old)
        sum1 = np.sum(ctimes/latencies)
        sum2 = np.sum(1./latencies)
        Pi = (P + sum1 - ctimes * sum2)/(latencies * sum2)
        # For the scheme to work I need that avgt > ctimes, if not
        # it means that a machine will be assigned negative number fo particles
        # I need to put a lower bound on the number of particles that
        # a machine can get, example 10% of the total/n_workers
        Pi = np.maximum(Pi, 0.1 * P / self.workers)
        dPi = np.rint(Pi_old - Pi)
        for i in range(len(dPi)):
            if dPi[i] < 0 and -dPi[i] > Pi[i]:
                dPi[i] = -Pi[i]
            elif dPi[i] > Pi[i]:
                dPi[i] = Pi[i]
        # Need better definition of the cutoff
        # Maybe as a percentage of the number of particles
        # Let's say that each transaction has to be at least
        # 1% of total/n_workers
        transactions = calc_transactions(
            dPi, cutoff=self.dlb['cutoff'] * P / self.workers)[self.rank]
        if dPi[self.rank] > 0 and len(transactions) > 0:
            reqs = []
            tot_to_send = np.sum([t[1] for t in transactions])
            i = beam.n_macroparticles - tot_to_send
            for t in transactions:
                # I need to send t[1] particles to t[0]
                # buf[:t[1]] de, then dt, then id
                buf = np.empty(3*t[1], dtype=float)
                buf[0:t[1]] = beam.dE[i:i+t[1]]
                buf[t[1]:2*t[1]] = beam.dt[i:i+t[1]]
                buf[2*t[1]:3*t[1]] = beam.id[i:i+t[1]]
                i += t[1]
                # self.logger.critical(
                #     '[{}]: Sending {} parts to {}.'.format(self.rank, t[1], t[0]))
                reqs.append(self.intercomm.Isend(buf, t[0]))
            # Then I need to resize local beam.dt and beam.dE, also
            # beam.n_macroparticles
            beam.dE = beam.dE[:beam.n_macroparticles-tot_to_send]
            beam.dt = beam.dt[:beam.n_macroparticles-tot_to_send]
            beam.id = beam.id[:beam.n_macroparticles-tot_to_send]
            beam.n_macroparticles -= tot_to_send
            for req in reqs:
                req.Wait()
            # req[0].Waitall(req)
        elif dPi[self.rank] < 0 and len(transactions) > 0:
            reqs = []
            recvbuf = []
            for t in transactions:
                # I need to receive t[1] particles from t[0]
                # The buffer contains: de, dt, id
                buf = np.empty(3*t[1], float)
                recvbuf.append(buf)
                # self.logger.critical(
                #     '[{}]: Receiving {} parts from {}.'.format(self.rank, t[1], t[0]))
                reqs.append(self.intercomm.Irecv(buf, t[0]))
            for req in reqs:
                req.Wait()
            # req[0].Waitall(req)
            # Then I need to resize local beam.dt and beam.dE, also
            # beam.n_macroparticles
            tot_to_recv = np.sum([t[1] for t in transactions])
            beam.dE = np.resize(
                beam.dE, beam.n_macroparticles + tot_to_recv)
            beam.dt = np.resize(
                beam.dt, beam.n_macroparticles + tot_to_recv)
            beam.id = np.resize(
                beam.id, beam.n_macroparticles + tot_to_recv)
            i = beam.n_macroparticles
            for buf, t in zip(recvbuf, transactions):
                beam.dE[i:i+t[1]] = buf[0:t[1]]
                beam.dt[i:i+t[1]] = buf[t[1]:2*t[1]]
                beam.id[i:i+t[1]] = buf[2*t[1]:3*t[1]]
                i += t[1]
            beam.n_macroparticles += tot_to_recv
        if np.sum(np.abs(dPi))/2 < 1e-4 * P:
            self.interval = min(2*self.interval, 4000)
            return self.interval
        else:
            self.interval = self.start_interval
            return self.start_turn
    '''

    @timing.timeit(key='comm:intra_redistribute')
    # @mpiprof.traceit(key='comm:redistribute')
    def intra_redistribute(self, turn, beam, tcomp, tconst):
        self.coefficients['intra_particles'].append(beam.n_macroparticles)
        self.coefficients['intra_times'].append(tcomp)

        # Keep only the last values
        self.coefficients['intra_particles'] = self.coefficients['intra_particles'][-self.dlb['coeffs_keep']:]
        self.coefficients['intra_times'] = self.coefficients['intra_times'][-self.dlb['coeffs_keep']:]

        # We pass weights to the polyfit
        # The weight function I am using is:
        # e(-x/T), where x is the abs(distance) from the last
        # datapoint, and T is the decay coefficient
        ncoeffs = len(self.coefficients['intra_times'])
        weights = np.exp(-(ncoeffs - 1 - np.arange(ncoeffs))/self.dlb['decay'])
        # We model the runtime as latency * particles + c
        # where latency = p[1] and c = p[0]

        p = np.polyfit(
            self.coefficients['intra_particles'],
            self.coefficients['intra_times'],
            deg=1,
            w=weights)
        latency = p[0]
        tconst += p[1]

        sendbuf = np.array(
            [latency, tconst, beam.n_macroparticles], dtype=float)
        recvbuf = np.empty(len(sendbuf) * self.nodeworkers, dtype=float)
        self.nodecomm.Allgather(sendbuf, recvbuf)

        latencies = recvbuf[::3]
        ctimes = recvbuf[1::3]
        Pi_old = recvbuf[2::3]

        # avgt = np.mean(synctimes)
        P = np.sum(Pi_old)

        sum1 = np.sum(ctimes/latencies)
        sum2 = np.sum(1./latencies)
        Pi = (P + sum1 - ctimes * sum2)/(latencies * sum2)

        # For the scheme to work I need that avgt > ctimes, if not
        # it means that a machine will be assigned negative number fo particles
        # I need to put a lower bound on the number of particles that
        # a machine can get, example 10% of the total/n_workers
        Pi = np.maximum(Pi, 0.1 * P / self.nodeworkers)

        # Here we store the percent of the total node load that goes to each
        # intra-node worker
        self.coefficients['intra_load'] = Pi[self.noderank] / P

        dPi = np.rint(Pi_old - Pi)

        for i in range(len(dPi)):
            if dPi[i] < 0 and -dPi[i] > Pi[i]:
                dPi[i] = -Pi[i]
            elif dPi[i] > Pi[i]:
                dPi[i] = Pi[i]

        # Need better definition of the cutoff
        # Maybe as a percentage of the number of particles
        # Let's say that each transaction has to be at least
        # 1% of total/n_workers
        transactions = calc_transactions(
            dPi, cutoff=self.dlb['cutoff'] * P / self.nodeworkers)[self.noderank]
        if dPi[self.noderank] > 0 and len(transactions) > 0:
            reqs = []
            tot_to_send = np.sum([t[1] for t in transactions])
            i = beam.n_macroparticles - tot_to_send
            for t in transactions:
                # I need to send t[1] particles to t[0]
                # buf[:t[1]] de, then dt, then id
                buf = np.empty(3*t[1], dtype=float)
                buf[0:t[1]] = beam.dE[i:i+t[1]]
                buf[t[1]:2*t[1]] = beam.dt[i:i+t[1]]
                buf[2*t[1]:3*t[1]] = beam.id[i:i+t[1]]
                i += t[1]
                # self.logger.critical(
                #     '[{}]: Sending {} parts to {}.'.format(self.rank, t[1], t[0]))
                reqs.append(self.nodecomm.Isend(buf, t[0]))
            # Then I need to resize local beam.dt and beam.dE, also
            # beam.n_macroparticles
            beam.dE = beam.dE[:beam.n_macroparticles-tot_to_send]
            beam.dt = beam.dt[:beam.n_macroparticles-tot_to_send]
            beam.id = beam.id[:beam.n_macroparticles-tot_to_send]
            beam.n_macroparticles -= tot_to_send
            for req in reqs:
                req.Wait()
            # req[0].Waitall(req)
        elif dPi[self.noderank] < 0 and len(transactions) > 0:
            reqs = []
            recvbuf = []
            for t in transactions:
                # I need to receive t[1] particles from t[0]
                # The buffer contains: de, dt, id
                buf = np.empty(3*t[1], float)
                recvbuf.append(buf)
                # self.logger.critical(
                #     '[{}]: Receiving {} parts from {}.'.format(self.rank, t[1], t[0]))
                reqs.append(self.nodecomm.Irecv(buf, t[0]))
            for req in reqs:
                req.Wait()
            # req[0].Waitall(req)
            # Then I need to resize local beam.dt and beam.dE, also
            # beam.n_macroparticles
            tot_to_recv = np.sum([t[1] for t in transactions])
            beam.dE = np.resize(
                beam.dE, beam.n_macroparticles + tot_to_recv)
            beam.dt = np.resize(
                beam.dt, beam.n_macroparticles + tot_to_recv)
            beam.id = np.resize(
                beam.id, beam.n_macroparticles + tot_to_recv)
            i = beam.n_macroparticles
            for buf, t in zip(recvbuf, transactions):
                beam.dE[i:i+t[1]] = buf[0:t[1]]
                beam.dt[i:i+t[1]] = buf[t[1]:2*t[1]]
                beam.id[i:i+t[1]] = buf[2*t[1]:3*t[1]]
                i += t[1]
            beam.n_macroparticles += tot_to_recv

        if np.sum(np.abs(dPi))/2 < 1e-4 * P:
            self.interval = min(2*self.interval, 4000)
            return self.interval
        else:
            self.interval = self.start_interval
            return self.start_turn

    def report(self, scope, turn, beam, tcomp, tcomm, tconst, tsync):
        latency = tcomp / beam.n_macroparticles
        if self.log:
            self.logger.critical('Scope {} [{}]: Turn {}, Tconst {:g}, Tcomp {:g}, Tcomm {:g}, Tsync {:g}, Latency {:g}, Particles {:g}'.format(
                scope, self.rank, turn, tconst, tcomp, tcomm, tsync, latency, beam.n_macroparticles))

    def greet(self):
        if self.log:
            self.logger.debug('greet')
        print('[{}]@{}: Hello World!'.format(self.rank, self.hostname))

    def print_version(self):
        if self.log:
            self.logger.debug('version')
        # print('[{}] Library version: {}'.format(self.rank, MPI.Get_library_version()))
        # print('[{}] Version: {}'.format(self.rank,MPI.Get_version()))
        print('[{}] Library: {}'.format(self.rank, MPI.get_vendor()))

    def timer_start(self, phase):
        if phase not in self.times:
            self.times[phase] = {'start': MPI.Wtime(), 'total': 0.}
        else:
            self.times[phase]['start'] = MPI.Wtime()

    def timer_stop(self, phase):
        self.times[phase]['total'] += MPI.Wtime() - self.times[phase]['start']

    def timer_reset(self, phase):
        self.times[phase] = {'start': MPI.Wtime(), 'total': 0.}

    def initDLB(self, lbstr, n_iter):
        # lbstr = lbtype,lbarg,cutoff,decay
        self.inter_lb_turns = []

        self.lb_type = lbstr.split(',')[0]

        if self.lb_type != 'off':
            assert len(lbstr.split(',')) == 5, 'Wrong number of LB arguments'
            lb_arg, cutoff, decay, keep = lbstr.split(',')[1:]
            if not cutoff:
                cutoff = 0.03
            if not decay:
                decay = 5
            if not keep:
                keep = 20

            if self.lb_type == 'times':
                if lb_arg:
                    intv = max(n_iter // (int(lb_arg)+1), 1)
                else:
                    intv = max(n_iter // (10 + 1), 1)
                self.inter_lb_turns = np.arange(0, n_iter, intv)[1:]

            elif self.lb_type == 'interval':
                if lb_arg:
                    self.inter_lb_turns = np.arange(0, n_iter, int(lb_arg))[1:]
                else:
                    self.inter_lb_turns = np.arange(0, n_iter, 1000)[1:]
            elif self.lb_type == 'dynamic':
                self.inter_lb_turns = [self.start_turn]
            elif self.lb_type == 'reportonly':
                if lb_arg:
                    self.inter_lb_turns = np.arange(0, n_iter, int(lb_arg))
                else:
                    self.inter_lb_turns = np.arange(0, n_iter, 100)
            self.dlb = {'tcomp': 0, 'tcomm': 0,
                        'tconst': 0, 'tsync': 0,
                        'cutoff': float(cutoff), 'decay': float(decay),
                        'coeffs_keep': int(keep),
                        'intra_tcomp': 0,
                        'intra_tconst': 0,
                        'intra_tsync': 0,
                        'intra_tcomm': 0}
        # to begin with, we make them equal
        self.intra_lb_turns = np.copy(self.inter_lb_turns)
        return self.inter_lb_turns

    def DLB(self, turn, beam, withtp=False):
        intv = 0
        if (withtp):
            if turn in self.intra_lb_turns:
                tcomp_new = timing.get(['comp:'])
                # tcomm_new = timing.get(['comm:'])
                tconst_new = 0
                tsync_new = timing.get(
                    ['serial:sync', 'serial:intraSync', 'serial:gpuSync'])
                if self.lb_type != 'reportonly':
                    intv = self.intra_redistribute(turn, beam,
                                                tcomp=tcomp_new -
                                                self.dlb['intra_tcomp'],
                                                # tsync=tsync_new - self.dlb['tsync'])
                                                tconst=0)

                    # tconst=((tconst_new-self.dlb['tconst'])
                    #         + (tcomm_new - self.dlb['tcomm'])))
                # if self.lb_type == 'dynamic':
                #     self.inter_lb_turns[0] += intv
                self.report('intra', turn, beam, tcomp=tcomp_new-self.dlb['intra_tcomp'],
                            tcomm=0,
                            tconst=0,
                            tsync=0)
                self.dlb['intra_tcomp'] = tcomp_new
                # self.dlb['intra_tcomm'] = tcomm_new
                # self.dlb['intra_tconst'] = tconst_new
                # self.dlb['intra_tsync'] = tsync_new
                # return intv
        else:
        # to considere tconst also for the dlb
            if turn in self.intra_lb_turns:
                tcomp_new = timing.get(['comp:'])
                tcomm_new = timing.get(['comm:'])
                tconst_new = timing.get(['serial:'], exclude_lst=[
                                'serial:sync', 'serial:intraSync', 'serial:gpuSync'])

                tsync_new = timing.get(
                    ['serial:sync', 'serial:intraSync', 'serial:gpuSync'])
                if self.lb_type != 'reportonly':
                    intv = self.intra_redistribute(turn, beam,
                                                tcomp=tcomp_new -
                                                self.dlb['intra_tcomp'],
                                                tconst=((tconst_new-self.dlb['intra_tconst'])
                                                    + (tcomm_new - self.dlb['intra_tcomm'])))

                self.report('intra', turn, beam, tcomp=tcomp_new-self.dlb['intra_tcomp'],
                            tcomm=tcomm_new-self.dlb['intra_tcomm'],
                            tconst=tconst_new-self.dlb['intra_tconst'],
                            tsync=tsync_new-self.dlb['intra_tsync'])
                self.dlb['intra_tcomp'] = tcomp_new
                self.dlb['intra_tcomm'] = tcomm_new
                self.dlb['intra_tconst'] = tconst_new
                self.dlb['intra_tsync'] = tsync_new
                # return intv
        # This is the external LB
        if turn in self.inter_lb_turns:
            tcomp_new = timing.get(['comp:'])
            tcomm_new = timing.get(['comm:'])
            tconst_new = timing.get(['serial:'], exclude_lst=[
                                    'serial:sync', 'serial:intraSync', 'serial:gpuSync'])
            tsync_new = timing.get(
                ['serial:sync', 'serial:intraSync', 'serial:gpuSync'])
            if self.lb_type != 'reportonly':
                intv = self.redistribute(turn, beam,
                                         tcomp=tcomp_new-self.dlb['tcomp'],
                                         # tsync=tsync_new - self.dlb['tsync'])
                                         tconst=0)
                                         # tconst=((tconst_new-self.dlb['tconst'])
                                         #         + (tcomm_new - self.dlb['tcomm'])))
            if self.lb_type == 'dynamic':
                self.inter_lb_turns[0] += intv
            self.report('inter', turn, beam, tcomp=tcomp_new-self.dlb['tcomp'],
                        tcomm=tcomm_new-self.dlb['tcomm'],
                        tconst=tconst_new-self.dlb['tconst'],
                        tsync=tsync_new-self.dlb['tsync'])
            self.dlb['tcomp'] = tcomp_new
            self.dlb['tcomm'] = tcomm_new
            self.dlb['tconst'] = tconst_new
            self.dlb['tsync'] = tsync_new
        
        # return intv

def calc_transactions(dpi, cutoff):
    trans = {}
    arr = []
    for i in enumerate(dpi):
        trans[i[0]] = []
        arr.append({'val': i[1], 'id': i[0]})

    # First pass is to prioritize transactions within the same node
    # basically transactions between worker i and i + 1, i: 0, 2, 4, ...
    i = 0
    # e = len(arr)-1
    while i < len(arr)-1:
        if (arr[i]['val'] < 0) and (arr[i+1]['val'] > 0):
            s = i+1
            r = i
        elif (arr[i]['val'] > 0) and (arr[i+1]['val'] < 0):
            s = i
            r = i+1
        else:
            i += 2
            continue
        diff = int(min(abs(arr[s]['val']), abs(arr[r]['val'])))
        if diff > cutoff:
            trans[arr[s]['id']].append((arr[r]['id'], diff))
            trans[arr[r]['id']].append((arr[s]['id'], diff))
            arr[s]['val'] -= diff
            arr[r]['val'] += diff
        i += 2
    # Then the internode transactions
    arr = sorted(arr, key=lambda x: x['val'], reverse=True)
    s = 0
    e = len(arr)-1
    while (s < e) and (arr[s]['val'] >= 0) and (arr[e]['val'] <= 0):
        if arr[s]['val'] <= cutoff:
            s += 1
            continue
        if abs(arr[e]['val']) <= cutoff:
            e -= 1
            continue
        diff = int(min(abs(arr[s]['val']), abs(arr[e]['val'])))
        trans[arr[s]['id']].append((arr[e]['id'], diff))
        trans[arr[e]['id']].append((arr[s]['id'], diff))
        arr[s]['val'] -= diff
        arr[e]['val'] += diff

    return trans


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