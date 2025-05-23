import unittest

from blond.utils.mpi_config import master_wrap, sequential_wrap, MPILog, Worker


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_master_wrap(self):
        # TODO: implement test for `master_wrap`
        master_wrap(func=None)

    @unittest.skip
    def test_sequential_wrap(self):
        # TODO: implement test for `sequential_wrap`
        sequential_wrap(func=None, beam=None, split_args=None, gather_args=None)



class TestMPILog(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.mpi_log = MPILog(rank=None, log_dir=None)
    @unittest.skip
    def test___init__(self):
        pass # calls __init__ in  self.setUp

    @unittest.skip
    def test_critical(self):
        # TODO: implement test for `critical`
        self.mpi_log.critical(string=None)

    @unittest.skip
    def test_debug(self):
        # TODO: implement test for `debug`
        self.mpi_log.debug(string=None)

    @unittest.skip
    def test_disable(self):
        # TODO: implement test for `disable`
        self.mpi_log.disable()

    @unittest.skip
    def test_info(self):
        # TODO: implement test for `info`
        self.mpi_log.info(string=None)


class TestWorker(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.worker = Worker()
    @unittest.skip
    def test___init__(self):
        pass # calls __init__ in  self.setUp

    @unittest.skip
    def test_allgather(self):
        # TODO: implement test for `allgather`
        self.worker.allgather(var=None)

    @unittest.skip
    def test_allreduce(self):
        # TODO: implement test for `allreduce`
        self.worker.allreduce(sendbuf=None, recvbuf=None, dtype=None, operator=None, comm=None)

    @unittest.skip
    def test_assign_gpus(self):
        # TODO: implement test for `assign_gpus`
        self.worker.assign_gpus(num_gpus=None)

    @unittest.skip
    def test_broadcast(self):
        # TODO: implement test for `broadcast`
        self.worker.broadcast(var=None, root=None)

    @unittest.skip
    def test_finalize(self):
        # TODO: implement test for `finalize`
        self.worker.finalize()

    @unittest.skip
    def test_gather(self):
        # TODO: implement test for `gather`
        self.worker.gather(var=None)

    @unittest.skip
    def test_greet(self):
        # TODO: implement test for `greet`
        self.worker.greet()

    @unittest.skip
    def test_init_log(self):
        # TODO: implement test for `init_log`
        self.worker.init_log(log=None, logdir=None)

    @unittest.skip
    def test_print_version(self):
        # TODO: implement test for `print_version`
        self.worker.print_version()

    @unittest.skip
    def test_reduce(self):
        # TODO: implement test for `reduce`
        self.worker.reduce(sendbuf=None, recvbuf=None, dtype=None, operator=None, comm=None)

    @unittest.skip
    def test_scatter(self):
        # TODO: implement test for `scatter`
        self.worker.scatter(var=None)

    @unittest.skip
    def test_sync(self):
        # TODO: implement test for `sync`
        self.worker.sync()
