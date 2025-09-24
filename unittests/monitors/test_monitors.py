import unittest

from blond.monitors.monitors import BunchMonitor, MultiBunchMonitor, SlicesMonitor


class TestBunchMonitor(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.bunch_monitor = BunchMonitor(
            Ring=None,
            RFParameters=None,
            Beam=None,
            filename=None,
            buffer_time=None,
            Profile=None,
            PhaseLoop=None,
            LHCNoiseFB=None,
        )

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_init_buffer(self):
        # TODO: implement test for `init_buffer`
        self.bunch_monitor.init_buffer()

    @unittest.skip
    def test_init_data(self):
        # TODO: implement test for `init_data`
        self.bunch_monitor.init_data(filename=None, dims=None)

    @unittest.skip
    def test_write_buffer(self):
        # TODO: implement test for `write_buffer`
        self.bunch_monitor.write_buffer()

    @unittest.skip
    def test_write_data(self):
        # TODO: implement test for `write_data`
        self.bunch_monitor.write_data(h5group=None, dims=None)


class TestMultiBunchMonitor(unittest.TestCase):
    @unittest.skip
    def test___del__(self):
        # TODO: implement test for `__del__`
        self.multi_bunch_monitor.__del__()

    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.multi_bunch_monitor = MultiBunchMonitor(
            filename=None,
            n_turns=None,
            profile=None,
            rf=None,
            Nbunches=None,
            buffer_size=None,
        )

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_close(self):
        # TODO: implement test for `close`
        self.multi_bunch_monitor.close()

    @unittest.skip
    def test_create_data(self):
        # TODO: implement test for `create_data`
        self.multi_bunch_monitor.create_data(
            name=None, h5group=None, dims=None, dtype=None
        )

    @unittest.skip
    def test_track(self):
        # TODO: implement test for `track`
        self.multi_bunch_monitor.track(turn=None)

    @unittest.skip
    def test_write_buffer(self):
        # TODO: implement test for `write_buffer`
        self.multi_bunch_monitor.write_buffer(turn=None)

    @unittest.skip
    def test_write_data(self):
        # TODO: implement test for `write_data`
        self.multi_bunch_monitor.write_data()


class TestSlicesMonitor(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.slices_monitor = SlicesMonitor(filename=None, n_turns=None, profile=None)

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_close(self):
        # TODO: implement test for `close`
        self.slices_monitor.close()

    @unittest.skip
    def test_create_data(self):
        # TODO: implement test for `create_data`
        self.slices_monitor.create_data(h5group=None, dims=None)

    @unittest.skip
    def test_track(self):
        # TODO: implement test for `track`
        self.slices_monitor.track(bunch=None)

    @unittest.skip
    def test_write_data(self):
        # TODO: implement test for `write_data`
        self.slices_monitor.write_data(bunch=None, h5group=None, i_turn=None)
