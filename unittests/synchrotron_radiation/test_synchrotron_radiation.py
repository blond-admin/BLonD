import unittest

from blond.synchrotron_radiation.synchrotron_radiation import SynchrotronRadiation


class TestSynchrotronRadiation(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.synchrotron_radiation = SynchrotronRadiation(
            Ring=None,
            RFParameters=None,
            Beam=None,
            bending_radius=None,
            n_kicks=None,
            quantum_excitation=None,
            python=None,
            seed=None,
            shift_beam=None,
        )

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_print_SR_params(self):
        # TODO: implement test for `print_SR_params`
        self.synchrotron_radiation.print_SR_params()

    @unittest.skip
    def test_to_cpu(self):
        # TODO: implement test for `to_cpu`
        self.synchrotron_radiation.to_cpu(recursive=None)

    @unittest.skip
    def test_to_gpu(self):
        # TODO: implement test for `to_gpu`
        self.synchrotron_radiation.to_gpu(recursive=None)

    @unittest.skip
    def test_track_SR_C(self):
        # TODO: implement test for `track_SR_C`
        self.synchrotron_radiation.track_SR_C()

    @unittest.skip
    def test_track_SR_python(self):
        # TODO: implement test for `track_SR_python`
        self.synchrotron_radiation.track_SR_python()

    @unittest.skip
    def test_track_full_C(self):
        # TODO: implement test for `track_full_C`
        self.synchrotron_radiation.track_full_C()

    @unittest.skip
    def test_track_full_python(self):
        # TODO: implement test for `track_full_python`
        self.synchrotron_radiation.track_full_python()
