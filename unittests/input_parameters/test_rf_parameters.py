import unittest

from blond.input_parameters.rf_parameters import calculate_phi_s, RFStation


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_calculate_phi_s(self):
        # TODO: implement test for `calculate_phi_s`
        calculate_phi_s(RFStation=None, Particle=None, accelerating_systems=None)


class TestRFStation(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.rf_station = RFStation(
            Ring=None,
            harmonic=None,
            voltage=None,
            phi_rf_d=None,
            n_rf=None,
            section_index=None,
            omega_rf=None,
            phi_noise=None,
            phi_modulation=None,
            RFStationOptions=None,
        )

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_eta_tracking(self):
        # TODO: implement test for `eta_tracking`
        self.rf_station.eta_tracking(beam=None, counter=None, dE=None)

    @unittest.skip
    def test_to_cpu(self):
        # TODO: implement test for `to_cpu`
        self.rf_station.to_cpu(recursive=None)

    @unittest.skip
    def test_to_gpu(self):
        # TODO: implement test for `to_gpu`
        self.rf_station.to_gpu(recursive=None)
