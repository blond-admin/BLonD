import unittest

from blond.trackers.utilities import (
    hamiltonian,
    is_in_separatrix,
    potential_well_cut,
    separatrix,
    synchrotron_frequency_distribution,
    total_voltage,
    synchrotron_frequency_tracker,
)


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_hamiltonian(self):
        # TODO: implement test for `hamiltonian`
        hamiltonian(
            Ring=None, RFStation=None, Beam=None, dt=None, dE=None, total_voltage=None
        )

    @unittest.skip
    def test_is_in_separatrix(self):
        # TODO: implement test for `is_in_separatrix`
        is_in_separatrix(
            Ring=None, RFStation=None, Beam=None, dt=None, dE=None, total_voltage=None
        )

    @unittest.skip
    def test_potential_well_cut(self):
        # TODO: implement test for `potential_well_cut`
        potential_well_cut(time_potential=None, potential_array=None)

    @unittest.skip
    def test_separatrix(self):
        # TODO: implement test for `separatrix`
        separatrix(Ring=None, RFStation=None, dt=None)

    @unittest.skip
    def test_synchrotron_frequency_distribution(self):
        # TODO: implement test for `synchrotron_frequency_distribution`
        synchrotron_frequency_distribution(
            Beam=None,
            FullRingAndRF=None,
            main_harmonic_option=None,
            turn=None,
            TotalInducedVoltage=None,
            smoothOption=None,
        )

    @unittest.skip
    def test_total_voltage(self):
        # TODO: implement test for `total_voltage`
        total_voltage(RFsection_list=None, harmonic=None)


class Testsynchrotron_frequency_tracker(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.synchrotron_frequency_tracker = synchrotron_frequency_tracker(
            Ring=None,
            n_macroparticles=None,
            theta_coordinate_range=None,
            FullRingAndRF=None,
            TotalInducedVoltage=None,
        )

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_frequency_calculation(self):
        # TODO: implement test for `frequency_calculation`
        self.synchrotron_frequency_tracker.frequency_calculation(
            n_sampling=None, start_turn=None, end_turn=None
        )

    @unittest.skip
    def test_track(self):
        # TODO: implement test for `track`
        self.synchrotron_frequency_tracker.track()
