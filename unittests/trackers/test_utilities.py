import unittest
import numpy as np

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
            Ring=None,
            RFStation=None,
            Beam=None,
            dt=None,
            dE=None,
            total_voltage=None,
        )

    @unittest.skip
    def test_is_in_separatrix(self):
        # TODO: implement test for `is_in_separatrix`
        is_in_separatrix(
            Ring=None,
            RFStation=None,
            Beam=None,
            dt=None,
            dE=None,
            total_voltage=None,
        )

    @unittest.skip
    def test_potential_well_cut(self):
        time_potential = np.linspace(0, 1, 50)
        potential_array = np.array([np.sin(k * np.pi / 51) for k in range(10)])
        with self.assertRaises(
            ValueError, msg=f"{len(time_potential)} != {len(potential_array)}"
        ):
            potential_well_cut(
                time_potential=time_potential, potential_array=potential_array
            )
        # No minimum
        potential_array = np.array([np.sin(k * np.pi / 51) for k in range(50)])
        with self.assertRaises(
            RuntimeError, msg="'The potential well has no minima...'"
        ):
            potential_well_cut(
                time_potential=time_potential, potential_array=potential_array
            )
        # n_minima > n_maxima == 1:
        # TODO: implement test for `potential_well_cut`
        potential_well_cut(time_potential=None, potential_array=None)

        # No maximum
        potential_array = np.array(
            [1 / np.sin((k + 1) * np.pi / 50) for k in range(50)]
        )

        with self.assertWarns(
            UserWarning,
            msg="The maximum of the potential well could not be found... \
                    You may reconsider the options to calculate the potential well \
                    as the main harmonic is probably not the expected one. \
                    You may also increase the percentage of margin to compute \
                    the potentiel well. The full potential well will be taken'",
        ):
            time_potential_sep, potential_well_sep = potential_well_cut(
                time_potential=time_potential, potential_array=potential_array
            )
        np.testing.assert_equal(time_potential_sep, time_potential)
        np.testing.assert_equal(potential_well_sep, potential_array)

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
