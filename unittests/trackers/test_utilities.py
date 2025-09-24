import unittest
import numpy as np

from blond.trackers.utilities import minmax_location, potential_well_cut


class TestUtilities(unittest.TestCase):
    def _total_voltage(self):
        pass
    def _hamiltonian(self):
        pass
    def _separatrix(self):
        pass
    def _test_is_in_separatrix(self):
        pass
    def _test_minmax_location(self):
        pass
    def test_potential_well_cut(self):
        time_potential = np.linspace(0, 1, 50)
        potential_array = np.array([np.sin(k * np.pi / 51) for k in range(10)])
        with self.assertRaises(ValueError,
                               msg=f"{len(time_potential)} != "
                                   f"{len(potential_array)}"):
            potential_well_cut(time_potential=time_potential,
                               potential_array=potential_array
                               )
        # No minimum
        potential_array = np.array([np.sin(k * np.pi / 51) for k in range(50)])
        with self.assertRaises(RuntimeError,
                               msg="'The potential well has no minima...'"):
            potential_well_cut(time_potential=time_potential,
                               potential_array=potential_array
                               )
        #n_minima > n_maxima == 1:

        #No maximum
        potential_array = np.array([1/np.sin((k+1) * np.pi / 50) for k in
                                    range(50)])

        with self.assertWarns(UserWarning,
                               msg="The maximum of the potential well could not be found... \
                You may reconsider the options to calculate the potential well \
                as the main harmonic is probably not the expected one. \
                You may also increase the percentage of margin to compute \
                the potentiel well. The full potential well will be taken'"):
            time_potential_sep, potential_well_sep = potential_well_cut(
                time_potential=time_potential,
                               potential_array=potential_array
                               )
        np.testing.assert_equal(time_potential_sep, time_potential)
        np.testing.assert_equal(potential_well_sep, potential_array)
    def _test_phase_modulo_above_transition(self):
        pass
    def _test_phase_modulo_below_transition(self):
        pass
    def _time_modulo(self):
        pass