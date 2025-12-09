import unittest

# BLonD imports
# --------------
from blond.beam.beam import Beam, Electron
from blond.beam.profile import Profile, CutOptions
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.trackers.tracker import FullRingAndRF, RingAndRFTracker
from blond.impedances.impedance_sources import CoherentSynchrotronRadiation
from blond.impedances.impedance import InducedVoltageFreq, \
    TotalInducedVoltage
from blond.synchrotron_radiation.synchrotron_radiation import SynchrotronRadiation
from blond.beam.distributions import Haissinski
import scipy

from blond.beam.distributions import (
    x0_from_bunch_length,
    distribution_function,
    line_density,
    matched_from_distribution_function,
    matched_from_line_density,
    parabolic,
)


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_x0_from_bunch_length(self):
        # TODO: implement test for `x0_from_bunch_length`
        x0_from_bunch_length(
            bunch_length=None,
            bunch_length_fit=None,
            X_grid=None,
            sorted_X_dE0=None,
            n_points_grid=None,
            time_potential_low_res=None,
            distribution_function_=None,
            distribution_type=None,
            distribution_exponent=None,
            beam=None,
            full_ring_and_RF=None,
        )

    @unittest.skip
    def test_distribution_function(self):
        # TODO: implement test for `distribution_function`
        distribution_function(
            action_array=None, dist_type=None, length=None, exponent=None
        )

    @unittest.skip
    def test_line_density(self):
        # TODO: implement test for `line_density`
        line_density(
            coord_array=None,
            dist_type=None,
            bunch_length=None,
            bunch_position=None,
            exponent=None,
        )

    @unittest.skip
    def test_matched_from_distribution_function(self):
        # TODO: implement test for `matched_from_distribution_function`
        matched_from_distribution_function(
            beam=None,
            full_ring_and_RF=None,
            distribution_function_input=None,
            distribution_user_table=None,
            main_harmonic_option=None,
            TotalInducedVoltage=None,
            n_iterations=None,
            n_points_potential=None,
            n_points_grid=None,
            dt_margin_percent=None,
            extraVoltageDict=None,
            seed=None,
            distribution_exponent=None,
            distribution_type=None,
            emittance=None,
            bunch_length=None,
            bunch_length_fit=None,
            distribution_variable=None,
            process_pot_well=None,
            turn_number=None,
        )

    @unittest.skip
    def test_matched_from_line_density(self):
        # TODO: implement test for `matched_from_line_density`
        matched_from_line_density(
            beam=None,
            full_ring_and_RF=None,
            line_density_input=None,
            main_harmonic_option=None,
            TotalInducedVoltage=None,
            plot=None,
            figdir=None,
            half_option=None,
            extraVoltageDict=None,
            n_iterations=None,
            n_points_potential=None,
            n_points_grid=None,
            dt_margin_percent=None,
            n_points_abel=None,
            bunch_length=None,
            line_density_type=None,
            line_density_exponent=None,
            seed=None,
            process_pot_well=None,
        )

    @unittest.skip
    def test_parabolic(self):
        # TODO: implement test for `parabolic`
        parabolic(
            Ring=None,
            RFStation=None,
            Beam=None,
            bunch_length=None,
            bunch_position=None,
            bunch_energy=None,
            energy_spread=None,
            seed=None,
        )


class testHaissinskiSolution(unittest.TestCase):
    # Run before every test
    def setUp(self):
        # based on BLonD example 29
        R_bend = 5.559  # [m]
        ring = Ring(110.4, 5e-4, 1.3e9, Electron(), n_turns=1,
                    synchronous_data_type='total energy')
        self.beam = Beam(ring, 100, 344e6)
        rf_station = RFStation(ring, 184, 1e6, -0.045466)
        profile = Profile(self.beam,
                          cut_options = CutOptions(cut_left=-5*4.12e-12,
                                                   cut_right=5*4.12e-12,
                                                   n_slices=128))

        cSR_source = CoherentSynchrotronRadiation(R_bend)
        cSR_impedance = InducedVoltageFreq(self.beam, profile, [cSR_source],
                                           frequency_resolution=6.826e9)
        inducedVoltage = TotalInducedVoltage(self.beam, profile, [cSR_impedance])

        tracker = RingAndRFTracker(rf_station, self.beam, profile=profile,
                                   total_induced_voltage=inducedVoltage,
                                   interpolation=True)
        self.ring_tracker = FullRingAndRF([tracker])

        self.iSR = SynchrotronRadiation(ring, rf_station, self.beam, R_bend, n_kicks=1,
                                        quantum_excitation=True, shift_beam=False)

    # Run after every test
    def tearDown(self):
        del self.ring_tracker
        del self.beam
        del self.iSR

    def test_Haissinski_verbose_1(self):
        haissinski_solution = Haissinski(self.ring_tracker, self.iSR, verbose=False)
        self.assertIsNone(haissinski_solution)

    def test_Haissinski_verbose_2(self):
        haissinski_solution = Haissinski(self.ring_tracker, self.iSR, verbose=True)
        self.assertIsInstance(haissinski_solution, scipy.optimize._optimize.OptimizeResult)

    def test_Haissinski_root_kwargs(self):
        haissinski_solution = Haissinski(self.ring_tracker, self.iSR, verbose=True,
                                         root_kwargs={'method':'lm'})
        self.assertEqual(haissinski_solution.method, 'lm',
                         msg="Root key word arguments not passed correctly.")


if __name__ == "__main__":
    unittest.main()
