import unittest

# BLonD imports
# --------------
from blond.legacy.blond2.beam.beam import Beam, Electron
from blond.legacy.blond2.beam.profile import Profile, CutOptions
from blond.legacy.blond2.input_parameters.rf_parameters import RFStation
from blond.legacy.blond2.input_parameters.ring import Ring
from blond.legacy.blond2.trackers.tracker import FullRingAndRF, RingAndRFTracker
from blond.legacy.blond2.impedances.impedance_sources import CoherentSynchrotronRadiation
from blond.legacy.blond2.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.legacy.blond2.synchrotron_radiation.synchrotron_radiation import (
    SynchrotronRadiation,
)
from blond.legacy.blond2.beam.distributions import Haissinski
from blond.legacy.blond2.utils import bmath as bm
import scipy

from blond.legacy.blond2.beam.distributions import (
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
        ring = Ring(
            110.4,
            5e-4,
            1.3e9,
            Electron(),
            n_turns=1,
            synchronous_data_type="total energy",
        )
        self.beam = Beam(ring, 100, 344e6)
        self.rf_station = RFStation(ring, 184, 1e6, -0.045466)
        self.profile = Profile(
            self.beam,
            cut_options=CutOptions(
                cut_left=-5 * 4.12e-12, cut_right=5 * 4.12e-12, n_slices=128
            ),
        )

        cSR_source = CoherentSynchrotronRadiation(R_bend)
        cSR_impedance = InducedVoltageFreq(
            self.beam, self.profile, [cSR_source], frequency_resolution=6.826e9
        )
        inducedVoltage = TotalInducedVoltage(
            self.beam, self.profile, [cSR_impedance]
        )

        tracker = RingAndRFTracker(
            self.rf_station,
            self.beam,
            profile=self.profile,
            total_induced_voltage=inducedVoltage,
            interpolation=True,
        )
        self.ring_tracker = FullRingAndRF([tracker])

        self.SR = SynchrotronRadiation(
            ring,
            self.rf_station,
            self.beam,
            R_bend,
            n_kicks=1,
            quantum_excitation=True,
            shift_beam=False,
        )
        # equilibrium energy spread [eV]
        self.sigma_E = ring.energy[0, 0] * self.SR.sigma_dE
        # zero-current bunch duration [s]
        self.sigma_dt0 = (
            abs(ring.eta_0[0, 0])
            / (ring.beta[0, 0] * ring.energy[0, 0])
            * self.sigma_E
            / self.rf_station.omega_s0[0]
        )

    # Run after every test
    def tearDown(self):
        del self.ring_tracker
        del self.beam
        del self.SR

    def test_Haissinski_verbose_1(self):
        haissinski_solution = Haissinski(
            self.ring_tracker, self.SR, verbose=False
        )
        self.assertIsNone(haissinski_solution)

    def test_Haissinski_verbose_2(self):
        haissinski_solution = Haissinski(
            self.ring_tracker, self.SR, verbose=True
        )
        self.assertIsInstance(
            haissinski_solution, scipy.optimize._optimize.OptimizeResult
        )

    def test_Haissinski_root_kwargs(self):
        haissinski_solution = Haissinski(
            self.ring_tracker,
            self.SR,
            verbose=True,
            root_kwargs={"method": "lm"},
        )
        self.assertEqual(
            haissinski_solution.method,
            "lm",
            msg="Root key word arguments not passed correctly.",
        )

    def test_sigma_E(self):
        # test that Haissinski solution has equilibrium energy spread
        Haissinski(self.ring_tracker, self.SR, verbose=False, seed=1789 * 1989)
        self.assertAlmostEqual(self.sigma_E / bm.std(self.beam.dE), 0.99, 2)

    def test_sigma_zero(self):
        # test zero-current bunch length
        self.beam.n_macroparticles = int(
            1e5
        )  # increase MPs for better statistics
        tracker = RingAndRFTracker(
            self.rf_station,
            self.beam,
            profile=self.profile,
            total_induced_voltage=None,  # no collective effects
            interpolation=True,
        )
        _ring_tracker = FullRingAndRF([tracker])
        Haissinski(_ring_tracker, self.SR, verbose=False, seed=1789 * 1989)
        self.assertAlmostEqual(
            bm.std(self.beam.dt) * 1e12, self.sigma_dt0 * 1e12, 1
        )

    def test_sigma_dt(self):
        # test bunch length of Haissinski solution; close to 3.92 ps for 100 MPs
        Haissinski(self.ring_tracker, self.SR, verbose=False, seed=1789 * 1989)
        self.assertAlmostEqual(bm.std(self.beam.dt) * 1e12, 3.92, 2)


if __name__ == "__main__":
    unittest.main()
