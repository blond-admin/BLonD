import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, m_p

from blond.beam.beam import Beam, Proton
from blond.beam.profile import CutOptions, Profile
from blond.impedances.impedance import (
    TotalInducedVoltage,
    InducedVoltageFreq,
    InducedVoltageTime,
)
from blond.impedances.impedance_sources import Resonators
from blond.impedances.induced_voltage_compact_wake_solver import (
    InducedVoltageCompactWakeMultiTurnSolver,
    ProfileRangeMultiTurnEvolution,
)
from blond.impedances.induced_voltage_compact_wake_solver import (
    InducedVoltageContainer,
)
from blond.input_parameters.ring import Ring


class TestProfileMultiTurnEvolution(unittest.TestCase):
    def setUp(self):
        self.p = ProfileRangeMultiTurnEvolution(
            np.linspace(1, 2, 25),
            np.linspace(2, 3, 25),
            np.linspace(2, 3, 25),
        )

    def test___init__(self):
        pass

    def test_max_turns(self):
        self.assertEqual(self.p.max_turns, 25)

    def test_update_profile(self):
        # SIMULATION PARAMETERS -------------------------------------------------------

        # Beam parameters
        sigma_dt = 180e-9 / 4  # [s]
        kin_beam_energy = 1.4e9  # [eV]

        E_0 = m_p * c**2 / e  # [eV]
        tot_beam_energy = E_0 + kin_beam_energy  # [eV]

        ring = Ring(
            2 * np.pi * 25,
            1 / 4.4**2,
            np.sqrt(tot_beam_energy**2 - E_0**2),
            Proton(),
            2,
        )
        np.random.seed(1)
        dt = np.random.randn(1001) * sigma_dt
        dE = np.random.randn(len(dt)) / sigma_dt
        beam = Beam(ring, len(dt), 1e11, dt=dt, dE=dE)

        # DEFINE SLICES----------------------------------------------------------------
        profile1 = Profile(
            beam,
            CutOptions(cut_left=np.min(dt), cut_right=np.max(dt), n_slices=64),
        )
        self.p.update_profile(profile=profile1, turn_i=0)
        self.assertEqual(profile1.cut_left, 1)
        self.assertEqual(profile1.cut_right, 2)

    def test_get_mutliturn_profile_limits0(self):
        starts, stops = self.p.get_mutliturn_profile_limits(turn_start=0)
        self.assertEqual(starts[0] + stops[0], 0)
        self.assertEqual(starts.shape, (25,))
        self.assertEqual(stops.shape, (25,))

    def test_get_mutliturn_profile_limits2(self):
        starts, stops = self.p.get_mutliturn_profile_limits(turn_start=2)
        self.assertAlmostEqual(starts[0] + stops[0], 0, places=12)
        self.assertEqual(starts.shape, (23,))
        self.assertEqual(stops.shape, (23,))


class TestTotalInducedVoltageNew(unittest.TestCase):
    def setUp(self):
        # SIMULATION PARAMETERS -------------------------------------------------------

        # Beam parameters
        sigma_dt = 180e-9 / 4  # [s]
        kin_beam_energy = 1.4e9  # [eV]

        E_0 = m_p * c**2 / e  # [eV]
        tot_beam_energy = E_0 + kin_beam_energy  # [eV]

        ring = Ring(
            2 * np.pi * 25,
            1 / 4.4**2,
            np.sqrt(tot_beam_energy**2 - E_0**2),
            Proton(),
            20,
        )
        np.random.seed(1)
        dt = np.random.randn(1001) * sigma_dt
        dE = np.random.randn(len(dt)) / sigma_dt
        beam = Beam(ring, len(dt), 1e11, dt=dt, dE=dE)
        profile1 = Profile(
            beam,
            CutOptions(cut_left=np.min(dt), cut_right=np.max(dt), n_slices=64),
        )
        profile1.track()

        """steps = InductiveImpedance(
            beam,
            profile1,
            34.6669349520904 / 10e9 * ring.f_rev,
            rf_station,
            deriv_mode="diff",
        )"""

        # direct space charge
        resonators = Resonators([4.5e6], [200.222e6], [200])
        t_wake = 16 * (profile1.cut_right - profile1.cut_left)
        freqsolver = InducedVoltageFreq(
            beam=beam,
            profile=profile1,
            impedance_source_list=[resonators],
            frequency_resolution=float(1 / t_wake),
        )
        timesolver = InducedVoltageTime(
            beam=beam,
            profile=profile1,
            wake_source_list=[resonators],
            wake_length=t_wake,
        )
        induced_voltage_container = InducedVoltageContainer()
        induced_voltage_container.add_induced_voltage(freqsolver)
        # induced_voltage_container.add_induced_voltage(steps)

        profile_evolution1 = ProfileRangeMultiTurnEvolution(
            starts=profile1.cut_left * np.ones(ring.n_turns + 1),
            stops=profile1.cut_right * np.ones(ring.n_turns + 1),
            t_revs=ring.t_rev,
        )
        self.total_induced_voltage_NEW = InducedVoltageCompactWakeMultiTurnSolver(
            beam=beam,
            profile=profile1,
            induced_voltage_container=induced_voltage_container,
            profile_evolution=profile_evolution1,
        )

        self.total_induced_voltage_ORG = TotalInducedVoltage(
            beam,
            profile1,
            [
                timesolver,
            ],
        )

    def test___init__(self):
        pass

    def test_track(self):
        self.total_induced_voltage_NEW.track()

    def test__induced_voltage_sum_single_profile(self):
        self.total_induced_voltage_NEW.induced_voltage_sum()
        self.total_induced_voltage_ORG.induced_voltage_sum()
        DEV_DEBUG = True
        if DEV_DEBUG:
            plt.subplot(4, 1, 1)
            plt.plot(
                self.total_induced_voltage_NEW.profile.n_macroparticles,
                "-x",
            )
            plt.subplot(4, 1, 2)
            plt.plot(self.total_induced_voltage_NEW._compressed_wake_kernel)
            plt.axvline(64)
            plt.subplot(4, 1, 3)
            plt.plot(
                self.total_induced_voltage_NEW.induced_voltage[1:],
                label="TotalInducedVoltageNew",
            )  # NOQA
            # `wake` is hidden variable of `_induced_voltage_sum`
            # plt.twinx()
            plt.plot(
                self.total_induced_voltage_ORG.induced_voltage[1:],
                "--",
                label="total_induced_voltage_ORG",
            )
            plt.legend(loc="upper left")

            plt.subplot(4, 1, 4)
            plt.plot(
                self.total_induced_voltage_NEW.induced_voltage[1:]
                - self.total_induced_voltage_ORG.induced_voltage[1:],
                label="TotalInducedVoltageNew",
            )
            """compressed_wake = fftconvolve(
                self.total_induced_voltage_NEW._compressed_wake_kernel[:],
                self.total_induced_voltage_NEW._profile_container._profiles[0].n_macroparticles[:],
                mode="same",
            )
            plt.legend(loc="upper right")
            #plt.twinx()
            plt.plot(compressed_wake[:],
                     "--",
                     label="fftconvolve",
                     )"""
            plt.legend(loc="upper left")
            plt.show()
        np.testing.assert_allclose(
            self.total_induced_voltage_NEW.induced_voltage[1:],
            self.total_induced_voltage_ORG.induced_voltage[1:],
            atol=1e-12,
        )

    def test__induced_voltage_sum_single_profile_tracking(self):
        self.total_induced_voltage_NEW.track()
        l1 = len(self.total_induced_voltage_NEW._compressed_wake_kernel)
        self.total_induced_voltage_NEW.track()
        self.total_induced_voltage_NEW.track()
        l2 = len(self.total_induced_voltage_NEW._compressed_wake_kernel)

        self.assertEqual(self.total_induced_voltage_NEW._turn_i, 3)
        self.assertLess(l2, l1)


if __name__ == "__main__":
    unittest.main()
