import unittest

import matplotlib.pyplot as plt
import numpy as np

from blond3 import (
    Beam,
    Ring,
    Simulation,
    proton,
    ConstantMagneticCycle,
    StaticProfile,
    SingleHarmonicCavity,
    DriftSimple,
    WakeField,
)
from blond3.physics.impedances.sources import Resonators
from blond3.physics.impedances.solvers import PeriodicFreqSolver, TimeDomainSolver

DEV_PLOT = False


class TestResonatorImpedances(unittest.TestCase):
    def setUp(self):
        from blond3._core.backends.backend import backend, Numpy64Bit

        backend.change_backend(Numpy64Bit)

    def tearDown(self):
        from blond3._core.backends.backend import backend, Numpy32Bit

        backend.change_backend(Numpy32Bit)

    def test_equal(self):
        voltages = {}
        for i, solver in enumerate(
            (
                PeriodicFreqSolver(
                    t_periodicity=960.0,
                    allow_next_fast_len=False,
                ),
                TimeDomainSolver(),
            )
        ):
            ring = Ring(
                circumference=6911.56,
            )
            profile = StaticProfile(
                cut_left=0,
                cut_right=1 * 96,
                n_bins=256 * 96,
            )
            cavity1 = SingleHarmonicCavity()
            cavity1.voltage = 0
            cavity1.phi_rf = 0
            cavity1.harmonic = 1
            drift = DriftSimple(
                orbit_length=ring.circumference,
            )
            drift.transition_gamma = 1
            resonators = Resonators(
                shunt_impedances=100 * np.ones(1),
                center_frequencies=10 * np.ones(1),
                quality_factors=100 * np.ones(1),
            )
            np.random.seed(1)
            distr = np.random.randn(10000, 2)

            beam = Beam(
                n_particles=1e10,
                particle_type=proton,
            )
            beam.setup_beam(dt=distr[:, 0] + 5, dE=distr[:, 1])
            profile.track(beam)
            profile._hist_y[3000:] = 0
            plt.figure(0)
            plt.subplot(2, 1, 1)
            plt.plot(
                profile.hist_x,
                profile.hist_y,
                ["-", "--", ":"][i],
            )

            wake = WakeField(
                sources=(resonators,),
                solver=solver,
                profile=profile,
            )
            ring.add_elements((profile, cavity1, drift, wake))
            magnetic_cycle = ConstantMagneticCycle(
                reference_particle=proton,
                value=25.92e9,
                in_unit="momentum",
            )
            sim = Simulation(
                ring=ring,
                magnetic_cycle=magnetic_cycle,
            )
            wake_ = np.fft.irfft(
                resonators.get_wake_impedance(
                    profile.hist_x,
                    simulation=sim,
                    beam=beam,
                    n_fft=profile.n_bins,
                )
            )
            induced_voltage = wake.calc_induced_voltage(
                beam=beam,
            )
            if DEV_PLOT:
                plt.figure(0)
                plt.subplot(2, 1, 2)
                plt.plot(
                    profile.hist_x,
                    wake_,
                    ["-", "--", ":"][i],
                )
                plt.figure(1)
                plt.plot(
                    induced_voltage * 1e9,
                    ["-", "--", ":"][i],
                    label=str(type(solver)),
                )
                # plt.plot(np.convolve(profile.hist_y, wake_))
                plt.legend()
            voltages[str(solver)] = induced_voltage
        if DEV_PLOT:
            plt.figure(0)
            plt.subplot(2, 1, 1)
            plt.xlim(0, 96)
            plt.subplot(2, 1, 2)
            plt.xlim(0, 96)
            plt.figure(1)
            plt.xlim(0, 96)
            plt.show()
        for i, solver in enumerate(voltages.keys()):
            if i == 0:
                reference = voltages[solver]  # arbitrary choice
            else:
                np.testing.assert_allclose(
                    reference * 1e9,
                    voltages[solver] * 1e9,
                    atol=0.03,  # because get wake and get impedance use two
                    # different formulas, the results differ more than only
                    # numerical noise.
                    # This is because the frequency domain is cut off
                    # instead of using all frequencies/impedances,
                    # that would clip to the lower frequency region.
                )
