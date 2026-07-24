import unittest

import matplotlib.pyplot as plt
import numpy as np
import pytest

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    momentum_compaction_factor,
    proton,
)
from blond.physics.impedances.solvers import (
    PeriodicFreqSolver,
    TimeDomainFftSolver,
)
from blond.physics.impedances.sources import Resonators

DEV_PLOT = False


class TestResonatorImpedances(unittest.TestCase):
    def setUp(self):
        from blond.core.backends.backend import Numpy64Bit, backend

        backend.change_backend(Numpy64Bit)

    def tearDown(self):
        from blond.core.backends.backend import Numpy64Bit, backend

        backend.change_backend(Numpy64Bit)

    @pytest.mark.backend_mutation
    def test_equal(self):
        voltages = {}
        for i, solver in enumerate(
            (
                PeriodicFreqSolver(
                    t_periodicity=960.0,
                    allow_next_fast_len=False,
                ),
                TimeDomainFftSolver(),
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
            cavity1 = SingleHarmonicRFStation()
            cavity1.voltage = 0
            cavity1.phi_rf_design = 0
            cavity1.harmonic = 1
            drift = DriftSimple(
                orbit_length=ring.circumference,
            )
            drift.momentum_compaction_factor = momentum_compaction_factor(1)
            resonators = Resonators(
                shunt_impedances=100 * np.ones(1),
                center_frequencies=10 * np.ones(1),
                quality_factors=100 * np.ones(1),
            )
            np.random.seed(1)
            distr = np.random.randn(10000, 2)

            beam = Beam(
                intensity=1e10,
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
                resonators.get_impedance_from_wake(
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

    @pytest.mark.backend_mutation
    def test_low_q_resonator_time_matches_freq(self):
        """Under-resolved low-Q resonator: time solver must match freq solver.

        Regression for the InducedVoltageTime vs InducedVoltageFreq
        discrepancy. The resonator sits at 20x the RF harmonic, so its wake
        oscillates several times within a handful of profile bins. Point-
        sampling the wake aliased badly (~230% error) and silently gave a
        wrong induced voltage; bin-integrating the wake makes the two solvers
        agree, and the residual shrinks as the profile grid is refined.
        """
        from blond.core.backends.backend import Numpy64Bit, backend

        backend.change_backend(Numpy64Bit)

        def induced_voltage(solver, n_bins):
            circumference = 2 * np.pi * 4242.89
            ring = Ring(circumference=circumference)
            cycle = ConstantMagneticCycle(
                reference_particle=proton, value=6800e9, in_unit="momentum"
            )
            rf = SingleHarmonicRFStation()
            rf.voltage = 6e6
            rf.phi_rf_design = 0
            rf.harmonic = 35640
            drift = DriftSimple(orbit_length=circumference)
            drift.momentum_compaction_factor = momentum_compaction_factor(
                55.759505
            )
            t_rev = cycle.get_t_rev_init(circumference=circumference)
            t_rf = float(t_rev / 35640)
            f_res = 20.0 / t_rf
            r_shunt = 0.7 * 1.0 * f_res * t_rev  # Z/n = 0.7 Ohm, Q = 1
            resonators = Resonators(r_shunt, f_res, 1.0)

            profile = StaticProfile(
                cut_left=0, cut_right=2 * t_rf, n_bins=n_bins
            )
            # deterministic Gaussian histogram (no Monte-Carlo noise, so the
            # solver difference is purely the wake representation)
            sigma = 0.83162555241781e-9 / 2
            centre = 2.5e-9
            x = np.asarray(profile.hist_x)
            profile._hist_y = backend.array(
                np.exp(-0.5 * ((x - centre) / sigma) ** 2)
            )
            # normally set by profile.track; irrelevant here as it is identical
            # for both solvers and cancels in the relative comparison
            profile.hist_y_to_density_factor = 1.0
            beam = Beam(intensity=3e11, particle_type=proton)

            wake = WakeField(
                sources=(resonators,), solver=solver, profile=profile
            )
            ring.add_elements((profile, rf, drift, wake))
            Simulation(ring=ring, magnetic_cycle=cycle)
            return np.asarray(wake.calc_induced_voltage(beam=beam))

        def max_rel_dev(n_bins):
            v_freq = induced_voltage(PeriodicFreqSolver(), n_bins)
            v_time = induced_voltage(TimeDomainFftSolver(), n_bins)
            return np.max(np.abs(v_time - v_freq)) / np.max(np.abs(v_freq))

        dev_coarse = max_rel_dev(256)
        dev_fine = max_rel_dev(1024)

        # point-sampling gave ~2.3 here; bin-integration keeps it small
        assert dev_coarse < 0.15, dev_coarse
        # and it converges as the grid is refined
        assert dev_fine < dev_coarse
        assert dev_fine < 0.05, dev_fine
