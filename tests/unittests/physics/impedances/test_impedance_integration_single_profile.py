import unittest

import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.constants import c

from blond import (
    Beam,
    DriftSimple,
    InductiveImpedance,
    InductiveImpedanceSolver,
    MagneticCyclePerTurn,
    Numpy64Bit,
    PeriodicFreqSolver,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    TimeDomainFftSolver,
    WakeField,
    backend,
    momentum_compaction_factor,
    proton,
)
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.generals.formatting_ import si_format
from blond.handle_results.helpers import callers_relative_path
from blond.physics.impedances.solvers import (
    MultiPassResonatorSolver,
    SingleTurnResonatorConvolutionSolver,
)
from blond.physics.impedances.sources import (
    TravelingWaveCavity,
)
from blond.testing.backend_testing import multi_backend_testcase

DEV_DRAW = False


# The pinned voltages agree with the analytic results (direct convolution
# with the textbook wake, or -L dI/dt) to machine precision. Solvers that
# use the wake, or the exact inductive kick, reproduce them to ~1e-14.
# `PeriodicFreqSolver` samples the analytic Z(f) instead of transforming
# the sampled wake, which leaves a deterministic discretisation error of
# ~5e-4 at 128 bins (shrinking ~3x per doubling of n_bins).
TOLERANCE_EXACT = 1e-9
TOLERANCE_PERIODIC_FREQ = 1e-3


def tolerance_of(solver) -> float:
    if isinstance(solver, PeriodicFreqSolver):
        return TOLERANCE_PERIODIC_FREQ
    return TOLERANCE_EXACT


def plot_diff_if_failed(
    actual, expected, tolerance: float, save: str | None = None
):
    """Assert `actual` matches `expected` within `tolerance` of its peak."""
    actual = copy_to_cpu(actual)
    if save is not None:
        np.savetxt(
            save,
            actual,
            delimiter=",",
        )
        return
    try:
        # Absolute tolerance relative to the peak: a plain `rtol` would be
        # ill-defined at the zero crossings of the induced voltage.
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0,
            atol=tolerance * np.max(np.abs(expected)),
        )
    except Exception as exc:
        if DEV_DRAW:
            plt.plot(expected, "o")
            plt.plot(actual, "x")
            plt.show()
        raise exc


class TestWakeFields(unittest.TestCase):
    def init_simulation(self, source, solver):
        beam1 = Beam.simple_gaussian(
            dt_scale=0.4e-9 / 4,
            dE_scale=1e9 / 4,
            seed=1,
            n_macroparticles=1e6,
            intensity=1e9,
            particle_type=proton,
        )

        ring = Ring(circumference=26658.883)
        rf_station = SingleHarmonicRFStation()

        rf_station.harmonic = 35640
        rf_station.voltage = 6e6
        rf_station.phi_rf_design = 0

        N_TURNS = 10
        energy_cycle = MagneticCyclePerTurn(
            value_init=450e9,
            values_after_turn=np.linspace(450e9, 450e9, N_TURNS),
            reference_particle=proton,
            in_unit="momentum",
        )

        drift1 = DriftSimple(
            orbit_length=26658.883,
        )
        drift1.momentum_compaction_factor = momentum_compaction_factor(
            transition_gamma=55.759505
        )
        profile = StaticProfile(
            cut_left=2 * beam1.dt_min, cut_right=2 * beam1.dt_max, n_bins=128
        )
        print(f"{si_format(profile.cutoff_frequency)=}")

        wake_field = WakeField(
            sources=(source,), solver=solver, profile=profile
        )
        self.beam = beam1
        self.wake_field = wake_field
        self.profile = profile

        simulation = Simulation.from_locals(locals())
        self.simulation = simulation

    @multi_backend_testcase
    def test_source_InductiveImpedance(self):
        markers = ["s", "o", "1", "2", "3", "."]
        solvers = (
            InductiveImpedanceSolver(),
            # The default periodicity (one LHC turn) means a ~5e6-point FFT;
            # this wake is short, so 2 us gives identical results ~7x faster
            PeriodicFreqSolver(t_periodicity=2e-6),
            TimeDomainFftSolver(),
            # SingleTurnResonatorConvolutionSolver(), # not applicable with `InductiveImpedance`
            # MultiPassResonatorSolver(), # TravelingWaveCavitySolver not applicable with `InductiveImpedance`
            # ContinuousMultiTurnTimeDomainSolver(n_turns=2),
            # not applicable with `short profile`
        )
        filepath = callers_relative_path(
            "resources/test_impedance_integration_single_profile"
            "/test_source_InductiveImpedance.csv",
            stacklevel=1,
        )
        induced_voltage_pinned = np.loadtxt(
            filepath,
            delimiter=",",
        )

        for i, solver in enumerate(solvers):
            # Deliberately large Z/n (realistic LHC: ~0.1 Ohm) to get a
            # strong signal; the response is linear, so only the scale
            # changes.
            source = InductiveImpedance(1234)
            self.init_simulation(source=source, solver=solver)
            self.simulation.run_simulation(self.beam, n_turns=1)
            if DEV_DRAW:
                plt.plot(
                    self.wake_field.induced_voltage,
                    markers[i],
                    label=type(solver).__name__,
                )
            plot_diff_if_failed(
                self.wake_field.induced_voltage,
                induced_voltage_pinned,
                tolerance=tolerance_of(solver),
                # save=filepath, # use to overwrite pinned values
            )
        if DEV_DRAW:
            plt.legend()
            plt.show()

    @multi_backend_testcase
    def test_source_TravelingWaveCavity(self):
        makers = ["s", "o", "1", "2", "3", "."]
        solvers = (
            # Short wake: 2 us periodicity instead of one LHC turn keeps
            # the FFT small (identical results, ~7x faster)
            PeriodicFreqSolver(t_periodicity=2e-6),
            TimeDomainFftSolver(),
            # SingleTurnResonatorConvolutionSolver(), # not applicable with `TravelingWaveCavity`
            # MultiPassResonatorSolver(), # not applicable with `TravelingWaveCavity`
            # ContinuousMultiTurnTimeDomainSolver(n_turns=2),
            # not applicable with `short profile`
        )
        filepath = callers_relative_path(
            "resources/test_impedance_integration_single_profile"
            "/test_source_TravelingWaveCavity.csv",
            stacklevel=1,
        )
        induced_voltage_pinned = np.loadtxt(
            filepath,
            delimiter=",",
        )
        for i, solver in enumerate(solvers):
            source = TravelingWaveCavity(
                R_S=400,
                frequency_R=200.222e6,
                a_factor=62e-9 * 2 * np.pi,
            )
            self.init_simulation(source=source, solver=solver)
            self.simulation.run_simulation(self.beam, n_turns=1)
            if DEV_DRAW:
                plt.figure(0)
                plt.plot(
                    self.wake_field.induced_voltage,
                    makers[i],
                    label=type(solver).__name__,
                )
                if isinstance(solver, PeriodicFreqSolver):
                    plt.figure(1)
                    solver._plot_debug_internal_state()
            plot_diff_if_failed(
                self.wake_field.induced_voltage,
                induced_voltage_pinned,
                tolerance=tolerance_of(solver),
                # save=filepath, # use to overwrite pinned values
            )
        if DEV_DRAW:
            plt.legend()
            plt.show()

    @multi_backend_testcase
    def test_source_Resonators(self):
        makers = ["s", "o", "1", "2", "3", "."]
        solvers = (
            # InductiveImpedanceSolver(),
            PeriodicFreqSolver(t_periodicity=26658.883 / c),
            TimeDomainFftSolver(),
            SingleTurnResonatorConvolutionSolver(),
            MultiPassResonatorSolver(),
            # ContinuousMultiTurnTimeDomainSolver(n_turns=2),
            # not applicable with `short profile`
        )
        filepath = callers_relative_path(
            "resources/test_impedance_integration_single_profile"
            "/test_source_Resonators.csv",
            stacklevel=1,
        )
        induced_voltage_pinned = np.loadtxt(
            filepath,
            delimiter=",",
        )
        for i, solver in enumerate(solvers):
            source = Resonators(
                # Generic narrow-band test resonator at the LHC RF
                # frequency (R/Q ~ 440 Ohm, not an LHC cavity model).
                # Ringing time Q/(pi f_r) ~ 3.6 us << T_rev, so the
                # wake from previous turns is negligible here.
                shunt_impedances=2.0e6,
                center_frequencies=400.79e6,
                quality_factors=4.5e3,
            )
            self.init_simulation(source=source, solver=solver)
            self.simulation.run_simulation(self.beam, n_turns=1)
            if DEV_DRAW:
                plt.subplot(2, 1, 1)
                plt.plot(
                    self.profile.hist_x,
                    self.profile.hist_y,
                )
                plt.subplot(2, 1, 2)
                plt.plot(
                    self.profile.hist_x,
                    self.wake_field.induced_voltage,
                    makers[i],
                    label=type(solver).__name__,
                )
            plot_diff_if_failed(
                self.wake_field.induced_voltage,
                induced_voltage_pinned,
                tolerance=tolerance_of(solver),
                # save=filepath, # use to overwrite pinned values
            )
        if DEV_DRAW:
            plt.legend()
            plt.show()


if __name__ == "__main__":
    unittest.main()
