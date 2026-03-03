import unittest

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import c

from blond import (
    Beam,
    DriftSimple,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    momentum_compaction_factor,
    proton,
)
from blond.physics.impedances.solvers import (
    InductiveImpedanceSolver,
    MultiPassResonatorSolver,
    PeriodicFreqSolver,
    SingleTurnResonatorConvolutionSolver,
    TimeDomainFftSolver,
)
from blond.physics.impedances.sources import (
    InductiveImpedance,
    Resonators,
    TravelingWaveCavity,
)


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
            cut_left=2 * beam1.dt_min, cut_right=2 * beam1.dt_max, n_bins=1024
        )
        print(f"{profile.cutoff_frequency}=")

        wake_field = WakeField(
            sources=(source,), solver=solver, profile=profile
        )
        self.beam = beam1
        self.wake_field = wake_field
        self.profile = profile

        simulation = Simulation.from_locals(locals())
        simulation.print_one_turn_execution_order()
        self.simulation = simulation

    def test_source_InductiveImpedance(self):
        makers = ["s", "o", "1", "2", "3", "."]
        solvers = (
            InductiveImpedanceSolver(),
            PeriodicFreqSolver(),
            TimeDomainFftSolver(),
            # SingleTurnResonatorConvolutionSolver(), # not applicable with `InductiveImpedance`
            # MultiPassResonatorSolver(), # nTravelingWaveCavitySolver not applicable with `InductiveImpedance`
            # ContinuousMultiTurnTimeDomainSolver(n_turns=2),
            # not applicable with `short profile`
        )

        for i, solver in enumerate(solvers):
            source = InductiveImpedance(1234)
            self.init_simulation(source=source, solver=solver)
            self.simulation.run_simulation(self.beam, n_turns=1)
            plt.plot(
                self.wake_field.induced_voltage,
                makers[i],
                label=type(solver).__name__,
            )
        plt.legend()
        plt.show()

    def test_source_TravelingWaveCavity(self):
        makers = ["s", "o", "1", "2", "3", "."]
        solvers = (
            PeriodicFreqSolver(),
            TimeDomainFftSolver(),
            # SingleTurnResonatorConvolutionSolver(), # not applicable with `TravelingWaveCavity`
            # MultiPassResonatorSolver(), # not applicable with `TravelingWaveCavity`
            # ContinuousMultiTurnTimeDomainSolver(n_turns=2),
            # not applicable with `short profile`
        )

        for i, solver in enumerate(solvers):
            source = TravelingWaveCavity(
                R_S=0.876e6,
                frequency_R=200.222e6,
                a_factor=3.899,
            )
            self.init_simulation(source=source, solver=solver)
            self.simulation.run_simulation(self.beam, n_turns=1)
            plt.figure(0)
            plt.plot(
                self.wake_field.induced_voltage,
                makers[i],
                label=type(solver).__name__,
            )
            if isinstance(solver, PeriodicFreqSolver):
                plt.figure(1)
                solver._plot_debug_internal_state()

        plt.legend()
        plt.show()

    def test_debug_source_Resonators(self):
        makers = ["s", "o", "1", "2", "3", "."]
        i = 0
        solver = PeriodicFreqSolver(t_periodicity=26658.883 / c)
        source = Resonators(
            shunt_impedances=2.0e6,  # ~ (R/Q)*Q_L for LHC cavity
            center_frequencies=400.79e6,  # LHC RF frequency
            quality_factors=4.5e3,  # Loaded Q
        )
        self.init_simulation(source=source, solver=solver)
        self.simulation.run_simulation(self.beam, n_turns=1)
        plt.figure()
        solver._plot_debug_internal_state()
        plt.figure()
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
        plt.legend()
        plt.show()

    def test_source_Resonators(self):
        makers = ["s", "o", "1", "2", "3", "."]
        solvers = (
            # InductiveImpedanceSolver(),
            PeriodicFreqSolver(t_periodicity=26658.883 / c),
            TimeDomainFftSolver(),
            SingleTurnResonatorConvolutionSolver(),  # not applicable with `InductiveImpedance`
            MultiPassResonatorSolver(),  # not applicable with
            # `InductiveImpedance`
            # ContinuousMultiTurnTimeDomainSolver(n_turns=2),
            # not applicable with `short profile`
        )

        for i, solver in enumerate(solvers):
            source = Resonators(
                shunt_impedances=2.0e6,  # ~ (R/Q)*Q_L for LHC cavity
                center_frequencies=400.79e6,  # LHC RF frequency
                quality_factors=4.5e3,  # Loaded Q
            )
            self.init_simulation(source=source, solver=solver)
            self.simulation.run_simulation(self.beam, n_turns=1)
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
        plt.legend()
        plt.show()
