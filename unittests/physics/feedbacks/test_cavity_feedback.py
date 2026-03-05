import unittest
from unittest.mock import Mock

import numpy as np
import pytest

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    mu_plus,
)
from blond.generals.distributed.distributed_array import DistributedArray
from blond.physics.feedbacks.cavity_feedback import (
    IQCavityFeedback,
    IQCavityFeedbackTimingClass,
)
from blond.physics.impedances.solvers import (
    SingleTurnResonatorConvolutionSolver,
)


class IQFDBKTester(IQCavityFeedback):
    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def circuit_track(self, no_beam: bool = False) -> None:
        pass

    def update_fb_variables(self) -> None:
        pass


@pytest.mark.skip
class wtftest(unittest.TestCase):
    def setUp(self) -> None:
        # self.profile = StaticProfile.from_cutoff(0, 1e-9, 5e9)
        self.profile = Mock(StaticProfile)
        self.rf_station = SingleHarmonicRFStation(
            phi_rf=0,
            harmonic=5,
            voltage=5e6,
            local_wakefield=WakeField(
                profile=self.profile,
                solver=SingleTurnResonatorConvolutionSolver(),
                sources=[
                    Resonators(
                        center_frequencies=1,
                        quality_factors=1,
                        shunt_impedances=1,
                    )
                ],
            ),
        )
        circumference = 5
        drift = DriftSimple(circumference, momentum_compaction_factor=0)
        self.ring = Ring(
            circumference=circumference, check_section_indices=False
        )
        self.ring.add_elements([self.rf_station, drift])

        self.beam = Beam(
            intensity=1, particle_type=mu_plus, is_counter_rotating=False
        )
        self.beam._dt = DistributedArray(np.zeros(5))
        self.beam._dE = DistributedArray(np.zeros(5))
        self.beam._ids = DistributedArray(np.arange(5))
        self.beam._flags = DistributedArray(np.zeros(5))

        cnst_cycle = ConstantMagneticCycle(
            reference_particle=mu_plus, value=63.0e9, in_unit="momentum"
        )

        sim = Simulation(self.ring, cnst_cycle)

        sim.run_simulation(self.beam, n_turns=5)

    def test__init__(self) -> None:
        pass


class IQCavityFeedbackTimingClassTest(unittest.TestCase):
    def setUp(self):
        # single section
        self.profile = StaticProfile.from_cutoff(0, 1e-9, 5e9)
        # self.profile = Mock(spec=StaticProfile)
        self.rf_station = SingleHarmonicRFStation(
            phi_rf=0, harmonic=3, voltage=5e6
        )
        circumference = 5
        drift = DriftSimple(circumference, momentum_compaction_factor=0)
        self.ring = Ring(
            circumference=circumference, check_section_indices=False
        )
        self.ring.add_elements([self.rf_station, drift])

        self.beam = Beam(
            intensity=1, particle_type=mu_plus, is_counter_rotating=False
        )
        self.beam._dt = DistributedArray(np.zeros(5))
        self.beam._dE = DistributedArray(np.zeros(5))
        self.beam._ids = DistributedArray(np.arange(5))
        self.beam._flags = DistributedArray(np.zeros(5))

    def test_for_discontinuity(self) -> None:
        import logging

        logging.basicConfig(level=logging.DEBUG)
        cav_fdbk_timing = IQCavityFeedbackTimingClass(
            profile=self.profile,
        )
        self.rf_station.attach_cavity_feedback(cav_fdbk_timing)

        cnst_cycle = ConstantMagneticCycle(
            reference_particle=mu_plus, value=63.0e9, in_unit="momentum"
        )

        sim = Simulation(self.ring, cnst_cycle)

        voltage_array = []
        time_array = []

        vals_per_turn = 500
        n_turns_to_simulate = 5

        def callback(sim, beam):
            time_array.append(
                np.linspace(
                    0,
                    2
                    * np.pi
                    / self.rf_station.omega_rf_design
                    * self.rf_station.harmonic,
                    num=vals_per_turn,
                )
            )

            voltage_array.append(
                cav_fdbk_timing.get_rf_waveform_for_current_turn(
                    time_array[-1]
                )
            )

        sim.run_simulation(
            self.beam, n_turns=n_turns_to_simulate, callbacks=[callback]
        )

        time_array = np.array(time_array)
        for time_index in range(1, len(time_array)):
            time_array[time_index] += time_array[time_index - 1][-1]
        time_array = time_array.flatten()
        voltage_array = np.array(voltage_array).flatten()

        import matplotlib.pyplot as plt

        plt.plot(time_array, voltage_array)

        for trn_ind in range(1, n_turns_to_simulate):
            plt.axvline(
                x=time_array[int(trn_ind * vals_per_turn)],
                color="red",
                ls="--",
            )

        plt.show()

        assert all(np.diff(voltage_array) < 0.04)

        pass
