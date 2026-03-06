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
    mu_plus,
)
from blond.generals.distributed.distributed_array import DistributedArray
from blond.physics.feedbacks.cavity_feedback import (
    IQCavityFeedback,
    IQCavityFeedbackTimingClass,
)

DEBUG_PLOTTING = False


class IQFDBKTester(IQCavityFeedback):
    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def circuit_track(self, no_beam: bool = False) -> None:
        pass

    def update_fb_variables(self) -> None:
        pass


class TestIQCavityFeedbackTimingClass:
    def setup_simulation(self):
        # single section
        self.profile = StaticProfile.from_cutoff(0, 1e-9, 5e9)
        self.rf_station = SingleHarmonicRFStation(
            phi_rf=0.0, harmonic=3, voltage=5e6
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

    test_data_discontinuity = [
        (0, 0),
        (0, 0.13),
        (0, -0.13),
        (-1, 0),
        (-1, 0.13),
        (-1, -0.13),
        (1, 0),
        (1, 0.13),
        (1, -0.13),
    ]

    @pytest.mark.parametrize(
        "phase_shift,delta_omega_factor", test_data_discontinuity
    )
    def test_for_discontinuity_single_section_no_acceleration(
        self, phase_shift: float, delta_omega_factor: float
    ) -> None:
        self.setup_simulation()
        cav_fdbk_timing = IQCavityFeedbackTimingClass(
            profile=self.profile,
        )
        self.rf_station.attach_cavity_feedback(cav_fdbk_timing)
        self.rf_station.phi_rf_design = phase_shift

        cnst_cycle = ConstantMagneticCycle(
            reference_particle=mu_plus, value=63.0e9, in_unit="momentum"
        )

        sim = Simulation(self.ring, cnst_cycle)

        voltage_array = []
        time_array = []
        rf_centers_array = []

        vals_per_turn = 5000

        def callback(simulation: Simulation, beam: Beam):
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
            rf_centers_array.append(cav_fdbk_timing.rf_centers_current_turn)
            if simulation.turn_i.value == 0:
                self.rf_station.delta_omega_rf = (
                    delta_omega_factor * self.rf_station.omega_rf
                )

        n_turns_to_simulate = 10

        sim.run_simulation(
            self.beam, n_turns=n_turns_to_simulate, callbacks=(callback,)
        )

        time_array = np.array(time_array)
        voltage_array = np.array(voltage_array)
        for time_index in range(1, len(time_array)):
            rf_centers_array[time_index] += time_array[time_index - 1][-1]
            time_array[time_index] += time_array[time_index - 1][-1]

        total_time_array = time_array.flatten()
        import matplotlib.pyplot as plt

        if DEBUG_PLOTTING:
            for trn_ind in range(0, n_turns_to_simulate):
                plt.plot(
                    time_array[trn_ind], voltage_array[trn_ind], marker="o"
                )
                for _ in range(len(rf_centers_array[trn_ind])):
                    plt.axvline(
                        x=rf_centers_array[trn_ind][_],
                        marker="x",
                        color="green",
                    )
                if trn_ind != 0:
                    plt.axvline(
                        x=total_time_array[int(trn_ind * vals_per_turn)],
                        color="red",
                        ls="--",
                    )

            plt.show()

        for ind in range(1, len(voltage_array) - 1):
            np.testing.assert_allclose(
                voltage_array[ind - 1][-1] + 3, voltage_array[ind][0] + 3
            )  # +3 to be robust against zero-relative problems
        assert all(np.diff(voltage_array.flatten()) < 0.04)

        pass
