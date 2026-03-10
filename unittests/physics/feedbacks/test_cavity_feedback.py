from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytest

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    MagneticCyclePerTurn,
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
            phi_rf=0.0, harmonic=self.harmonic, voltage=5e6
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
        (0, 0, 1),
        (0, 0.13, 1),
        (0, -0.13, 1),
        (-1, 0, 1),
        (-1, 0.13, 1),
        (-1, -0.13, 1),
        (1, 0, 1),
        (1, 0.13, 1),
        (1, -0.13, 1),
        (0, 0, 2),
        (0, 0.13, 2),
        (0, -0.13, 2),
        (-1, 0, 2),
        (-1, 0.13, 2),
        (-1, -0.13, 2),
        (1, 0, 2),
        (1, 0.13, 2),
        (1, -0.13, 2),
    ]

    # @pytest.mark.skip
    @pytest.mark.parametrize(
        "phase_shift,delta_omega_factor,n_rf_points", test_data_discontinuity
    )
    def test_for_discontinuity_distances_single_section_no_acceleration(
        self, phase_shift: float, delta_omega_factor: float, n_rf_points: int
    ) -> None:
        self.harmonic = 5
        self.setup_simulation()
        cav_fdbk_timing = IQCavityFeedbackTimingClass(
            profile=self.profile, n_rf_periods_per_coarse_grid=n_rf_points
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
        self.t_rf_init = 0

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
                np.sin(
                    cav_fdbk_timing._parent_rf_station.omega_rf
                    * time_array[-1]
                    + cav_fdbk_timing._parent_rf_station.phi_rf
                )
            )
            rf_centers_array.append(cav_fdbk_timing.rf_centers_current_turn)
            if simulation.turn_i.value == 0:
                self.t_rf_init = 2 * np.pi / self.rf_station.omega_rf_design
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

        if DEBUG_PLOTTING:
            for trn_ind in range(0, n_turns_to_simulate):
                plt.plot(
                    time_array[trn_ind], voltage_array[trn_ind], marker="o"
                )
                for _ in range(len(rf_centers_array[trn_ind])):
                    plt.axvline(
                        x=rf_centers_array[trn_ind][_],
                        marker="x",
                        color="green" if trn_ind == 0 else "black",
                    )
                if trn_ind != 0:
                    plt.axvline(
                        x=total_time_array[int(trn_ind * vals_per_turn)],
                        color="red",
                        ls="--",
                    )

            plt.show(block=True)

        # discontinutity testing
        for ind in range(1, len(voltage_array) - 1):
            np.testing.assert_allclose(
                voltage_array[ind - 1][-1] + 3, voltage_array[ind][0] + 3
            )  # +3 to be robust against zero-relative problems

        # distance testing
        timestep_end = (
            2
            * np.pi
            / self.rf_station.omega_rf
            * cav_fdbk_timing.n_rf_periods_per_coarse_grid
        )
        for ind in range(3, len(voltage_array) - 1):
            # between two turns
            assert np.isclose(
                rf_centers_array[ind][0] - rf_centers_array[ind - 1][-1],
                timestep_end,
                atol=timestep_end * 1e-7,
            ), (
                f"{rf_centers_array[ind][0] - rf_centers_array[ind - 1][-1]} , {timestep_end}"
            )

            np.testing.assert_allclose(
                np.diff(rf_centers_array[ind][1:]), timestep_end
            )

        np.testing.assert_allclose(
            np.diff(rf_centers_array[0]),
            self.t_rf_init * cav_fdbk_timing.n_rf_periods_per_coarse_grid,
        )

    # @pytest.mark.skip
    @pytest.mark.parametrize(
        "phase_shift,delta_omega_factor,n_rf_points", test_data_discontinuity
    )
    def test_for_discontinuity_distances_single_section_acceleration(
        self, phase_shift: float, delta_omega_factor: float, n_rf_points: int
    ) -> None:
        self.harmonic = 5
        self.setup_simulation()
        cav_fdbk_timing = IQCavityFeedbackTimingClass(
            profile=self.profile, n_rf_periods_per_coarse_grid=n_rf_points
        )
        self.rf_station.attach_cavity_feedback(cav_fdbk_timing)
        self.rf_station.phi_rf_design = phase_shift

        n_turns_to_simulate = 5
        delta_E = 5e6
        inj_energy = 5e6

        cnst_cycle = MagneticCyclePerTurn(
            reference_particle=mu_plus,
            value_init=inj_energy,
            values_after_turn=inj_energy
            + np.arange(1, n_turns_to_simulate + 1) * delta_E,
            in_unit="momentum",
        )

        sim = Simulation(self.ring, cnst_cycle)

        voltage_array = []
        time_array = []
        rf_centers_array = []
        omega_rf_save = []

        vals_per_turn = 5000
        self.t_rf_init = 0

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
                np.sin(
                    cav_fdbk_timing._parent_rf_station.omega_rf
                    * time_array[-1]
                    + cav_fdbk_timing._parent_rf_station.phi_rf
                )
            )
            rf_centers_array.append(cav_fdbk_timing.rf_centers_current_turn)
            omega_rf_save.append(cav_fdbk_timing.omega_rf)
            print(cav_fdbk_timing._parent_rf_station.omega_rf_design)
            if simulation.turn_i.value == 0:
                self.t_rf_init = 2 * np.pi / self.rf_station.omega_rf_design
                self.rf_station.delta_omega_rf = (
                    delta_omega_factor * self.rf_station.omega_rf
                )

        sim.run_simulation(
            self.beam, n_turns=n_turns_to_simulate, callbacks=(callback,)
        )

        time_array = np.array(time_array)
        voltage_array = np.array(voltage_array)
        for time_index in range(1, len(time_array)):
            rf_centers_array[time_index] += time_array[time_index - 1][-1]
            time_array[time_index] += time_array[time_index - 1][-1]

        total_time_array = time_array.flatten()

        if DEBUG_PLOTTING:
            for trn_ind in range(0, n_turns_to_simulate):
                plt.plot(
                    time_array[trn_ind], voltage_array[trn_ind], marker="o"
                )
                for _ in range(len(rf_centers_array[trn_ind])):
                    plt.axvline(
                        x=rf_centers_array[trn_ind][_],
                        marker="x",
                        color="green" if trn_ind == 0 else "black",
                    )
                if trn_ind != 0:
                    plt.axvline(
                        x=total_time_array[int(trn_ind * vals_per_turn)],
                        color="red",
                        ls="--",
                    )

            plt.show(block=True)

        # discontinutity testing
        for ind in range(1, len(voltage_array) - 1):
            np.testing.assert_allclose(
                voltage_array[ind - 1][-1] + 3, voltage_array[ind][0] + 3
            )  # +3 to be robust against zero-relative problems

        # distance testing
        for ind in range(3, len(voltage_array) - 1):
            timestep = (
                2
                * np.pi
                / omega_rf_save[ind]
                * cav_fdbk_timing.n_rf_periods_per_coarse_grid
            )
            # between two turns
            assert rf_centers_array[ind][0] - rf_centers_array[ind - 1][
                -1
            ] > timestep or np.isclose(
                rf_centers_array[ind][0] - rf_centers_array[ind - 1][-1],
                timestep,
            ), (
                f"{rf_centers_array[ind][0] - rf_centers_array[ind - 1][-1]} , {timestep}"
            )
            np.testing.assert_allclose(
                np.diff(rf_centers_array[ind][1:]), timestep
            )

        np.testing.assert_allclose(
            np.diff(rf_centers_array[0]),
            self.t_rf_init * cav_fdbk_timing.n_rf_periods_per_coarse_grid,
        )

        pass

    @pytest.mark.parametrize("n_sections", [1, 4, 20])
    def test_get_slice_of_elements_this_section_cnst_cycle(
        self, n_sections: int
    ):
        self.harmonic = 5
        self.setup_simulation()

        n_sections = 4
        circumference = 20
        n_turns_to_simulate = 5

        ring = Ring(circumference=circumference, check_section_indices=False)
        n_rf_points = 1
        element_list = []
        timing_fdbk_list = []
        for section in range(n_sections):
            element_list.append(
                DriftSimple(
                    momentum_compaction_factor=5,
                    orbit_length=circumference / n_sections / 2,
                    section_index=section,
                )
            )
            timing_fdbk_list.append(
                IQCavityFeedbackTimingClass(
                    profile=self.profile,
                    n_rf_periods_per_coarse_grid=n_rf_points,
                    debug=True,
                )
            )
            rf_station = SingleHarmonicRFStation(
                phi_rf=0.0,
                harmonic=self.harmonic,
                voltage=5e6,
                section_index=section,
            )
            rf_station.attach_cavity_feedback(timing_fdbk_list[-1])
            element_list.append(rf_station)
            element_list.append(
                DriftSimple(
                    momentum_compaction_factor=5,
                    orbit_length=circumference / n_sections / 2,
                    section_index=section,
                )
            )
            element_list.append(
                StaticProfile.from_cutoff(0, 1e-9, 5e9, section_index=section)
            )
            element_list.append(
                WakeField(
                    section_index=section,
                    sources=(
                        Resonators(
                            center_frequencies=1.0,
                            shunt_impedances=1.0,
                            quality_factors=1.0,
                        ),
                    ),
                    solver=SingleTurnResonatorConvolutionSolver(),
                )
            )
        ring.add_elements(element_list)

        cnst_cycle = ConstantMagneticCycle(
            reference_particle=mu_plus, value=63.0e9, in_unit="momentum"
        )

        sim = Simulation(
            ring,
            cnst_cycle,
        )

        def callback(simulation: Simulation, beam: Beam):
            time_passed_list = []
            omega_list = []
            for fdbk in timing_fdbk_list:
                fdbk: IQCavityFeedbackTimingClass
                if (
                    fdbk._parent_rf_station
                    not in fdbk.current_slice_elements_forward
                ):
                    pytest.fail(
                        f"parent rf station not in current_slice element list in turn {simulation.turn_i.value} section {fdbk.section_index}"
                    )
                if len(fdbk.current_slice_elements_forward) != 3:
                    pytest.fail(
                        f"{len(fdbk.current_slice_elements_forward)} != 3 in turn {simulation.turn_i.value} section {fdbk.section_index}"
                    )
                time_passed_list.append(fdbk.passed_time_forward)
                omega_list.append(fdbk.omega_rf_design_forward)

                assert (
                    fdbk.tracked_forward_until_element
                    not in fdbk.current_slice_elements_forward
                )
                assert (
                    fdbk.tracked_forward_until_element
                    is fdbk.reference_altering_elements[
                        (fdbk.own_index_in_reference_list + 3)
                        % len(fdbk.reference_altering_elements)
                    ]
                )

            np.testing.assert_allclose(
                time_passed_list, time_passed_list[0]
            )  # with no acceleration, this has to be true
            np.testing.assert_allclose(
                omega_list, omega_list[0]
            )  # with no acceleration, this has to be true

        sim.run_simulation(
            self.beam, callbacks=(callback,), n_turns=n_turns_to_simulate
        )
