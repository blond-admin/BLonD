from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytest

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    MagneticCyclePerTurn,
    MagneticCyclePerTurnAllRFStations,
    Numpy64Bit,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    backend,
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


def check_allclose_turn_printing_nested(
    array: list, turn: int, array_name: str
):
    check_allclose = [
        np.allclose(array_entry, array[0], rtol=1e-12, atol=0)
        for array_entry in array
    ]
    newline = "\n"
    if not all(check_allclose):
        pytest.fail(
            f"problem in turn {turn} with {array_name}:\n\n{newline.join(str(ln) for ln in array)}\n--> {check_allclose}",
        )


def check_allclose_turn_printing(
    array_1: list, array_2: list, turn: int, array_name: str
):
    check_allclose = [
        np.isclose(array_1[array_idx], array_2[array_idx], rtol=1e-12, atol=0)
        for array_idx in range(len(array_1))
    ]
    newline = "\n"
    if not all(check_allclose):
        pytest.fail(
            f"problem in idx {turn} with {array_name}:\n\n{newline.join(str(ln) for ln in array_1)}\n--> {check_allclose}",
        )


def check_fail_printing(bool_expr: bool, msg: str):
    if bool_expr:
        pytest.fail(msg)


class TestIQCavityFeedbackTimingClass:
    def setup_simulation(self):
        # single section
        self.profile = StaticProfile.from_cutoff(0, 1e-9, 5e9)
        self.rf_station = SingleHarmonicRFStation(
            phi_rf=0.0, harmonic=self.harmonic, voltage=5e6
        )
        self.circumference = 5
        drift = DriftSimple(self.circumference, momentum_compaction_factor=0)
        self.ring = Ring(
            circumference=self.circumference, check_section_indices=False
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
        # (0, 0, 1),
        (0, 0.1, 1),
        (0, -0.1, 1),
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
        (0, 0, 3),
        (0, 0.13, 3),
        (0, -0.13, 3),
        (-1, 0, 3),
        (-1, 0.13, 3),
        (-1, -0.13, 3),
        (1, 0, 3),
        (1, 0.13, 3),
        (1, -0.13, 3),
    ]

    @pytest.mark.backend_mutation
    @pytest.mark.parametrize(
        "phase_shift,delta_omega_factor,n_rf_points", test_data_discontinuity
    )
    def test_for_discontinuity_distances_single_section_no_acceleration(
        self, phase_shift: float, delta_omega_factor: float, n_rf_points: int
    ) -> None:
        backend.change_backend(Numpy64Bit)
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
            rf_centers_array.append(cav_fdbk_timing.rf_centers)
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
    @pytest.mark.backend_mutation
    @pytest.mark.parametrize(
        "phase_shift,delta_omega_factor,n_rf_points", test_data_discontinuity
    )
    def test_for_discontinuity_distances_single_section_acceleration(
        self, phase_shift: float, delta_omega_factor: float, n_rf_points: int
    ) -> None:
        backend.change_backend(Numpy64Bit)
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
            # self.rf_station.omega_rf_design = self.rf_station.calc_omega_rf_design(beam.reference.beta, ring_circumference=self.circumference)
            time_array.append(
                np.linspace(
                    0,
                    # 2 * np.pi /
                    # cav_fdbk_timing.
                    # cav_fdbk_timing._parent_rf_station.calc_main_harmonic_omega_rf_design(beam_beta=beam.reference.beta, ring_circumference=self.circumference)
                    # cav_fdbk_timing._parent_rf_station.calc_main_harmonic_t_rf(beam_beta=beam.reference.beta, ring_circumference=self.circumference)
                    cav_fdbk_timing.forward_tracking_time,
                    # * self.rf_station.harmonic,
                    num=vals_per_turn,
                )
            )
            phase_offset = (
                cav_fdbk_timing._parent_rf_station.phase_correction_frequency_offset
                if simulation.turn_i.value > 1
                else 0
            )
            voltage_array.append(
                np.sin(
                    cav_fdbk_timing.forward_tracking_omega_rf * time_array[-1]
                    + cav_fdbk_timing.phase_offset_frwrd  # - 2*phase_offset
                )
            )
            rf_centers_array.append(cav_fdbk_timing.rf_centers)
            # omega_rf_save.append(cav_fdbk_timing.forward_tracking_omega_rf)
            omega_rf_save.append(cav_fdbk_timing.forward_tracking_omega_rf)
            print(cav_fdbk_timing.omega_rf)
            print(cav_fdbk_timing.forward_tracking_omega_rf)
            print(cav_fdbk_timing._parent_rf_station.delta_omega_rf)
            print(cav_fdbk_timing.phi_rf)
            print(
                cav_fdbk_timing._parent_rf_station.phase_correction_frequency_offset
            )

            print(r"\n---------------------------\n")

            if simulation.turn_i.value == 0:
                self.t_rf_init = (
                    2 * np.pi / cav_fdbk_timing.forward_tracking_omega_rf
                )
                self.rf_station.delta_omega_rf = (
                    delta_omega_factor
                    * cav_fdbk_timing.forward_tracking_omega_rf
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

    def setup_simulation_multisection(self, n_sections, circumference):
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

        return ring, element_list, timing_fdbk_list

    @pytest.mark.backend_mutation
    @pytest.mark.parametrize("n_sections", [1, 4, 20])
    def test_get_slice_of_elements_this_section_cnst_cycle_fwrd(
        self, n_sections: int
    ):
        backend.change_backend(Numpy64Bit)
        self.harmonic = 20
        self.setup_simulation()

        n_sections = 4
        circumference = 20

        ring, element_list, timing_fdbk_list = (
            self.setup_simulation_multisection(
                circumference=circumference, n_sections=n_sections
            )
        )

        n_turns_to_simulate = 5

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
            rf_centers_list = []
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
                time_passed_list.append(fdbk.forward_tracking_time)
                omega_list.append(fdbk.forward_tracking_omega_rf)
                rf_centers_list.append(fdbk.rf_centers)

                assert (
                    fdbk.tracked_forward_until_element
                    not in fdbk.current_slice_elements_forward
                )  # this element should be tracked afterwards, not now
                assert (
                    fdbk.tracked_forward_until_element
                    is fdbk.reference_altering_elements[
                        (
                            fdbk.own_index_in_reference_list + 3
                        )  # first element is skipped, since its the cavity itself
                        % len(fdbk.reference_altering_elements)
                    ]
                )  # 2 elements between two cavities

            np.testing.assert_allclose(
                time_passed_list, time_passed_list[0]
            )  # with no acceleration, this has to be true (all time_passed are the same
            np.testing.assert_allclose(
                omega_list, omega_list[0]
            )  # with no acceleration, this has to be true (all omegas are the same)
            last_x_indices = (
                0
                if simulation.turn_i.value > 0
                else -int(self.harmonic / n_sections)
            )
            # this is necessary since the arrays have different lengths in the first turn
            # every cavity has a different amount of tracking before it gets passed by the beam
            [
                np.testing.assert_allclose(
                    rf_centers_list_entry[last_x_indices:],
                    rf_centers_list[0][last_x_indices:],
                )
                for rf_centers_list_entry in rf_centers_list
            ]  # with no acceleration, this has to be true (all RF centers are the same)

        sim.run_simulation(
            self.beam, callbacks=(callback,), n_turns=n_turns_to_simulate
        )

    @pytest.mark.backend_mutation
    @pytest.mark.parametrize("n_sections", [1, 4, 20])
    def test_get_slice_of_elements_this_section_cnst_cycle_reverse(
        self, n_sections: int
    ):
        backend.change_backend(Numpy64Bit)
        self.harmonic = 20
        self.setup_simulation()

        circumference = 20

        ring, element_list, timing_fdbk_list = (
            self.setup_simulation_multisection(
                circumference=circumference, n_sections=n_sections
            )
        )

        n_turns_to_simulate = 5

        cnst_cycle = ConstantMagneticCycle(
            reference_particle=mu_plus, value=63.0e9, in_unit="momentum"
        )

        sim = Simulation(
            ring,
            cnst_cycle,
        )

        def callback(simulation: Simulation, beam: Beam):
            if simulation.turn_i.value == 0:  # TODO: and not CR
                return
            time_passed_list = []
            omega_list = []
            rf_centers_list = []
            for idx, fdbk in enumerate(timing_fdbk_list):
                fdbk: IQCavityFeedbackTimingClass
                # TODO: check rf centers --> add calculation of rf centers
                if not n_sections == 1:  # checks only apply to multi-section
                    check_fail_printing(
                        not np.isclose(
                            fdbk.current_beam_reference_time,
                            fdbk.reference_time_after_reverse,
                            atol=0,
                            rtol=1e-12,
                        ),
                        f"reference time after reverse not within tolerance {fdbk.current_beam_reference_time}, {fdbk.reference_time_after_reverse} in turn {simulation.turn_i.value} section {fdbk.section_index}",
                    )
                    check_fail_printing(
                        not np.isclose(
                            fdbk.current_beam_reference_energy,
                            fdbk.reference_energy_after_reverse,
                            atol=0,
                            rtol=1e-12,
                        ),
                        f"reference time after reverse not within tolerance {fdbk.current_beam_reference_time}, {fdbk.reference_time_after_reverse} in turn {simulation.turn_i.value} section {fdbk.section_index}",
                    )

                    time_passed_list.append(fdbk.reverse_tracking_time_array)
                    msk = fdbk.reverse_tracking_time_array != 0
                    used_time_array = np.array(
                        fdbk.reverse_tracking_time_array
                    )[msk]
                    check_fail_printing(
                        len(used_time_array) != 1,
                        f"time arr length err {len(used_time_array)} != 1 section {idx}, trn {simulation.turn_i.value}",
                    )  # should be unified to 1 value, since only one frequency is used, regardless of number of sections (no acceleration)
                    omega_list.append(fdbk.reverse_tracking_omega_list)
                    check_fail_printing(
                        len(fdbk.reverse_tracking_omega_list)
                        != len(fdbk.reverse_tracking_time_array),
                        f"omega list not equal, {len(fdbk.reverse_tracking_omega_list)}, {len(fdbk.reverse_tracking_time_array)}, section {idx}, trn {simulation.turn_i.value}",
                    )

                # rf_centers_list.append(fdbk.rf_centers_reverse_direction)

                # assert (
                #         fdbk.tracked_forward_until_element
                #         not in fdbk.current_slice_elements_forward
                # )  # this element should be tracked afterwards, not now
                # assert (
                #         fdbk.tracked_forward_until_element
                #         is fdbk.reference_altering_elements[
                #             (fdbk.own_index_in_reference_list + 3)
                #             % len(fdbk.reference_altering_elements)
                #             ]
                # )  # 3 elements between two cavities

            check_allclose_turn_printing_nested(
                time_passed_list, simulation.turn_i.value, "time_passed"
            )  # with no acceleration, this has to be true (all time_passed are the same
            check_allclose_turn_printing_nested(
                omega_list, simulation.turn_i.value, "omega_list"
            )  # with no acceleration, this has to be true (all omegas are the same)
            # check_allclose_turn_printing(rf_centers_list, simulation.turn_i.value)  # with no acceleration, this has to be true (all RF centers are the same)

        sim.run_simulation(
            self.beam, callbacks=(callback,), n_turns=n_turns_to_simulate
        )

    @pytest.mark.backend_mutation
    @pytest.mark.parametrize("n_sections", [2])  #
    def test_get_slice_of_elements_this_section_accelerating_cycle_cycle_reverse(
        self, n_sections: int
    ):
        backend.change_backend(Numpy64Bit)
        self.harmonic = 20
        self.setup_simulation()

        # n_sections = 4
        circumference = 20

        ring, element_list, timing_fdbk_list = (
            self.setup_simulation_multisection(
                circumference=circumference, n_sections=n_sections
            )
        )

        n_turns_to_simulate = 6
        injection_energy = 5e9
        en_gain_per_turn = 20e9
        ejection_energy = (
            injection_energy + en_gain_per_turn * n_turns_to_simulate
        )

        vals_after_rf_station = np.linspace(
            injection_energy + en_gain_per_turn / n_sections,
            ejection_energy,
            num=n_sections * n_turns_to_simulate,
        )
        vals_after_rf_station = np.reshape(
            vals_after_rf_station, (n_sections, n_turns_to_simulate), order="F"
        )

        cnst_cycle = MagneticCyclePerTurnAllRFStations(
            reference_particle=mu_plus,
            value_init=injection_energy,
            values_after_rf_station_per_turn=vals_after_rf_station,
            in_unit="momentum",
        )

        sim = Simulation(
            ring,
            cnst_cycle,
        )

        time_passed_list = [
            [[] for _ in range(n_turns_to_simulate - 1)]
            for _ in range(n_sections)
        ]
        omega_list = [
            [[] for _ in range(n_turns_to_simulate - 1)]
            for _ in range(n_sections)
        ]
        rf_center_list = [
            [[] for _ in range(n_turns_to_simulate - 1)]
            for _ in range(n_sections)
        ]

        def callback(simulation: Simulation, beam: Beam):
            if simulation.turn_i.value == 0:  # TODO: and not CR
                return
            for idx, fdbk in enumerate(timing_fdbk_list):
                fdbk: IQCavityFeedbackTimingClass
                if (
                    n_sections != 1
                ):  # only relevant/only gets set on multistation
                    check_fail_printing(
                        not np.isclose(
                            fdbk.current_beam_reference_time,
                            fdbk.reference_time_after_reverse,
                            atol=0,
                            rtol=1e-12,
                        ),
                        f"reference time after reverse not within tolerance {fdbk.current_beam_reference_time}, {fdbk.reference_time_after_reverse} in turn {simulation.turn_i.value} section {fdbk.section_index}",
                    )
                    check_fail_printing(
                        not np.isclose(
                            fdbk.current_beam_reference_energy,
                            fdbk.reference_energy_after_reverse,
                            atol=0,
                            rtol=1e-12,
                        ),
                        f"reference energy after reverse not within tolerance {fdbk.current_beam_reference_energy}, {fdbk.reference_energy_after_reverse} in turn {simulation.turn_i.value} section {fdbk.section_index}",
                    )

                    msk = fdbk.reverse_tracking_time_array != 0
                    used_time_array = np.array(
                        fdbk.reverse_tracking_time_array
                    )[msk]
                    used_omega_array = np.array(
                        fdbk.reverse_tracking_omega_list
                    )[msk]
                    target_length = n_sections - 1
                    check_fail_printing(
                        len(used_time_array) != target_length,
                        f"time arr length err {len(used_time_array)} != {target_length} section {idx}, trn {simulation.turn_i.value}",
                    )  # two drifts per section, 3 sections in between cavities
                    check_fail_printing(
                        len(fdbk.reverse_tracking_omega_list)
                        != len(fdbk.reverse_tracking_time_array),
                        f"omega list not equal, {len(fdbk.reverse_tracking_omega_list)}, {len(fdbk.reverse_tracking_time_array)}, section {idx}, trn {simulation.turn_i.value}",
                    )
                    used_omega_array = np.append(
                        used_omega_array, fdbk.forward_tracking_omega_rf
                    )
                    used_time_array = np.append(
                        used_time_array, fdbk.forward_tracking_time
                    )

                    time_passed_list[idx][sim.turn_i.value - 1] = (
                        used_time_array
                    )
                    omega_list[idx][sim.turn_i.value - 1] = used_omega_array

                    assert (
                        fdbk.tracked_forward_until_element
                        not in fdbk.current_slice_elements_forward
                    )  # this element should be tracked afterwards, not now

                    rf_center_list[idx][sim.turn_i.value - 1] = fdbk.rf_centers
                    assert (
                        fdbk.tracked_forward_until_element
                        is fdbk.reference_altering_elements[
                            (fdbk.own_index_in_reference_list + 3)
                            % len(fdbk.reference_altering_elements)
                        ]
                    )  # 3 elements between two cavities

        sim.run_simulation(
            self.beam, callbacks=(callback,), n_turns=n_turns_to_simulate
        )

        # test for time_array consistency
        comp_len = 0
        continuousfdbk_time = []
        for idx, fdbk in enumerate(time_passed_list):
            current_fdbk_total_time = np.array(fdbk).flatten()
            continuousfdbk_time.append(current_fdbk_total_time)
            increaser = np.diff(current_fdbk_total_time) < 0
            assert all(increaser), (
                f"time must be decreasing, but its not: {increaser}"
            )
            if idx == 0:
                comp_len = len(current_fdbk_total_time)
            else:
                assert comp_len == len(current_fdbk_total_time)
            if idx > 0:
                np.testing.assert_allclose(
                    continuousfdbk_time[-1][:-1],
                    continuousfdbk_time[-2][1:],
                    rtol=1e-12,
                    atol=0,
                )  # shifted by one, but otherwise equal

        comp_len = 0
        continuous_omega = []
        for idx, fdbk in enumerate(omega_list):
            current_fdbk_omega_list = np.array(fdbk).flatten()
            continuous_omega.append(current_fdbk_omega_list)
            increaser = np.diff(current_fdbk_omega_list) > 0
            assert all(increaser), (
                f"omega must be increasing, but its not: {increaser}"
            )
            if idx == 0:
                comp_len = len(current_fdbk_omega_list)
            else:
                assert comp_len == len(current_fdbk_omega_list)
            if idx > 0:
                np.testing.assert_allclose(
                    continuous_omega[-1][:-1],
                    continuous_omega[-2][1:],
                    rtol=1e-12,
                    atol=0,
                )  # shifted by one, but otherwise equal

    @pytest.mark.backend_mutation
    @pytest.mark.parametrize("n_sections", [4])  # [1, 4, 20]
    def test_get_slice_of_elements_this_section_accelerating_cycle_cycle_reverse_rf_centers(
        self, n_sections: int
    ):
        backend.set_specials("cpp")
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")
        self.harmonic = 20
        self.setup_simulation()

        # n_sections = 4
        circumference = 20

        # only debugging/testing requirement, not for running
        assert circumference % n_sections == 0, (
            "simulation setup wrong, check input changes"
        )
        assert self.harmonic % n_sections == 0, (
            "simulation setup wrong, check input changes"
        )

        ring, element_list, timing_fdbk_list = (
            self.setup_simulation_multisection(
                circumference=circumference, n_sections=n_sections
            )
        )

        n_turns_to_simulate = 3
        injection_energy = 5e8
        en_gain_per_turn = 20e9
        ejection_energy = (
            injection_energy + en_gain_per_turn * n_turns_to_simulate
        )

        vals_after_rf_station = np.linspace(
            injection_energy + en_gain_per_turn / n_sections,
            ejection_energy,
            num=n_sections * n_turns_to_simulate,
        )
        vals_after_rf_station = np.append(
            np.ones(n_sections) * injection_energy, vals_after_rf_station
        )
        vals_after_rf_station = np.append(
            np.ones(n_sections) * injection_energy, vals_after_rf_station
        )
        n_turns_to_simulate += 2
        vals_after_rf_station = np.reshape(
            vals_after_rf_station, (n_sections, n_turns_to_simulate), order="F"
        )

        cnst_cycle = MagneticCyclePerTurnAllRFStations(
            reference_particle=mu_plus,
            value_init=injection_energy,
            values_after_rf_station_per_turn=vals_after_rf_station,
            in_unit="momentum",
        )

        sim = Simulation(
            ring,
            cnst_cycle,
        )

        time_passed_list = [
            [[] for _ in range(n_turns_to_simulate - 1)]
            for _ in range(n_sections)
        ]
        omega_list = [
            [[] for _ in range(n_turns_to_simulate - 1)]
            for _ in range(n_sections)
        ]
        rf_center_list = [
            [[] for _ in range(n_turns_to_simulate - 1)]
            for _ in range(n_sections)
        ]

        def callback(simulation: Simulation, beam: Beam):
            if simulation.turn_i.value == 0:  # TODO: and not CR
                for idx, fdbk in enumerate(timing_fdbk_list):
                    fdbk: IQCavityFeedbackTimingClass
                    assert (
                        len(fdbk.rf_centers) == 2 + fdbk.section_index * 5 + 5
                    )  # 5 are forward, 5 per full drift and 2 for the inital drift, which is half length
                return
            for idx, fdbk in enumerate(timing_fdbk_list):
                fdbk: IQCavityFeedbackTimingClass
                if (
                    n_sections != 1
                ):  # only relevant/only gets set on multistation
                    assert (
                        len(fdbk.rf_centers) == 20,
                        f"failed in {simulation.turn_i.value} {idx} {len(fdbk.rf_centers)}",  # 15 from reverse and 5 from frwrd
                    )
                    msk = fdbk.reverse_tracking_time_array != 0
                    used_time_array = np.array(
                        fdbk.reverse_tracking_time_array
                    )[msk]
                    used_omega_array = np.array(
                        fdbk.reverse_tracking_omega_list
                    )[msk]
                    used_omega_array = np.append(
                        used_omega_array, fdbk.forward_tracking_omega_rf
                    )
                    used_time_array = np.append(
                        used_time_array, fdbk.forward_tracking_time
                    )

                    time_passed_list[idx][sim.turn_i.value - 1] = (
                        used_time_array
                    )
                    omega_list[idx][sim.turn_i.value - 1] = used_omega_array

                    rf_center_list[idx][sim.turn_i.value - 1] = fdbk.rf_centers

        sim.run_simulation(
            self.beam, callbacks=(callback,), n_turns=n_turns_to_simulate
        )

        if DEBUG_PLOTTING:
            continuousfdbk_time = []
            for idx, fdbk in enumerate(time_passed_list):
                current_fdbk_total_time = np.array(fdbk).flatten()
                continuousfdbk_time.append(current_fdbk_total_time)

            continuousfdbk_omega = []
            for idx, fdbk in enumerate(omega_list):
                current_fdbk_omega_list = np.array(fdbk).flatten()
                continuousfdbk_omega.append(current_fdbk_omega_list)

            voltage_array = [[] for _ in range(n_sections)]
            global_time_array = [[] for _ in range(n_sections)]

            for fdbk_ind in range(len(continuousfdbk_time)):
                # plotting incorrect, should start at different positions for different feedbacks --> also in past turn --> essentially just shifting the turn marker
                for time_ind in range(len(continuousfdbk_time[fdbk_ind])):
                    start_time = np.cumsum(
                        continuousfdbk_time[fdbk_ind][:time_ind]
                    )
                    until_time = (
                        start_time + continuousfdbk_time[fdbk_ind][time_ind]
                    )
                    time_arr_local = np.linspace(
                        start_time, until_time, num=500
                    )
                    voltage_array[fdbk_ind] = np.sin(
                        time_arr_local
                        * continuousfdbk_omega[fdbk_ind][time_ind]
                    )
                    global_time_array[fdbk_ind] = time_arr_local
                plt.figure(f"Feedback section {fdbk_ind + 1}")
                plt.title(f"Feedback section {fdbk_ind + 1}")
                plt.plot(
                    global_time_array[fdbk_ind],
                    voltage_array[fdbk_ind],
                    label="continuous voltage",
                    marker="o",
                )

                for _ in range(1, n_turns_to_simulate):  # end of turns
                    plt.axvline(
                        x=np.cumsum(
                            continuousfdbk_time[fdbk_ind][
                                : int(_ * n_sections)
                            ]
                        )[-1],
                        ls="--",
                        color="red",
                    )
                offset = 0
                sec_ind = 0
                for trn_ind in range(n_turns_to_simulate - 1):  # RF centers
                    for rf_center_ind, rf_center in enumerate(
                        rf_center_list[fdbk_ind][trn_ind]
                    ):
                        if (
                            rf_center
                            <= rf_center_list[fdbk_ind][trn_ind][
                                rf_center_ind - 1
                            ]
                        ):
                            print(rf_center_ind)
                            offset += continuousfdbk_time[fdbk_ind][sec_ind]
                            sec_ind += 1
                        plt.axvline(
                            x=rf_center + offset, marker="x", color="green"
                        )
                plt.legend()

                plt.show(
                    block=False
                    if fdbk_ind != len(continuousfdbk_time) - 1
                    else True
                )
        harm_per_section = self.harmonic // n_sections
        rf_center_list = np.array(rf_center_list)
        for trn_ind in range(
            0, n_turns_to_simulate - 1
        ):  # first turn is not recorded
            if (
                trn_ind == 0
            ):  # first recorded turn --> no acceleration, should all be equal
                for fdbk_ind in range(1, len(timing_fdbk_list)):
                    np.testing.assert_allclose(
                        rf_center_list[fdbk_ind][trn_ind],
                        rf_center_list[fdbk_ind - 1][trn_ind],
                    )
            if trn_ind == 1:
                for fdbk_ind in range(
                    1, len(timing_fdbk_list) - 1
                ):  # last one won't have any overlap --> only tracks within 2nd turn
                    overlapping_elements = (
                        len(timing_fdbk_list) - fdbk_ind - 1
                    ) * harm_per_section
                    np.testing.assert_allclose(
                        rf_center_list[fdbk_ind][trn_ind][
                            0:overlapping_elements
                        ],
                        rf_center_list[fdbk_ind - 1][trn_ind][
                            0:overlapping_elements
                        ],
                    )
                for fdbk_ind in range(
                    1, len(timing_fdbk_list)
                ):  # last elements of previous should be inside current
                    np.testing.assert_allclose(
                        rf_center_list[fdbk_ind][trn_ind][
                            -2 * harm_per_section : -harm_per_section
                        ],
                        rf_center_list[fdbk_ind - 1][trn_ind][
                            -harm_per_section:
                        ],
                    )
            if trn_ind >= 2:  # const acceleration
                for fdbk_ind in range(1, len(timing_fdbk_list)):
                    np.testing.assert_allclose(
                        rf_center_list[fdbk_ind][trn_ind][0:-harm_per_section],
                        rf_center_list[fdbk_ind - 1][trn_ind][
                            harm_per_section:
                        ],
                    )
            # Test of last backward and forward not overlapping
            # this should never be the case, as these are either lumped or have different frequencies.
            for fdbk_ind in range(
                1, len(timing_fdbk_list)
            ):  # last elements of previous should be inside current
                assert not any(
                    np.isclose(
                        rf_center_list[fdbk_ind][trn_ind][
                            -2 * harm_per_section : -harm_per_section
                        ],
                        rf_center_list[fdbk_ind][trn_ind][-harm_per_section:],
                        atol=0,
                        rtol=1e-12,
                    )
                ), f"{fdbk_ind}, {trn_ind}"  # type: ignore

        pass
