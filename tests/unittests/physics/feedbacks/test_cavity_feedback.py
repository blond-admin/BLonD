import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytest
from _pytest import unittest

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
    mu_minus,
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


_phase_shifts = [0, -1, 1]
# delta_omega_factor scales the RF-frequency offset applied to the station
# (delta_omega_rf = factor * omega_rf). The coarse grid tracks the *actual* RF
# frequency (design + delta_omega_rf; see forward_tracking_omega_rf and
# reverse_tracking_omega_list), so the rf_center spacing follows the detuned
# period the distance check compares against.
_delta_omega_factors = [0.0, 0.13, -0.13]
# Each entry is (n_rf_periods_per_coarse_grid, Q_L). The per-step decay is
# n * pi / Q_L and must stay below the hard stability cap of 2.0:
#   - integer n >= 1 places one coarse point every n RF periods; even n = 1
#     gives decay = pi / Q_L, so a large Q_L is needed to stay stable;
#   - n < 1 is the sub-stepping mode (several coarse points per RF period);
#     0.4 and 0.6 deliberately do not divide the harmonic evenly, exercising
#     the inter-turn carry-over, and stay stable even at Q_L = 1.
_n_ql_settings = [
    (1, 100),
    (2, 100),
    (3, 100),
    (0.25, 1),
    (0.4, 1),
    (0.6, 1),
]
test_data_discontinuity = [
    (phase_shift, delta_omega_factor, n_rf_points, q_l)
    for (n_rf_points, q_l) in _n_ql_settings
    for delta_omega_factor in _delta_omega_factors
    for phase_shift in _phase_shifts
]


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

    def _make_timing_feedback(self, n_rf_periods_per_coarse_grid, **kwargs):
        self.profile = StaticProfile.from_cutoff(0, 1e-9, 5e9)
        return IQCavityFeedbackTimingClass(
            profile=self.profile,
            n_rf_periods_per_coarse_grid=n_rf_periods_per_coarse_grid,
            R_over_Q=0,
            Q_L=1e6,
            generator_current=0,
            n_cavities=1,
            **kwargs,
        )

    def test_fractional_n_below_one_is_supported_without_warning(self) -> None:
        # n in (0, 1) is the sub-stepping mode (several coarse points per RF
        # period). It is a deliberate configuration, so constructing the
        # feedback must not emit the "coupling between loops" warning.
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            cav = self._make_timing_feedback(0.5)
        assert cav.n_rf_periods_per_coarse_grid == 0.5
        assert not any(
            "coupling between loops" in str(w.message) for w in record
        ), [str(w.message) for w in record]

    def test_non_integer_n_above_one_still_warns(self) -> None:
        # A non-integer number of *whole* RF periods (n >= 1) de-aligns the
        # coarse grid from the RF buckets and must still warn.
        with pytest.warns(UserWarning, match="coupling between loops"):
            self._make_timing_feedback(1.5)

    def test_non_positive_n_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            self._make_timing_feedback(0)

    def test_stability_check_reflects_actual_step_decay(self) -> None:
        # The forward-Euler decay applied in cavity_response() is
        #   0.5 * omega_rf * dt / Q_L = n_rf_periods_per_coarse_grid * pi / Q_L,
        # i.e. it scales with the number of RF periods per coarse step. The
        # sanity check must reflect that actual step, not collapse to the
        # n-independent omega_carrier * sampling_time_coarse == 2*pi.
        #
        # With Q_L = 2.0 a single-period step (n=1, decay = pi/2 ~ 1.57) is
        # below the hard cap of 2.0, but a two-period step (n=2, decay = pi ~
        # 3.14) is above it and must be rejected.
        backend.change_backend(Numpy64Bit)
        self.harmonic = 5
        self.setup_simulation()
        cav_fdbk_timing = IQCavityFeedbackTimingClass(
            profile=self.profile,
            n_rf_periods_per_coarse_grid=2,
            R_over_Q=0,
            Q_L=2.0,
            generator_current=0,
            n_cavities=1,
        )
        self.rf_station.attach_cavity_feedback(cav_fdbk_timing)
        cnst_cycle = ConstantMagneticCycle(
            reference_particle=mu_plus, value=63.0e9, in_unit="momentum"
        )
        sim = Simulation(self.ring, cnst_cycle)
        with pytest.raises(ValueError, match="decay_per_step"):
            sim.run_simulation(self.beam, n_turns=1)

    @pytest.mark.backend_mutation
    @pytest.mark.parametrize(
        "phase_shift,delta_omega_factor,n_rf_points,Q_L",
        test_data_discontinuity,
    )
    def test_for_discontinuity_distances_single_section_no_acceleration(
        self,
        phase_shift: float,
        delta_omega_factor: float,
        n_rf_points: float,
        Q_L: float,
    ) -> None:
        backend.change_backend(Numpy64Bit)
        self.harmonic = 5
        self.setup_simulation()
        cav_fdbk_timing = IQCavityFeedbackTimingClass(
            profile=self.profile,
            n_rf_periods_per_coarse_grid=n_rf_points,
            R_over_Q=0,
            Q_L=Q_L,
            generator_current=0,
            n_cavities=1,
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
                    cav_fdbk_timing.forward_tracking_omega_rf * time_array[-1]
                    + cav_fdbk_timing.phase_offset_frwrd
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
                f"{rf_centers_array[ind][0] - rf_centers_array[ind - 1][-1]} , {timestep_end}, {ind=}"
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
        "phase_shift,delta_omega_factor,n_rf_points,Q_L",
        test_data_discontinuity,
    )
    def test_for_discontinuity_distances_single_section_acceleration(
        self,
        phase_shift: float,
        delta_omega_factor: float,
        n_rf_points: float,
        Q_L: float,
    ) -> None:
        backend.change_backend(Numpy64Bit)
        self.harmonic = 5
        self.setup_simulation()
        cav_fdbk_timing = IQCavityFeedbackTimingClass(
            profile=self.profile,
            n_rf_periods_per_coarse_grid=n_rf_points,
            R_over_Q=0,
            Q_L=Q_L,
            generator_current=0,
            n_cavities=1,
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
                    R_over_Q=0,
                    # Q_L only needs to keep the forward-Euler cavity step
                    # stable; these tests check rf-center timing/geometry, not
                    # the cavity voltage, so a comfortably high Q_L is used to
                    # stay below the stability cap.
                    Q_L=100,
                    generator_current=0,
                    n_cavities=1,
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

            check_allclose_turn_printing_nested(
                time_passed_list,
                simulation.turn_i.value,
                f"time_passed to init time passed no acceleration",
            )  # with no acceleration, this has to be true (all time_passed are the same
            check_allclose_turn_printing_nested(
                omega_list,
                simulation.turn_i.value,
                f"omega to init omega no acceleration",
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
            if simulation.turn_i.value == 0:
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
    @pytest.mark.parametrize("n_sections", [1, 4, 10])
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

                    check_fail_printing(
                        fdbk.tracked_forward_until_element
                        in fdbk.current_slice_elements_forward,
                        f"{fdbk.tracked_forward_until_element} is in the current slice but should not yet be. ",
                    )

                    rf_center_list[idx][sim.turn_i.value - 1] = fdbk.rf_centers
                    check_fail_printing(
                        not (
                            fdbk.tracked_forward_until_element
                            is fdbk.reference_altering_elements[
                                (fdbk.own_index_in_reference_list + 3)
                                % len(fdbk.reference_altering_elements)
                            ]
                        ),
                        f"tracking did no include 3 elements (two drift frwrd + element itself).",
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
    @pytest.mark.parametrize(
        "n_sections", [1, 4, 10]
    )  # [1, 4, 20]  # TODO: why does this fail with 2?
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

        harm_per_half_drift = self.harmonic / n_sections / 2
        harm_per_full_drift = harm_per_half_drift * 2

        def callback(simulation: Simulation, beam: Beam):
            if simulation.turn_counter.value == 0:  # TODO: and not CR
                for idx, fdbk in enumerate(timing_fdbk_list):
                    fdbk: IQCavityFeedbackTimingClass
                    check_fail_printing(
                        len(fdbk.rf_centers)
                        != int(
                            np.floor(harm_per_half_drift)
                            + fdbk.section_index * harm_per_full_drift
                            + harm_per_full_drift
                        ),
                        f"improper rf centers length in turn zero for fdbk idx {idx}.",
                    )

                return
            for idx, fdbk in enumerate(timing_fdbk_list):
                fdbk: IQCavityFeedbackTimingClass
                if (
                    n_sections != 1
                ):  # only relevant/only gets set on multistation
                    check_fail_printing(
                        len(fdbk.rf_centers) != 20,
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

    @pytest.mark.parametrize("n_sections", [2, 4, 10])
    def test_rf_centers_full_counterrotation_equality(self, n_sections):
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

        harm_per_half_drift = self.harmonic / n_sections / 2
        harm_per_full_drift = harm_per_half_drift * 2

        def callback(simulation: Simulation, beam: Beam):
            # correct element until check
            for idx, fdbk in enumerate(timing_fdbk_list):
                if idx >= n_sections // 2:
                    # second half, corot last
                    assert (
                        fdbk.reference_index_until_tracked
                        == (  # TODO: rework with proper printing
                            fdbk.own_index_in_reference_list + 3
                        )
                        % len(fdbk.reference_altering_elements)
                    )
                    assert fdbk.reference_index_until_tracked_reverse == (
                        len(fdbk.reference_altering_elements)
                        - (fdbk.own_index_in_reference_list + 3 + 1)
                    ) % len(fdbk.reference_altering_elements)
                else:
                    # first half --> counterrot last
                    assert fdbk.reference_index_until_tracked == (
                        fdbk.own_index_in_reference_list - 3
                    ) % len(fdbk.reference_altering_elements)
                    assert fdbk.reference_index_until_tracked_reverse == (
                        len(fdbk.reference_altering_elements)
                        - (fdbk.own_index_in_reference_list - 3 + 1)
                    ) % len(fdbk.reference_altering_elements)
            # equality of values check here
            for idx, fdbk in enumerate(
                timing_fdbk_list[0 : len(timing_fdbk_list) // 2]
            ):
                reverse_index = len(timing_fdbk_list) - idx - 1
                np.testing.assert_allclose(
                    fdbk.rf_centers,
                    timing_fdbk_list[reverse_index].rf_centers,
                    atol=0,
                    rtol=1e-12,
                )
                check_allclose_turn_printing(
                    fdbk.rf_centers,
                    timing_fdbk_list[reverse_index].rf_centers,
                    simulation.turn_i.value,
                    "rf_centers equality check",
                )

            # correct length check
            for idx, fdbk in enumerate(
                timing_fdbk_list[0 : len(timing_fdbk_list) // 2]
            ):
                if n_sections == 2:
                    check_fail_printing(
                        len(fdbk.rf_centers) != int(harm_per_full_drift),
                        f"problem with rf centers length check for fdbk idx {idx} in turn {simulation.turn_i.value},"
                        f" was {len(fdbk.rf_centers)} but should have been {int(harm_per_full_drift)}.",
                    )
                    check_fail_printing(
                        len(timing_fdbk_list[1].rf_centers)
                        != int(harm_per_full_drift),
                        f"problem with rf centers length check for fdbk idx {idx} in turn {simulation.turn_i.value},"
                        f" was f{len(timing_fdbk_list[1].rf_centers)} but should have been {int(harm_per_full_drift)}.",
                    )
                    break
                else:
                    reverse_index = len(timing_fdbk_list) - idx - 1
                    should_be_length = int(
                        ((((n_sections - 1) / 2) - fdbk.section_index) * 2)
                        * harm_per_full_drift
                    )
                    check_fail_printing(
                        len(fdbk.rf_centers) != should_be_length,
                        f"problem with rf centers length check for fdbk idx {idx} in turn {simulation.turn_i.value},"
                        f" was {len(fdbk.rf_centers)} but should have been {should_be_length}.",
                    )

                    check_fail_printing(
                        len(timing_fdbk_list[reverse_index].rf_centers)
                        != should_be_length,
                        f"problem with rf centers reverse length check for fdbk idx {reverse_index} in turn {simulation.turn_i.value},"
                        f" was {len(timing_fdbk_list[reverse_index].rf_centers)} but should have been {should_be_length}",
                    )

            # save for further analysis
            for idx, fdbk in enumerate(timing_fdbk_list):
                fdbk: IQCavityFeedbackTimingClass
                if (
                    n_sections != 1
                ):  # only relevant/only gets set on multistation
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

        beam_cr = deepcopy(self.beam)
        beam_cr.reference._particle_type = mu_minus
        beam_cr._is_counter_rotating = True

        sim.run_simulation(
            (
                self.beam,
                beam_cr,
            ),
            callbacks=(callback,),
            n_turns=n_turns_to_simulate,
        )

        for trn_ind in range(
            0, n_turns_to_simulate - 1
        ):  # first turn is not recorded
            for fdbk_idx in range(n_sections // 2):
                reverse_index = len(timing_fdbk_list) - fdbk_idx - 1
                np.testing.assert_allclose(
                    rf_center_list[fdbk_idx][trn_ind],
                    rf_center_list[reverse_index][trn_ind],
                )
                np.testing.assert_allclose(
                    omega_list[fdbk_idx][trn_ind],
                    omega_list[reverse_index][trn_ind],
                )
                np.testing.assert_allclose(
                    time_passed_list[fdbk_idx][trn_ind],
                    time_passed_list[reverse_index][trn_ind],
                )


class TestIQCavityFeedbackObservationClass(unittest.TestCase):
    pass
