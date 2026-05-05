import matplotlib.pyplot as plt
import numpy as np
from examples.scripts.EX_09_Semi_empiric_matcher import (
    bucket_fill_by_emittance_gaussian,
)
from experimental import SemiEmpiricMatcher
from handle_results.observables import IQCavityFeedbackObservation
from handle_results.observables_as_elements import (
    BeamObservationInRingElement,
    BunchObservationMetaParams,
    InducedVoltageObservationCR,
)
from physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass
from physics.impedances.solvers import (
    MultiPassResonatorSolver,
    SingleTurnResonatorConvolutionSolver,
)
from scipy.constants import elementary_charge, speed_of_light
from scipy.interpolate import interp1d
from specifics.muon_collider.beam_preparation import (
    load_beam_coordinates_counterrot_from_file,
    load_beam_coordinates_from_file,
)

from blond import (
    Beam,
    DriftSimple,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    backend,
    mu_plus,
)
from blond.cycles.magnetic_cycle import MagneticCyclePerTurnAllRFStations
from unittests.physics.impedances.comparisons.mtw import voltage_per_cavity

n_slices = 2**10


def match_beam(simulation, t_rf, beam):
    simulation.prepare_beam(
        preparation_routine=SemiEmpiricMatcher(
            time_limit=[1.0 * t_rf, t_rf * 2.0],
            n_macroparticles=int(1e6),
            hamilton_to_density_function=bucket_fill_by_emittance_gaussian,
            hamilton_to_density_kwargs={
                "emittance_list": [
                    0.025,  # RMS emittance for each bunch
                ],
                "intensity_frac_list": [
                    1.0,
                ],  # Intensity fraction of each bunch
                "n_buckets": 1,  # Number of buckets within time_limit
                "max_emittance_diff": 1e-6,
            },
            # verbose=True,
            animate=True,
            increment_intensity_effects_until_iteration_i=1,
            until_section_index=2,
        ),
        beam=beam,
    )


def setup_and_run(
    rcs: str = "RCS1",
    MTW: bool = False,
    n_stations: int = 8,
    beam_observation=False,
    n_turns_in: int = -1,
):
    """

    Parameters
    ----------
    rcs
        RCS string, like `RCS1` or `RCS2`.
    MTW
        If this flag is True, the simulation will be run with a convolution solver and not
        the feedback.
    Returns
    -------

    """
    backend.set_specials("cpp")

    if rcs == "RCS1":
        R_over_Q = 3 * 518
        Q_L = 1.29e6
        phi_s = 2.5830872929516078  # 143
        alpha_p = 10.395e-4
        bunch_intensity = 2.7e12
        circumference = 5990
        injection_energy = 63e9
        ejection_energy = 313.8e9
        f_det = -1040
        harmonic = 25900
        n_turns = 18

    elif rcs == "RCS4":
        R_over_Q = 518
        Q_L = 4.34e6
        phi_s = 2.0594885173533086  # 118
        alpha_p = 2.114e-4
        bunch_intensity = 2.0e12
        circumference = 35000
        injection_energy = 1500e9
        ejection_energy = 5000e9
        f_det = -190
        harmonic = 151400
        n_turns = 55
    else:
        raise ValueError("Unknown RCS")
    f_det = 0
    # phi_s = np.pi / 2
    harmonic = int(harmonic - harmonic % n_stations)

    voltage_per_cavity = 31140000.0
    energy_gain_per_turn = (ejection_energy - injection_energy) / n_turns
    total_voltage = energy_gain_per_turn / np.sin(phi_s)
    total_voltage = 1e9
    voltage_per_station = total_voltage / n_stations
    n_cavities = voltage_per_station / voltage_per_cavity * n_stations
    cav_per_station = n_cavities / n_stations

    delta_omega = 2 * np.pi * f_det

    ring = Ring(circumference=circumference, check_section_indices=False)
    magnetic_cycle = MagneticCyclePerTurnAllRFStations(
        value_init=injection_energy,
        # values_after_rf_station_per_turn=np.linspace(
        #     injection_energy + energy_gain_per_turn / n_stations,
        #     ejection_energy,
        #     n_turns * n_stations,
        # ).reshape(n_stations, n_turns, order="F"),
        values_after_rf_station_per_turn=injection_energy
        * np.ones(
            n_turns * n_stations,
        ).reshape(n_stations, n_turns, order="F"),
        in_unit="total energy",
        reference_particle=mu_plus,
    )

    t_rf = (
        magnetic_cycle.get_t_rev_init(
            ring.circumference,
            particle_type=mu_plus,
        )
        / harmonic
    )
    # t_rf

    beam_current = (
        elementary_charge
        * bunch_intensity
        * speed_of_light
        / ring.circumference
    )
    omega_rf = 1 / t_rf * 2 * np.pi

    F_b = 2 * (-0.8330691630689783 - 0.060605390015254904j)

    # delta_omega = omega_rf * R_over_Q * np.abs(F_b) * beam_current * np.cos(phi_s) / (2 * voltage_per_cavity)
    phi_s = np.pi / 2
    Q_L = voltage_per_cavity / (
        R_over_Q * (np.abs(F_b) * beam_current * np.sin(phi_s)) ** 2
        - (
            np.abs(F_b) * beam_current * np.cos(phi_s)
            + voltage_per_cavity * 2 * delta_omega / (omega_rf * R_over_Q)
        )
        ** 2
    )

    I_g = (
        voltage_per_cavity
        / (2 * R_over_Q)
        * (1 / Q_L - 2j * delta_omega / omega_rf)
        + np.abs(F_b) * beam_current * np.exp(1j * (phi_s - np.pi / 2)) / 2
    )

    I_g_ampl = np.abs(I_g)
    I_g_angle = np.angle(I_g)

    # initial_voltage = voltage_per_cavity + (
    #             -voltage_per_cavity / (2 * Q_L / omega_rf) + R_over_Q * omega_rf * I_g_ampl * np.cos(np.angle)) * t_rf * harmonic / 2
    # initial_phase = (delta_omega + R_over_Q * omega_rf * I_g_ampl * np.sin(I_g_angle) / voltage_per_cavity) * t_rf * harmonic / 2

    beam = Beam(
        intensity=bunch_intensity,
        particle_type=mu_plus,
        is_counter_rotating=False,
    )

    beam.reference.total_energy = injection_energy

    bunch_observation = BunchObservationMetaParams(each_turn_i=1, beam=beam)

    beam_observation_full_corot = (
        None
        if not beam_observation
        else BeamObservationInRingElement(beam=beam, each_turn_i=1)
    )

    ind_volt_obs_list = []

    one_turn_model = []
    profile_list = []
    shc_list = []
    cav_fdbk_obs_list = []

    for cavity_i in range(n_stations):
        profile_list.append(
            StaticProfile.from_rad(
                np.pi * 2,
                np.pi * 4,
                n_slices,
                t_rf,
                section_index=cavity_i,
            )
        )

        local_res = Resonators(
            shunt_impedances=R_over_Q * Q_L * cav_per_station,
            quality_factors=Q_L,
            center_frequencies=1 / t_rf + f_det,
        )
        wf = (
            WakeField(
                sources=(local_res,),
                # solver=SingleTurnResonatorConvolutionSolver(),
                solver=MultiPassResonatorSolver(
                    decay_fraction_threshold=1e-12, allow_delta_t_zero=True
                ),
                profile=profile_list[-1],
            )
            if MTW
            else None
        )
        cav_fdbk = (
            IQCavityFeedbackTimingClass(
                profile=profile_list[-1],
                R_over_Q=R_over_Q,
                Q_L=Q_L,
                n_rf_periods_per_coarse_grid=1,
                generator_current=I_g,
                n_cavities=cav_per_station,
                initial_voltage=voltage_per_cavity,
                delta_omega=delta_omega,
            )
            if not MTW
            else None
        )
        cav_fdbk_obs_list.append(
            IQCavityFeedbackObservation(
                each_turn_i=1,
                feedback=cav_fdbk,
            )
            if not MTW
            else None
        )
        shc_list.append(
            SingleHarmonicRFStation(
                voltage=voltage_per_station,
                phi_rf=0,
                harmonic=harmonic,
                cavity_feedback=cav_fdbk,
                local_wakefield=wf,
                profile=profile_list[-1],
                section_index=cavity_i,
            )
        )
        ind_volt_obs_list.append(
            InducedVoltageObservationCR(
                each_turn_i=1,
                wake_field=shc_list[-1]._local_wakefield,
                section_index=cavity_i,
            )
            if MTW
            else None
        )

        one_turn_model.extend(
            [
                DriftSimple(
                    momentum_compaction_factor=alpha_p,
                    orbit_length=circumference / n_stations / 2,
                    section_index=cavity_i,
                ),
                ind_volt_obs_list[-1],
                shc_list[-1],
                ind_volt_obs_list[-1],
                DriftSimple(
                    momentum_compaction_factor=alpha_p,
                    orbit_length=circumference / n_stations / 2,
                    section_index=cavity_i,
                ),
                bunch_observation,
            ]
        )
    ring.add_elements(one_turn_model, reorder=False)
    ####################################################################

    bunch_observation.active = False

    sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)

    # if MTW:
    #     match_beam(
    #         sim,
    #         t_rf,
    #         beam,
    #     )
    #     np.savez(
    #         "./fdbk_testing/init_distr_convol.npz",
    #         dE=beam.dE.array_local,
    #         dt=beam.dt.array_local,
    #     )
    # else:
    load_beam_coordinates_from_file(
        "./fdbk_testing/init_distr_convol.npz", beam
    )

    bunch_observation.active = True

    beam_freq = np.fft.rfftfreq(
        100 * profile_list[-1].n_bins, profile_list[-1].hist_step
    )
    beam_spectrum = profile_list[-1].beam_spectrum(
        100 * profile_list[-1].n_bins
    )
    rf_frequency_component = (
        interp1d(beam_freq, beam_spectrum)(omega_rf / (2 * np.pi))
        / beam_spectrum[0]
    )
    # (-0.8330691630689783-0.060605390015254904j)

    profile_list[-1].beam_spectrum(100 * profile_list[-1].n_bins)

    sim.run_simulation(
        (beam,),
        n_turns=None if n_turns_in == -1 else n_turns_in,
        observe=cav_fdbk_obs_list if not MTW else (),
    )

    return (
        bunch_observation,
        n_turns,
        ind_volt_obs_list,
        cav_fdbk_obs_list,
    )


def plot_results(bunch_obs_list, n_turns_list, ind_volt_obs_list):
    plt.title("sigma_dt")
    plt.plot(bunch_obs_list[0].sigma_dt, label="MTW")
    plt.plot(bunch_obs_list[1].sigma_dt, ls="--", label="fdbk")
    plt.legend()
    plt.show()

    plt.title("rms_emittance")
    plt.plot(bunch_obs_list[0].rms_emittance, label="MTW")
    plt.plot(bunch_obs_list[1].rms_emittance, ls="--", label="fdbk")
    plt.legend()
    plt.show()

    plt.title("sigma_dE")
    plt.plot(bunch_obs_list[0].sigma_dE, label="MTW")
    plt.plot(bunch_obs_list[1].sigma_dE, ls="--", label="fdbk")
    plt.legend()
    plt.show()

    plt.title("mean_dt")
    plt.plot(bunch_obs_list[0].mean_dt, label="MTW")
    plt.plot(bunch_obs_list[1].mean_dt, ls="--", label="fdbk")
    plt.legend()
    plt.show()

    plt.title("mean_dE")
    plt.plot(bunch_obs_list[0].mean_dE, label="MTW")
    plt.plot(bunch_obs_list[1].mean_dE, ls="--", label="fdbk")
    plt.legend()
    plt.show()


def plot_ind_volt_cav_fdbk_voltage(ind_volt_obs_list, cav_fdbk_obs_list):
    # plt.clf()
    # fix, ax = plt.subplots()
    # plt.title("ind_volt vs fdbk_kick")
    #
    # ax.plot(
    #     ind_volt_obs_list[0][0].induced_voltage[0], ls="--", label="ind_volt"
    # )
    # # ax.plot(cav_fdbk_obs_list[1][0].v_corr[0] * 30e6, label="rel_volt correction")
    # ax.plot(
    #     np.abs(cav_fdbk_obs_list[1][0].v_ant_fine[0]), label="abs v_ant_fine"
    # )
    # ax.plot(
    #     np.real(cav_fdbk_obs_list[1][0].v_ant_fine[0]), label="real v_ant_fine"
    # )

    # ax2 = ax.twinx()
    # ax2.plot(ind_volt_obs_list[0][0].beam_profile[0], label="beam profile")
    # ax2.plot(
    #     cav_fdbk_obs_list[1][0].v_corr[0] * 30e6, label="rel_volt correction"
    # )
    # ax2.plot(cav_fdbk_obs_list[1][0].phi_corr[0], label="rel_volt correction")
    #
    # plt.legend()
    # plt.show(block=False)
    #
    # fix, ax = plt.subplots()
    # plt.title("coarse")
    #
    # ax.plot(cav_fdbk_obs_list[1][0].v_ant_coarse[0], label="ind_volt")
    #
    # ax2 = ax.twinx()
    # ax2.plot(ind_volt_obs_list[0][0].beam_profile[0], label="beam profile")
    # # ax2.plot(cav_fdbk_obs_list[1][0].v_corr[0] * 30e6, label="rel_volt correction")
    #
    # plt.legend()
    # plt.show(block=False)

    fig, ax = plt.subplots(2, 2, sharex=True)
    for idx in range(1):
        clr = ax[0, 0]._get_lines.get_next_color()
        ax[0, 0].plot(
            ind_volt_obs_list[0][0].total_voltage[idx], color=clr, label="MTW"
        )
        ax[0, 0].plot(
            np.real(cav_fdbk_obs_list[1][0].kick_voltage_fine[idx]),
            ls="--",
            color=clr,
            label="real fdbk",
        )
        cavity_voltage_raw = (
            ind_volt_obs_list[0][0].total_voltage[idx]
            - ind_volt_obs_list[0][0].induced_voltage[idx]
        )
        ax[0, 1].plot(
            np.real(cav_fdbk_obs_list[1][0].kick_voltage_fine[idx])
            - cavity_voltage_raw,
            ls="--",
            color=clr,
            label="real fdbk",
        )
        ax[0, 1].plot(
            ind_volt_obs_list[0][0].induced_voltage[idx],
            color=clr,
            label="MTW",
        )
        ax[1, 0].set_title("v_corr")
        ax[1, 0].plot(
            np.real(cav_fdbk_obs_list[1][0].v_corr[idx]),
            color=clr,
            label="v_corr",
        )
        ax[1, 1].set_title("phi_corr")
        ax[1, 1].plot(
            np.real(cav_fdbk_obs_list[1][0].phi_corr[idx]),
            color=clr,
            label="phi_corr",
        )

        # ax[1].plot(np.abs(cav_fdbk_obs_list[1][0].kick_voltage_fine[idx]), label="abs fdbk")
    plt.tight_layout()
    # plt.legend()
    plt.show()

    pass


if __name__ == "__main__":
    n_sections = 8
    bunch_obs_list, n_turns_list, ind_volt_obs_list, cav_fdbk_obs_list = (
        [],
        [],
        [],
        [],
    )
    for MTW in [
        True,
        False,
    ]:
        (
            bunch_observation_buf,
            n_turns_buf,
            ind_volt_obs_list_buf,
            cav_fdbk_obs_list_buf,
        ) = setup_and_run("RCS1", MTW=MTW, n_stations=n_sections, n_turns_in=1)
        bunch_obs_list.append(bunch_observation_buf)
        n_turns_list.append(n_turns_buf)
        ind_volt_obs_list.append(ind_volt_obs_list_buf)
        cav_fdbk_obs_list.append(cav_fdbk_obs_list_buf)

    plot_ind_volt_cav_fdbk_voltage(ind_volt_obs_list, cav_fdbk_obs_list)

    plot_results(bunch_obs_list, n_turns_list, ind_volt_obs_list)


# TODO: tests to write
"""
check no voltage change over one turn for no detuned cavity --> purely real generator current, should not change between turns

"""
