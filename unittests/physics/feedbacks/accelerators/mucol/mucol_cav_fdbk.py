"""Muon collider RCS cavity-feedback simulation setup and plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import elementary_charge, speed_of_light
from scipy.interpolate import interp1d

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
from blond.examples.scripts.EX_09_Semi_empiric_matcher import (
    bucket_fill_by_emittance_gaussian,
)
from blond.experimental import SemiEmpiricMatcher
from blond.handle_results.observables import IQCavityFeedbackObservation
from blond.handle_results.observables_as_elements import (
    BeamObservationInRingElement,
    BunchObservationMetaParams,
    InducedVoltageObservationCR,
)
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass
from blond.physics.feedbacks.helpers import rf_beam_current
from blond.physics.impedances.solvers import (
    MultiPassResonatorSolver,
    SingleTurnResonatorConvolutionSolver,
)
from blond.specifics.muon_collider.beam_preparation import (
    load_beam_coordinates_from_file,
)

n_slices = 2**10


def match_beam(simulation, t_rf, beam):
    """
    Match the beam into the bucket using the semi-empiric matcher.

    Parameters
    ----------
    simulation
        Simulation object to prepare the beam on.
    t_rf
        RF period used to define the matching time limits.
    beam
        Beam object to be matched.
    """
    simulation.prepare_beam(
        preparation_routine=SemiEmpiricMatcher(
            time_limit=[1.2 * t_rf, t_rf * 2.0],
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
            until_section_index=1,
        ),
        beam=beam,
    )


# TODO: split this setup into helpers and remove the PLR0915 noqa
def setup_and_run(  # noqa: PLR0915
    rcs: str = "RCS1",
    MTW: bool = False,
    n_stations: int = 8,
    beam_observation=False,
    n_turns_in: int = -1,
    force_rematch: bool = False,
    acceleration: bool = True,
):
    """
    Set up and run a muon collider RCS cavity-feedback simulation.

    Parameters
    ----------
    rcs
        RCS string, like `RCS1` or `RCS2`.
    MTW
        If this flag is True, the simulation will be run with a convolution
        solver and not the feedback.
    n_stations
        Number of RF stations around the ring.
    beam_observation
        If True, enable per-turn beam observation.
    n_turns_in
        Number of turns to run; -1 uses the default for the chosen RCS.
    force_rematch
        If True, force a rematch of the beam before tracking.
    acceleration
        If True, run with an accelerating magnetic cycle.

    Returns
    -------
    bunch_observation
        Observation object holding the tracked bunch quantities.
    n_turns
        Number of turns the simulation was run for.
    ind_volt_obs_list
        List of induced-voltage observations.
    cav_fdbk_obs_list
        List of cavity-feedback observations.
    """
    backend.set_specials("cpp")

    if rcs == "RCS1":
        R_over_Q = 518
        # Q_L = 1.29e6
        phi_s = 2.5830872929516078  # 148
        # phi_s = phi_s - np.pi / 2
        alpha_p = 10.395e-4
        bunch_intensity = 2.7e12
        circumference = 5990
        injection_energy = 63e9
        ejection_energy = 313.8e9
        # f_det = -1040
        harmonic = 25900
        n_turns = 18
        F_b = 2 * (-0.9496176885609792 - 0.20292067206919637j)

    elif rcs == "RCS2":
        R_over_Q = 518
        # Q_L = 1.29e6
        phi_s = 2.670353755551324  # 153
        alpha_p = 8.986e-4
        bunch_intensity = 2.4e12
        circumference = 5990
        injection_energy = 313.8e9
        ejection_energy = 750e9
        # f_det = -1040
        harmonic = 25900
        n_turns = 55

        F_b = 2 * (-0.8578510208833927 - 0.47979182305969775j)

    elif rcs == "RCS4":
        R_over_Q = 518
        Q_L = 4.34e6
        phi_s = 2.0594885173533086  # 118
        alpha_p = 2.114e-4
        bunch_intensity = 2.0e12
        circumference = 35000
        injection_energy = 1500e9
        ejection_energy = 5000e9
        # f_det = -190
        harmonic = 151400
        n_turns = 55
        F_b = 2 * (-0.995635282077265 - 0.021613922916270206j)
    else:
        raise ValueError("Unknown RCS")
    # f_det = 0
    # phi_s = np.pi / 2
    # phi_s = np.pi

    voltage_per_cavity = 31140000.0
    energy_gain_per_turn = (ejection_energy - injection_energy) / n_turns / 20
    # phi_s = 170 * np.pi / 180
    scalor = 1
    harmonic /= scalor
    harmonic = int(
        harmonic - harmonic % (n_stations * 2)
    )  # every half drift has integer number of drifts
    circumference /= scalor
    total_voltage = energy_gain_per_turn / np.sin(phi_s)
    # total_voltage = 1e9
    voltage_per_station = total_voltage / n_stations
    n_cavities = voltage_per_station / voltage_per_cavity * n_stations
    cav_per_station = n_cavities / n_stations

    # delta_omega = 2 * np.pi * f_det

    ring = Ring(circumference=circumference, check_section_indices=False)
    magnetic_cycle = MagneticCyclePerTurnAllRFStations(
        value_init=injection_energy,
        values_after_rf_station_per_turn=np.linspace(
            injection_energy + energy_gain_per_turn / n_stations,
            ejection_energy,
            n_turns * n_stations,
        ).reshape(n_stations, n_turns, order="F")
        if acceleration
        else injection_energy
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

    delta_omega = (
        omega_rf
        * R_over_Q
        * np.abs(F_b)
        * beam_current
        * np.cos(phi_s)
        / (voltage_per_cavity)
    ) / np.sin(phi_s - np.pi / 2) ** 2
    # delta_omega = 0
    f_det = delta_omega / (2 * np.pi)
    # phi_s = np.pi / 2
    Q_L = voltage_per_cavity / (
        R_over_Q
        * np.sqrt(
            (2 * np.abs(F_b) * beam_current * np.sin(phi_s)) ** 2
            + (
                -np.abs(F_b) * beam_current * np.cos(phi_s)
                + voltage_per_cavity * delta_omega / (omega_rf * R_over_Q)
            )
            ** 2
        )
    )

    I_g = (
        voltage_per_cavity
        / (2 * R_over_Q)
        # * (1 / Q_L - 5.2 / 2 * 1j * delta_omega / omega_rf)
        * (1 / Q_L - 2j * delta_omega / omega_rf)
        # + np.abs(F_b) * beam_current * np.exp(1j * (phi_s - np.pi/2)) * 0.125
    )

    _I_g_ampl = np.abs(I_g)
    _I_g_angle = np.angle(I_g)

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

    _beam_observation_full_corot = (
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
                np.pi * 1.5,
                np.pi * 4.5,
                n_slices,
                t_rf,
                section_index=cavity_i,
            )
        )

        local_res = Resonators(
            shunt_impedances=R_over_Q * Q_L * cav_per_station,
            quality_factors=Q_L,
            center_frequencies=1 / t_rf,  # + f_det,
        )
        wf = (
            WakeField(
                sources=(local_res,),
                # solver=SingleTurnResonatorConvolutionSolver(),
                solver=MultiPassResonatorSolver(
                    decay_fraction_threshold=1e-12,
                    allow_delta_t_zero=False,
                    delta_f=f_det,
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
        # shc_list[-1].delta_omega_rf = delta_omega
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

    if MTW and force_rematch:
        match_beam(
            sim,
            t_rf,
            beam,
        )
        np.savez(
            f"./fdbk_testing/init_distr_convol_{rcs}_n_stations_{n_stations}.npz",
            dE=beam.dE.array_local,
            dt=beam.dt.array_local,
        )
    else:
        load_beam_coordinates_from_file(
            f"./fdbk_testing/init_distr_convol_{rcs}_n_stations_{n_stations}.npz",
            beam,
        )

    bunch_observation.active = True

    # F_B calculation
    profile_list[0].track(beam=beam)
    beam_freq = np.fft.rfftfreq(
        100 * profile_list[0].n_bins, profile_list[0].hist_step
    )
    beam_spectrum = profile_list[0].beam_spectrum(100 * profile_list[0].n_bins)
    _rf_frequency_component = (
        interp1d(beam_freq, beam_spectrum)(omega_rf / (2 * np.pi))
        / beam_spectrum[0]
    )  # needs to be multiplied by two for F_b
    # (-0.8330691630689783-0.060605390015254904j)

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
    """
    Plot the bunch length (sigma_dt) for the MTW and feedback runs.

    Parameters
    ----------
    bunch_obs_list
        List of bunch observations; index 0 is MTW, index 1 the feedback run.
    n_turns_list
        Number of turns per run (unused in the plot itself).
    ind_volt_obs_list
        List of induced-voltage observations (unused in the plot itself).
    """
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


def plot_ind_volt_cav_fdbk_voltage(
    ind_volt_obs_list, cav_fdbk_obs_list, n_turns_in: int, n_stations: int
):
    """
    Plot induced voltage against the cavity-feedback voltage per station.

    Parameters
    ----------
    ind_volt_obs_list
        List of induced-voltage observations.
    cav_fdbk_obs_list
        List of cavity-feedback observations.
    n_turns_in
        Number of turns that were simulated.
    n_stations
        Number of RF stations around the ring.
    """
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

    #
    # turn_idx = 1
    # fig, ax = plt.subplots(2, 2, sharex=True)
    # ax[0, 0].set_title("through stations")
    # for idx in range(3):
    #     clr = ax[0, 0]._get_lines.get_next_color()
    #     ax[0, 0].plot(
    #         ind_volt_obs_list[0][idx].total_voltage[turn_idx],
    #         color=clr,
    #         label="MTW",
    #     )
    #     ax[0, 0].plot(
    #         np.real(cav_fdbk_obs_list[1][idx].kick_voltage_fine[turn_idx]),
    #         ls="--",
    #         color=clr,
    #         label="real fdbk",
    #     )
    #     cavity_voltage_raw = (
    #         ind_volt_obs_list[0][idx].total_voltage[turn_idx]
    #         - ind_volt_obs_list[0][idx].induced_voltage[turn_idx]
    #     )
    #     ax[0, 1].plot(
    #         np.real(cav_fdbk_obs_list[1][idx].kick_voltage_fine[turn_idx])
    #         - cavity_voltage_raw,
    #         ls="--",
    #         color=clr,
    #         label="real fdbk",
    #     )
    #     ax[0, 1].plot(
    #         np.imag(cav_fdbk_obs_list[1][idx].kick_voltage_fine[turn_idx]),
    #         # - cavity_voltage_raw,
    #         ls=":",
    #         color=clr,
    #         label="real fdbk",
    #     )
    #     ax[0, 1].plot(
    #         ind_volt_obs_list[0][idx].induced_voltage[turn_idx],
    #         color=clr,
    #         label="MTW",
    #     )
    #     ax[1, 0].set_title("v_corr")
    #     ax[1, 0].plot(
    #         np.real(cav_fdbk_obs_list[1][idx].v_corr[turn_idx]),
    #         color=clr,
    #         label="v_corr",
    #     )
    #     ax[1, 1].set_title("phi_corr")
    #     ax[1, 1].plot(
    #         np.real(cav_fdbk_obs_list[1][idx].phi_corr[turn_idx]),
    #         color=clr,
    #         label="phi_corr",
    #     )
    #
    #     # ax[1].plot(np.abs(cav_fdbk_obs_list[1][0].kick_voltage_fine[idx]), label="abs fdbk")
    # plt.tight_layout()
    # # plt.legend()
    # plt.show(block=False)

    trn_idx = 0
    plt.figure("coarse_voltage")
    plt.plot(np.real(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]), color="k")
    plt.plot(np.imag(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]), color="k")
    # plt.plot(np.abs(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]), color="k")
    trn_idx = 1
    plt.plot(
        np.real(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]),
        color="b",
        ls="--",
    )
    plt.plot(
        np.imag(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]),
        color="b",
        ls="--",
    )
    # plt.plot(np.abs(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]), color="b", ls="--")
    plt.show(block=False)

    trn_idx = 0
    plt.figure("coarse_voltage_2")
    plt.plot(
        np.angle(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]), color="k"
    )
    trn_idx = 1
    plt.plot(
        np.angle(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]),
        color="b",
        ls="--",
    )
    # plt.plot(np.imag(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]), color="b", ls="--")
    # plt.plot(np.abs(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]), color="b", ls="--")
    plt.show(block=False)

    fig, ax = plt.subplots(2, 2, sharex=True)
    for idx in range(n_turns_in):
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
        feedback_argmax = np.argmax(
            np.real(cav_fdbk_obs_list[1][0].kick_voltage_fine[idx][0:300])
            - cavity_voltage_raw[0:300]
        )
        print(f"feedback argmax {feedback_argmax}")
        ax[0, 1].plot(
            np.real(cav_fdbk_obs_list[1][0].kick_voltage_fine[idx])
            - cavity_voltage_raw,
            ls="--",
            color=clr,
            label="real fdbk",
        )
        ax[0, 1].plot(
            np.imag(cav_fdbk_obs_list[1][0].kick_voltage_fine[idx]),
            # - cavity_voltage_raw,
            ls=":",
            color=clr,
            label="real fdbk",
        )
        print(
            f"ind volt argmax {np.argmax(ind_volt_obs_list[0][0].induced_voltage[idx][0:300])}"
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


# Sign relating the cavity IQ envelope reconstruction
# Re[V_ant * exp(i*omega_rf*t)] to the resonator induced voltage.
# Determined empirically; see benchmark_single_turn_fine_grid_vs_resonator.
_CAVITY_TO_INDUCED_VOLTAGE_SIGN = -1.0


def compute_single_turn_fine_grid_vs_resonator(
    delta_omega: float = 0.0,
    seed: int = 0,
    R_over_Q: float = 518.0,
    Q_L: float = 1287601.7251526634,
    f_rf: float = 1.3e9,
    intensity: float = 2.7e12,
    n_macroparticles: int = int(1e6),
    n_bins: int = 2**12,
    sigma_t_frac: float = 0.06,
    noise_fraction: float = 0.1,
):
    r"""
    Benchmark the single-turn (fine-grid) cavity beam-loading response.

    The fine-grid response of :class:`IQCavityFeedbackTimingClass`
    (``cavity_response_fine``) is solved for a real Gaussian-plus-noise beam
    profile with the generator current set to zero, so the antenna voltage is
    purely the beam-induced (beam-loading) voltage envelope.

    This is compared against an independent reference: the induced voltage of
    a :class:`Resonators` source (``R_s = R_over_Q * Q_L``, ``Q = Q_L``,
    ``f_r = f_rf + delta_omega / 2 / pi``) convolved with the same beam
    profile via :class:`SingleTurnResonatorConvolutionSolver`.

    The cavity result is a complex IQ envelope demodulated at ``omega_rf``;
    it is remodulated to the lab frame as
    ``Re[V_ant * exp(i * omega_rf * t)]`` (times a fixed sign convention) so
    it can be compared directly to the real resonator induced voltage.

    Parameters
    ----------
    delta_omega
        Cavity detuning in [rad/s]. The reference resonator is detuned by the
        same amount, so this exercises the detuning phase shift.
    seed
        Seed for the random beam distribution.
    R_over_Q, Q_L, f_rf, intensity, n_macroparticles, n_bins
        Cavity / beam / discretization parameters.
    sigma_t_frac
        Bunch RMS length as a fraction of the RF period.
    noise_fraction
        Fraction of macroparticles distributed uniformly (noise) instead of
        Gaussian.

    Returns
    -------
    result
        Dict with ``hist_x``, ``v_resonator``, ``v_cavity_lab`` and the
        agreement metrics ``scale``, ``nrmse`` and ``corr``.
    """
    from unittest.mock import Mock

    from scipy.signal import hilbert  # noqa: F401  (kept for env diagnostics)

    rng = np.random.default_rng(seed)

    t_rf = 1.0 / f_rf
    omega_rf = 2.0 * np.pi * f_rf

    # Profile covering 3 RF periods with the bunch in the central one.
    profile = StaticProfile.from_rad(0.5 * np.pi, 3.5 * np.pi, n_bins, t_rf)
    t_center = t_rf  # one RF period into the window
    sigma_t = sigma_t_frac * t_rf

    n_noise = int(noise_fraction * n_macroparticles)
    n_gauss = n_macroparticles - n_noise
    dt = np.concatenate(
        [
            rng.normal(t_center, sigma_t, n_gauss),
            rng.uniform(
                t_center - 4 * sigma_t, t_center + 4 * sigma_t, n_noise
            ),
        ]
    )
    beam = Beam(
        intensity=intensity,
        particle_type=mu_plus,
        is_counter_rotating=False,
    )
    beam.setup_beam(dt=dt, dE=np.zeros_like(dt), mpi_mode="root-distributes")
    profile.track(beam=beam)

    # --- cavity fine-grid beam-induced response (generator current = 0) ---
    charges_fine = rf_beam_current(
        beam=beam,
        profile=profile,
        omega_c=omega_rf,
        T_rev=t_rf,
        use_lowpass_filter=False,
        external_reference=False,
    )
    cav = IQCavityFeedbackTimingClass(
        profile=profile,
        R_over_Q=R_over_Q,
        Q_L=Q_L,
        generator_current=0.0 + 0.0j,
        n_cavities=1,
        initial_voltage=0.0,
        delta_omega=delta_omega,
    )
    cav.beam_current_fine_grid = charges_fine / profile.hist_step
    cav.generator_current_fine_grid = np.zeros(n_bins, dtype=complex)
    cav.cavity_response_fine(
        initial_voltage_fine_grid=0.0,
        initial_voltage_gradient_fine_grid=0.0,
        initial_generator_current_fine_grid=0.0,
        samples_per_rf_fine_grid=omega_rf * profile.hist_step,
        relative_detuning=delta_omega / omega_rf,
    )
    v_cavity_lab = _CAVITY_TO_INDUCED_VOLTAGE_SIGN * np.real(
        cav.antenna_voltage_fine_grid * np.exp(1j * omega_rf * profile.hist_x)
    )

    # --- resonator induced-voltage reference ---
    res = Resonators(
        shunt_impedances=R_over_Q * Q_L,
        quality_factors=Q_L,
        center_frequencies=f_rf + delta_omega / (2.0 * np.pi),
    )
    wf = WakeField(
        sources=(res,),
        solver=SingleTurnResonatorConvolutionSolver(),
        profile=profile,
    )
    wf.solver.on_wakefield_init_simulation(Mock(), wf)
    v_resonator = np.asarray(wf.solver.calc_induced_voltage(beam=beam))

    scale = float(
        np.dot(v_resonator, v_cavity_lab) / np.dot(v_cavity_lab, v_cavity_lab)
    )
    nrmse = float(
        np.sqrt(np.mean((v_resonator - scale * v_cavity_lab) ** 2))
        / np.abs(v_resonator).max()
    )
    corr = float(np.corrcoef(v_cavity_lab, v_resonator)[0, 1])

    return {
        "hist_x": profile.hist_x,
        "v_resonator": v_resonator,
        "v_cavity_lab": v_cavity_lab,
        "scale": scale,
        "nrmse": nrmse,
        "corr": corr,
    }


def benchmark_single_turn_fine_grid_vs_resonator(
    delta_omega_list=(0.0, 5e6, -2e7),
):
    """
    Plot and report the single-turn fine-grid vs resonator benchmark.

    Parameters
    ----------
    delta_omega_list
        Detuning values [rad/s] to benchmark, one subplot each.
    """
    fig, axes = plt.subplots(
        len(delta_omega_list), 1, sharex=True, figsize=(8, 9)
    )
    if len(delta_omega_list) == 1:
        axes = [axes]

    for ax, delta_omega in zip(axes, delta_omega_list):
        r = compute_single_turn_fine_grid_vs_resonator(delta_omega=delta_omega)
        print(
            f"delta_omega={delta_omega:+.3e} rad/s | "
            f"scale={r['scale']:+.4f}  nrmse={r['nrmse']:.3e}  "
            f"corr={r['corr']:+.6f}"
        )
        ax.plot(r["hist_x"], r["v_resonator"], label="resonator induced V")
        ax.plot(
            r["hist_x"],
            r["scale"] * r["v_cavity_lab"],
            ls="--",
            label="cavity fine-grid (scaled)",
        )
        ax.set_title(
            f"delta_omega = {delta_omega:+.2e} rad/s   "
            f"(nrmse={r['nrmse']:.2e}, corr={r['corr']:+.4f})"
        )
        ax.set_ylabel("voltage [V]")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(
        "Single-turn fine-grid cavity response vs resonator induced voltage"
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    n_sections = 1
    bunch_obs_list, n_turns_list, ind_volt_obs_list, cav_fdbk_obs_list = (
        [],
        [],
        [],
        [],
    )

    n_turns = 4
    for MTW in [
        True,
        False,
    ]:
        (
            bunch_observation_buf,
            n_turns_buf,
            ind_volt_obs_list_buf,
            cav_fdbk_obs_list_buf,
        ) = setup_and_run(
            "RCS1",
            MTW=MTW,
            n_stations=n_sections,
            n_turns_in=n_turns,
            force_rematch=False,
            acceleration=False,
        )
        bunch_obs_list.append(bunch_observation_buf)
        n_turns_list.append(n_turns_buf)
        ind_volt_obs_list.append(ind_volt_obs_list_buf)
        cav_fdbk_obs_list.append(cav_fdbk_obs_list_buf)

    plot_ind_volt_cav_fdbk_voltage(
        ind_volt_obs_list,
        cav_fdbk_obs_list,
        n_turns_in=n_turns,
        n_stations=n_sections,
    )

    plot_results(bunch_obs_list, n_turns_list, ind_volt_obs_list)


# TODO: tests to write
"""
check no voltage change over one turn for no detuned cavity --> purely real generator current, should not change between turns

"""
