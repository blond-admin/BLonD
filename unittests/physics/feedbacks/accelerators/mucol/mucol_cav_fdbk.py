"""Muon collider RCS cavity-feedback simulation driver."""

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
    mu_plus,
)
from blond.cycles.magnetic_cycle import MagneticCyclePerTurnAllRFStations
from blond.examples.scripts.EX_08_Semi_empiric_matcher import (
    bucket_fill_by_emittance_gaussian,
)
from blond.experimental import SemiEmpiricMatcher
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.handle_results.observables import IQCavityFeedbackObservation
from blond.handle_results.observables_as_elements import (
    BeamObservationInRingElement,
    BunchObservationMetaParams,
    InducedVoltageObservationCR,
)
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass
from blond.physics.feedbacks.generator_current_controller import (
    GeneratorCurrentPIController,
)
from blond.physics.impedances.solvers import MultiPassResonatorSolver
from blond.specifics.muon_collider.beam_preparation import (
    load_beam_coordinates_from_file,
)

try:
    # Imported as part of the package (the dirs above ``mucol`` carry no
    # __init__.py, so this is the pytest/package context).
    from .plotting import (  # noqa: F401  (re-export)
        plot_generator_power_and_voltage,
        plot_ind_volt_cav_fdbk_voltage,
        plot_results,
    )
except ImportError:
    # Run directly as a script (``python mucol_cav_fdbk.py``): the script
    # directory is on sys.path, so the sibling module imports flat.
    from plotting import (  # noqa: F401  (re-export)
        plot_generator_power_and_voltage,
        plot_ind_volt_cav_fdbk_voltage,
        plot_results,
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
    use_pi_feedback: bool = False,
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
    use_pi_feedback
        If True, drive the cavity with a PI-regulated generator current
        (:class:`IQCavityFeedbackTimingClass` with non-zero PI gains, so the
        generator current reacts to the beam loading) instead of a constant
        generator current. Ignored when ``MTW`` is True.

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
        if MTW:
            cav_fdbk = None
        elif use_pi_feedback:
            # Per-step proportional loop gain g = K_p * (R/Q) * omega * dt
            # = 0.1 (omega * dt = 2 pi for one rf period per coarse sample);
            # integral ~30x slower. Same tuning as the unit tests.
            gain_proportional = 0.1 / (R_over_Q * 2 * np.pi)
            controller = GeneratorCurrentPIController(
                gain_proportional=gain_proportional,
                gain_integral=gain_proportional / (30 * t_rf),
                generator_current_bias=I_g,  # TODO: this should be bias
                n_delay=5,
            )
            cav_fdbk = IQCavityFeedbackTimingClass(
                profile=profile_list[-1],
                R_over_Q=R_over_Q,
                Q_L=Q_L,
                n_rf_periods_per_coarse_grid=1,
                generator_current_bias=I_g,
                n_cavities=cav_per_station,
                initial_voltage=voltage_per_cavity,
                delta_omega=delta_omega,
                controller=controller,
            )
        else:
            cav_fdbk = IQCavityFeedbackTimingClass(
                profile=profile_list[-1],
                R_over_Q=R_over_Q,
                Q_L=Q_L,
                n_rf_periods_per_coarse_grid=1,
                generator_current_bias=I_g,
                n_cavities=cav_per_station,
                initial_voltage=voltage_per_cavity,
                delta_omega=delta_omega,
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
    beam_spectrum = copy_to_cpu(
        profile_list[0].beam_spectrum(100 * profile_list[0].n_bins)
    )
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
