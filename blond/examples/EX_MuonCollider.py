import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# import psutil
from scipy.constants import pi

from blond import (
    Beam,
    DriftSimple,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicCavity,
    StaticProfile,
    WakeField,
    mu_minus,
    mu_plus,
)
from blond._core.backends.backend import Numpy32Bit, Numpy64Bit, backend
from blond.handle_results.observables import (
    BunchObservation_meta_params,
    CavityPhaseObservation,
    StaticMultiProfileObservation,
    StaticProfileObservation,
)
from blond.physics.impedances.solvers import (
    SingleTurnResonatorConvolutionSolver,
    TimeDomainFftSolver,
)
from blond.physics.impedances.sources import Resonators
from blond.specifics.muon_collider.beam_matching.beam_matching_rountine import (
    load_beam_data_counterrot_from_file,
)

# p = psutil.Process(os.getpid())
# p.cpu_affinity(([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]))


backend.change_backend(
    Numpy32Bit
)  # TODO: without these lines, it does not work, default should be set somewhere to be Numpy64bit python
backend.set_specials("numba")


# phi_s = 128 * pi / 180  # deg
# inj_energy = 63e9
# ejection_energy = 313.83e9
# n_turns = 17
# alpha_p = 4.68e-4
# Q_factor = 0.96e6
def setup_and_run(int_eff=False):
    # RCS2
    phi_s = 148 * pi / 180  # deg
    inj_energy = 313.83e9
    ejection_energy = 750e9
    n_turns = 56
    alpha_p = 8.986e-4
    Q_factor = 1.76e6
    bunch_intensity = 2.4e12
    n_stations = 8
    circumference = 5990
    harmonic = 25928
    voltage_per_cavity = 31140000.0

    energy_gain_per_turn = (ejection_energy - inj_energy) / n_turns
    total_voltage = energy_gain_per_turn / np.sin(phi_s)
    voltage_per_station = total_voltage / n_stations
    n_cavities = int(np.ceil(total_voltage / voltage_per_cavity))
    cav_per_station = n_cavities / n_stations

    R_over_Q = 518 if int_eff else 0
    gamma_transition = 1 / np.sqrt(alpha_p)

    ring = Ring(circumference=circumference)
    magnetic_cycle = MagneticCyclePerTurn(
        value_init=inj_energy,
        values_after_turn=np.linspace(
            inj_energy + energy_gain_per_turn, ejection_energy, n_turns
        ),
        in_unit="total energy",
        reference_particle=mu_plus,
    )
    one_turn_model = []
    profile_list = []
    t_rf = (
        magnetic_cycle.get_t_rev_init(
            ring.circumference,
            turn_i_init=0,
            t_init=0,
            particle_type=mu_plus,
        )
        / harmonic
    )
    for cavity_i in range(n_stations):
        profile_list.append(
            StaticProfile.from_rad(
                1e-10 * 2 * pi / t_rf,
                2 * np.pi,
                2**9,
                t_rf,
                section_index=cavity_i,
            )
        )  # very slight difference in linspaces of bin_centers
        local_res = Resonators(
            center_frequencies=1 / t_rf,
            quality_factors=Q_factor,
            shunt_impedances=R_over_Q * Q_factor * cav_per_station / 2,
        )  # FM only
        one_turn_model.extend(
            [
                DriftSimple(
                    transition_gamma=-gamma_transition,
                    orbit_length=circumference / n_stations / 2,
                    section_index=cavity_i,
                ),
                profile_list[-1],
                SingleHarmonicCavity(
                    voltage=voltage_per_station / 2,
                    phi_rf=0,
                    harmonic=harmonic,
                    local_wakefield=WakeField(
                        sources=(local_res,),
                        solver=SingleTurnResonatorConvolutionSolver(),
                        profile=profile_list[-1],
                    )
                    if int_eff
                    else None,
                    section_index=cavity_i,
                ),
                profile_list[-1],  # for CR beam
                DriftSimple(
                    transition_gamma=-gamma_transition,
                    orbit_length=circumference / n_stations / 2,
                    section_index=cavity_i,
                ),
                # profile_list[-1],
                # SingleHarmonicCavity(
                #     voltage=voltage_per_station / 2,
                #     phi_rf=0,
                #     harmonic=harmonic,
                #     local_wakefield=WakeField(
                #         sources=(local_res,),
                #         solver=SingleTurnResonatorConvolutionSolver(),
                #         profile=profile_list[-1],
                #     )
                #     if int_eff
                #     else None,
                #     section_index=cavity_i,
                # ),
                # profile_list[-1],  # for CR beam
            ]
        )
    ring.add_elements(one_turn_model, reorder=False)
    ####################################################################
    beam = Beam(
        intensity=bunch_intensity,
        particle_type=mu_plus,
        is_counter_rotating=False,
    )
    beam_CR = Beam(
        intensity=bunch_intensity,
        particle_type=mu_plus,
        is_counter_rotating=True,
    )
    sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    # sim.print_one_turn_execution_order()
    load_beam_data_counterrot_from_file(
        str(Path(__file__).parent) + r"/RCS2_8_stations_no_int_eff.npz"
        if not int_eff
        else str(Path(__file__).parent)
        + r"/RCS2_8_stations_ind_volt_time.npz",
        beam,
        beam_CR,
    )

    bunch_observation = BunchObservation_meta_params(
        each_turn_i=1, obs_per_turn=n_cavities, beam=beam
    )
    bunch_observation_CR = BunchObservation_meta_params(
        each_turn_i=1, obs_per_turn=n_cavities, beam=beam_CR
    )
    profile_observation = StaticProfileObservation(
        each_turn_i=1, obs_per_turn=1, profile=profile_list[-1]
    )
    # cavity_observation = CavityPhaseObservation(each_turn_i=1, cavity=ring.elements.get_element(SingleHarmonicCavity, 0))
    # multi_profile_observation = StaticMultiProfileObservation(
    #     each_turn_i=1,
    #     obs_per_turn=1,
    #     profiles=profile_list,
    #     beam=beam_CR,
    # )
    sim.run_simulation(
        beams=([beam, beam_CR]),
        turn_i_init=0,
        n_turns=None,
        observe=(
            bunch_observation,
            bunch_observation_CR,
            profile_observation,
            # cavity_observation,
            # multi_profile_observation,
        ),
    )

    return (
        bunch_observation,
        bunch_observation_CR,
        profile_observation,
    )  # , cavity_observation


def plot_and_compare(
    bunch_observation, bunch_observation_CR, profile_observation, int_eff=False
):
    json_filename = (
        "results_int_eff.json" if int_eff else "results_no_int_eff.json"
    )
    with open(json_filename, "r") as jfile:
        jdict = json.load(jfile)
    emittance_blond2 = np.array(jdict["rms_emittance"])
    bunch_length_blond2 = np.array(jdict["sigma_bl_rms"])
    bunch_centroid_blond2 = np.array(jdict["bunch_centroid"])
    energy_spread_blond2 = np.array(jdict["energy_spread_rms"])
    energy_centroid_blond2 = np.array(jdict["energy_mean"])

    # plt.title("bunch length")
    # plt.plot(bunch_observation.sigma_dt * 1e12)
    # # plt.plot(
    # #     bunch_observation_CR.turns_array,
    # #     bunch_observation_CR.sigma_dt * 1e12,
    # #     label="CR",
    # # )
    # plt.plot(
    #     bunch_length_blond2[1:] * 1e12,
    #     label="blond2",
    # )
    # plt.ylabel("bunch length [ps]")
    # plt.xlabel("turns ")
    # plt.legend()
    # plt.show()

    plt.title("bunch centroid")
    plt.plot(bunch_observation.mean_dt * 1e9)
    plt.plot(bunch_observation_CR.mean_dt * 1e9, label="CR")
    plt.plot(bunch_centroid_blond2[1:] * 1e9, label="blond2")
    plt.ylabel("bunch centroid [ns]")
    plt.legend()
    plt.show()

    # plt.title("energy spread")
    # plt.plot(bunch_observation.sigma_dE / 1e9)
    # # plt.plot(bunch_observation_CR.sigma_dE / 1e9, label="CR")
    # plt.plot(energy_spread_blond2[1:] / 1e9, label="blond2")
    # plt.ylabel("energy spread [GeV]")
    # plt.legend()
    # plt.show()

    plt.title("energy centroid")
    plt.plot(bunch_observation.mean_dE)
    plt.plot(bunch_observation_CR.mean_dE, label="CR")
    plt.plot(energy_centroid_blond2[1:], label="blond2")
    plt.legend()
    plt.show()

    plt.title("emittance")
    plt.plot(bunch_observation.emittance_stat, label="emittance")
    plt.plot(bunch_observation_CR.emittance_stat, label="CR")
    plt.plot(emittance_blond2[1:], label="blond2")
    plt.ylabel("statistical emittance (eVs)")
    plt.legend()
    plt.show()

    # plt.title("sigma t * sigma E")
    # plt.plot(bunch_observation.sigma_dE * bunch_observation.sigma_dt)
    # # plt.plot(
    # #     bunch_observation_CR.sigma_dE * bunch_observation_CR.sigma_dt, label="CR"
    # # )
    # plt.legend()
    # plt.show()

    # profiles = profile_observation.hist_y
    # turn_arr = profile_observation.turns_array
    #
    # n_printed = 0
    # for prof_ind, prof in enumerate(profiles):
    #     if n_printed > 10:
    #         break
    #     if np.sum(prof) != 0:
    #         n_printed += 1
    #         plt.plot(
    #             profile_list[-1].hist_x * 1e9, prof, label=f"profile@ {prof_ind}"
    #         )
    # plt.xlabel("profile time [ns]")
    # plt.legend()
    # plt.show()
    #
    # prof_ = multi_profile_observation.hist_y
    # for prof_ind, prof in enumerate(prof_):
    #     if np.sum(prof) != 0:
    #         plt.plot(prof, label=f"profile@ {prof_ind}")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    int_eff = True
    bunch_observation, bunch_observation_CR, profile_observation = (
        setup_and_run(int_eff=int_eff)
    )
    plot_and_compare(
        bunch_observation,
        bunch_observation_CR,
        profile_observation,
        int_eff=int_eff,
    )
