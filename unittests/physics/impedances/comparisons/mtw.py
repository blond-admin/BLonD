import os

import matplotlib.pyplot as plt
import numpy as np
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
    mu_plus,
)
from blond._core.backends.backend import Numpy64Bit, backend
from blond.handle_results.observables import (
    BunchObservation_meta_params,
    StaticProfileObservation,
)
from blond.legacy.blond2.beam.beam import Beam as beam_b2
from blond.legacy.blond2.beam.beam import MuPlus as mu_plus_b2
from blond.legacy.blond2.beam.distributions import (
    matched_from_distribution_function,
)
from blond.legacy.blond2.beam.profile import CutOptions as cut_options_b2
from blond.legacy.blond2.beam.profile import Profile as profile_b2
from blond.legacy.blond2.impedances.impedance import (
    InducedVoltageResonator as ind_volt_res_b2,
)
from blond.legacy.blond2.impedances.impedance import (
    InducedVoltageTime as ind_volt_time_b2,
)
from blond.legacy.blond2.impedances.impedance import (
    TotalInducedVoltage as total_ind_volt_b2,
)
from blond.legacy.blond2.impedances.impedance_sources import (
    Resonators as res_b2,
)
from blond.legacy.blond2.input_parameters.rf_parameters import (
    RFStation as rf_station_blond2,
)
from blond.legacy.blond2.input_parameters.ring import Ring as ring_b2
from blond.legacy.blond2.trackers.tracker import (
    FullRingAndRF,
    RingAndRFTracker,
)
from blond.physics.impedances.solvers import (
    MultiPassResonatorSolver,
    SingleTurnResonatorConvolutionSolver,
)
from blond.physics.impedances.sources import Resonators
from blond.specifics.muon_collider.beam_matching.beam_matching_rountine import (
    load_beam_data_counterrot_from_file,
)

backend.change_backend(
    Numpy64Bit
)  # TODO: without these lines, it does not work, default should be set somewhere to be Numpy64bit python
backend.set_specials("numba")

# RCS2
phi_s = 120 * pi / 180  # deg
inj_energy = 313.83e9
ejection_energy = 750e9
n_turns = 56
alpha_p = 8.986e-4
Q_factor = 1.76e6
bunch_intensity = 2.4e12
station_downscale = 80
circumference = 5990
harmonic = 25928
voltage_per_cavity = 31140000.0

energy_gain_per_turn = (
    (ejection_energy - inj_energy) / n_turns / station_downscale
)

n_turns_downscale = 2000
ejection_energy = inj_energy + n_turns_downscale * energy_gain_per_turn
total_voltage = energy_gain_per_turn / np.sin(phi_s)
voltage_per_station = total_voltage
n_cavities = int(np.ceil(total_voltage / voltage_per_cavity))
cav_per_station = n_cavities / station_downscale

R_over_Q = 518
gamma_transition = 1 / np.sqrt(alpha_p)

n_slices_profile = 2**9
emittance = 0.025 * 4 * np.pi
n_macroparticles = int(1e6)

decay_fraction_threshold = 0.01

from blond.legacy.blond2.utils import bmath as bm

bm.use_numba()
bm.use_precision("double")


def setup_and_run_blond3(mtw: bool = False):
    ring = Ring(circumference=circumference)
    magnetic_cycle = MagneticCyclePerTurn(
        value_init=inj_energy,
        values_after_turn=np.linspace(
            inj_energy + energy_gain_per_turn,
            ejection_energy,
            n_turns_downscale,
        ),
        in_unit="total energy",
        reference_particle=mu_plus,
    )
    one_turn_model = []
    t_rf = (
        magnetic_cycle.get_t_rev_init(
            ring.circumference,
            turn_i_init=0,
            t_init=0,
            particle_type=mu_plus,
        )
        / harmonic
    )
    prof = StaticProfile.from_rad(
        1e-10 * 2 * pi / t_rf,
        2 * np.pi,
        n_slices_profile,
        t_rf,
        section_index=0,
    )  # very slight difference in linspaces of bin_centers
    local_res = Resonators(
        center_frequencies=1 / t_rf,
        quality_factors=Q_factor,
        shunt_impedances=R_over_Q * Q_factor * cav_per_station,
    )  # FM only
    one_turn_model.extend(
        [
            prof,
            SingleHarmonicCavity(
                voltage=voltage_per_station,
                phi_rf=0,
                harmonic=harmonic,
                local_wakefield=WakeField(
                    sources=(local_res,),
                    solver=MultiPassResonatorSolver(
                        decay_fraction_threshold=decay_fraction_threshold
                    )
                    if mtw
                    else SingleTurnResonatorConvolutionSolver(),
                    profile=prof,
                ),
                section_index=0,
            ),
            DriftSimple(
                transition_gamma=-gamma_transition,
                orbit_length=circumference,
                section_index=0,
            ),
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
    load_filename = "initial_beam.npz"
    load_beam_data_counterrot_from_file(
        load_filename,
        beam,
        beam_CR,
    )

    bunch_observation = BunchObservation_meta_params(
        each_turn_i=1, obs_per_turn=1, beam=beam
    )
    profile_observation = StaticProfileObservation(
        each_turn_i=1, obs_per_turn=1, profile=prof
    )
    sim.run_simulation(
        beams=([beam]),
        turn_i_init=0,
        n_turns=n_turns_downscale,
        observe=(
            bunch_observation,
            profile_observation,
        ),
    )

    return (
        bunch_observation,
        profile_observation,
    )


def setup_and_run_blond2(mtw=False):
    energy = np.linspace(
        inj_energy, ejection_energy, n_turns_downscale + 1, endpoint=True
    )
    ring = ring_b2(
        circumference,
        alpha_p,
        energy,
        mu_plus_b2(),
        synchronous_data_type="total energy",
        n_turns=n_turns_downscale,
    )

    rf_station = rf_station_blond2(
        ring, harmonic, voltage_per_station, 0, n_rf=1, section_index=1
    )  # indexing is 1...n_stations

    beam = beam_b2(
        ring, n_macroparticles=n_macroparticles, intensity=bunch_intensity
    )
    cut_options = cut_options_b2(
        cut_left=1e-10,
        cut_right=1 / (rf_station.omega_rf[0, 0] / 2 / np.pi),
        n_slices=n_slices_profile,
    )

    profile = profile_b2(beam, cut_options=cut_options)
    profile.track()
    profile.fwhm()

    res_fund = res_b2(
        R_over_Q * Q_factor * cav_per_station,
        rf_station.omega_rf[0, 0] / 2 / np.pi,
        Q_factor,
    )

    ind_volt_res = ind_volt_res_b2(
        beam,
        profile,
        res_fund,
        rf_station=rf_station,
        multi_turn_wake=mtw,
        mtw_mode="time",
        time_decay_factor=decay_fraction_threshold,
    )

    total_ind_volt = total_ind_volt_b2(beam, profile, [ind_volt_res])
    total_ind_volt.induced_voltage_sum()

    long_tracker = RingAndRFTracker(
        rf_station,
        beam,
        profile=profile,
        total_induced_voltage=total_ind_volt,
        interpolation=True,
    )  # without interpolation no ind voltage
    full_ring_and_rf_tracker = FullRingAndRF([long_tracker])

    # if not os.path.exists("initial_beam.npz"):
    matching = matched_from_distribution_function(
        beam,
        full_ring_and_rf_tracker,
        n_iterations=10,
        TotalInducedVoltage=total_ind_volt,
        dt_margin_percent=0.01,
        seed=1234,
        distribution_exponent=2,
        distribution_type="binomial",
        emittance=2 * emittance,
        distribution_variable="Hamiltonian",
        process_pot_well=True,
        turn_number=0,
    )
    np.savez("initial_beam.npz", dt=beam.dt, dE=beam.dE, id=beam.id)
    # else:
    #     beam.dt = np.load("initial_beam.npz")["dt"]
    #     beam.dE = np.load("initial_beam.npz")["dE"]
    #     beam.id = np.load("initial_beam.npz")["id"]

    profile.track()

    save_bunch_centroid = []
    save_energy_centroid = []

    from tqdm import tqdm

    iterator = range(ring.n_turns)
    iterator = tqdm(iterator)

    for trn in iterator:
        profile.track()
        profile.fwhm()

        total_ind_volt.induced_voltage_sum()

        long_tracker.track()

        # statistics

        save_bunch_centroid.append(np.mean(beam.dt))
        save_energy_centroid.append(np.mean(beam.dE))

    return np.array(save_bunch_centroid), np.array(save_energy_centroid)


def plot_and_compare(
    bunch_observation,
    bunch_centroid_blond2,
    energy_centroid_blond2,
):
    plt.title("bunch centroid")
    plt.plot(bunch_observation.mean_dt * 1e9)
    plt.plot(bunch_centroid_blond2 * 1e9, label="blond2", ls="--")
    plt.ylabel("bunch centroid [ns]")
    plt.legend()
    plt.show()

    plt.title("energy centroid")
    plt.plot(bunch_observation.mean_dE)
    plt.plot(energy_centroid_blond2, label="blond2", ls="--")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    mtw = True
    bunch_centroid_b2, energy_centroid_b2 = setup_and_run_blond2(mtw=mtw)

    bunch_observation, profile_observation = setup_and_run_blond3(mtw=mtw)
    plot_and_compare(
        bunch_observation,
        bunch_centroid_b2,
        energy_centroid_b2,
    )
