from os import PathLike
from pathlib import Path

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
    mu_minus,
    mu_plus,
)
from blond._core.backends.backend import Numpy64Bit, backend
from blond.handle_results.observables import (
    BunchObservation_meta_params,
    StaticProfileObservation,
)
from blond.physics.impedances.solvers import (
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


# phi_s = 128 * pi / 180  # deg
# inj_energy = 63e9
# ejection_energy = 313.83e9
# n_turns = 17
# alpha_p = 4.68e-4
# Q_factor = 0.96e6

# RCS2
phi_s = 148 * pi / 180  # deg
inj_energy = 313.83e9
ejection_energy = 750e9
n_turns = 55
alpha_p = 11.4e-4
Q_factor = 0.775e6

energy_gain_per_turn = (ejection_energy - inj_energy) / n_turns
total_voltage = energy_gain_per_turn / np.sin(phi_s)  # different from BlonD2
n_cavities = 8

R_over_Q = 518
gamma_transition = 1 / np.sqrt(alpha_p)
circumference = 5990
harmonic = 25900

ring = Ring(circumference=circumference)
magnetic_cycle = MagneticCyclePerTurn(
    value_init=inj_energy,
    values_after_turn=np.linspace(
        inj_energy + energy_gain_per_turn, ejection_energy, n_turns
    ),
    in_unit="kinetic energy",
    reference_particle=mu_plus,
)
one_turn_model = []
for cavity_i in range(n_cavities):
    profile_tmp = StaticProfile.from_rad(  # todo inside for loop?
        -np.pi,
        np.pi,
        2**10,
        magnetic_cycle.get_t_rev_init(
            ring.circumference,
            turn_i_init=0,
            t_init=0,
            particle_type=mu_plus,
        )
        / harmonic,
        section_index=cavity_i,
    )
    # TODO: adjust center frequency to harmonic
    local_res = Resonators(
        center_frequencies=1.3e9,
        quality_factors=Q_factor,
        shunt_impedances=R_over_Q * Q_factor,
    )  # FM only
    one_turn_model.extend(
        [
            profile_tmp,
            SingleHarmonicCavity(
                voltage=total_voltage / n_cavities,
                phi_rf=0,
                harmonic=harmonic,
                local_wakefield=WakeField(
                    sources=(local_res,),
                    solver=SingleTurnResonatorConvolutionSolver(),
                    profile=profile_tmp,
                ),
                section_index=cavity_i,
            ),
            DriftSimple(
                transition_gamma=gamma_transition,
                orbit_length=circumference / n_cavities,
                section_index=cavity_i,
            ),
        ]
    )
ring.add_elements(one_turn_model, reorder=False)
####################################################################
beam = Beam(
    n_particles=2.7e12,
    particle_type=mu_plus,
    is_counter_rotating=False,
)
beam_CR = Beam(
    n_particles=2.7e12,
    particle_type=mu_minus,
    is_counter_rotating=True,
)
sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
load_beam_data_counterrot_from_file(
    str(Path(__file__).parent) + r"\\RCS2_8_cavities.npz", beam, beam_CR
)

bunch_observation = BunchObservation_meta_params(
    each_turn_i=1, obs_per_turn=n_cavities, beam=beam
)
profile_observation = StaticProfileObservation(
    each_turn_i=1, obs_per_turn=n_cavities, profile=profile_tmp, beam=beam
)
# wakefield_observation = WakeFieldObservation(each_turn_i=1, obs_per_turn = 4)
sim.run_simulation(
    beams=(beam, beam_CR),
    turn_i_init=0,
    n_turns=n_turns,
    observe=[bunch_observation, profile_observation],
)

plt.title("bunch length")
plt.plot(bunch_observation.sigma_dt)
plt.show()

plt.title("bunch centroid")
plt.plot(bunch_observation.mean_dt)
plt.show()

plt.title("energy length")
plt.plot(bunch_observation.sigma_dE)
plt.show()

plt.title("energy centroid")
plt.plot(bunch_observation.mean_dE)
plt.show()

plt.title("emittance")
plt.plot(bunch_observation.emittance_stat, label="emittance")
plt.show()
plt.title("sigma t * sigma E")
plt.plot(bunch_observation.sigma_dE * bunch_observation.sigma_dt)
plt.show()

profiles = profile_observation.hist_y
turn_arr = profile_observation.turns_array

for prof_ind, prof in enumerate(profiles):
    if np.sum(prof) != 0:
        plt.plot(prof, label=f"profile@ {prof_ind}")
plt.legend()
plt.show()

pass
