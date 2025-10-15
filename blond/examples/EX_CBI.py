# pragma: no cover
import logging

import numpy as np
from matplotlib import pyplot as plt

from blond import (
    Beam,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicCavity,
    StaticProfile,
    StaticProfileObservation,
    WakeField,
    proton,
)
from blond._core.backends.backend import Numpy64Bit, backend
from blond.cycles.magnetic_cycle import MagneticCyclePerTurn
from blond.experimental.beam_preparation.empiric_matcher import EmpiricMatcher
from blond.handle_results.observables import (
    BunchObservation,
    BunchObservation_meta_params,
)
from blond.physics.impedances.solvers import MultiPassResonatorSolver
from blond.physics.impedances.sources import Resonators

backend.change_backend(Numpy64Bit)
backend.set_specials("numba")


def run_and_sim(N_TURNS: int):
    circumference = 17.933102862564866
    energy = 20364340099.9907  # eV
    n_slices = 2**8
    n_bunches = 12
    n_slices_profile = n_slices * n_bunches
    E_max = 202790499.59008813
    gamma_transition = 5.109102022523291

    f_rev = 16699550.578115981
    f_hom = 484286966.76536345
    Q_factor = 280.0
    R_shunt = 1e6

    decay_fraction_threshold = np.exp(-20 / f_rev / (Q_factor / f_hom / np.pi))

    ring = Ring(circumference)

    one_turn_model = []

    prof = StaticProfile.from_rad(
        0,
        2 * np.pi,
        n_slices_profile,
        1 / f_rev,
        section_index=0,
    )  # very slight difference in linspaces of bin_centers

    local_res = Resonators(
        center_frequencies=f_hom,
        quality_factors=Q_factor,
        shunt_impedances=R_shunt,
    )  # FM only
    wf_solver = MultiPassResonatorSolver(
        decay_fraction_threshold=decay_fraction_threshold
    )
    one_turn_model.extend(
        [
            prof,
            SingleHarmonicCavity(
                harmonic=12,
                voltage=6e6,
                phi_rf=0,
                local_wakefield=WakeField(
                    sources=(local_res,),
                    solver=wf_solver,
                    profile=prof,
                ),
            ),
            DriftSimple(
                transition_gamma=gamma_transition,
                orbit_length=circumference,
                section_index=0,
            ),
        ]
    )
    ring.add_elements(one_turn_model, reorder=False)

    energy_cycle = MagneticCyclePerTurn(
        value_init=energy,
        values_after_turn=np.linspace(energy, energy, N_TURNS),
        reference_particle=proton,
    )
    beam = Beam(
        intensity=12e9 * n_bunches,
        particle_type=proton,
    )

    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)

    sim.print_one_turn_execution_order()
    BIGAUS = True
    # if BIGAUS:
    #     sim.prepare_beam(
    #         beam=beam1,
    #         preparation_routine=BiGaussian(
    #             sigma_dt=0.4e-9 / 4,
    #             sigma_dE=1e9 / 4,
    #             reinsertion=False,
    #             seed=1,
    #             n_macroparticles=1e3,
    #         ),
    #     )
    # else:  # pragma: no cover
    sim.prepare_beam(
        beam=beam,
        preparation_routine=EmpiricMatcher(
            grid_base_dt=np.linspace(
                prof.cut_left, prof.cut_right, prof.n_bins
            ),
            grid_base_dE=np.linspace(-E_max * 2, E_max * 2, 500),
            n_macroparticles=1e6 * n_bunches,
            seed=0,
            maxiter_intensity_effects=10,
            animate=True,
        ),
    )

    prof.track(beam=beam)

    plt.plot(prof.hist_y)
    plt.show()

    bunch_observation = BunchObservation_meta_params(
        each_turn_i=1, obs_per_turn=1, beam=beam
    )

    profile_obs = StaticProfileObservation(
        each_turn_i=1, obs_per_turn=1, profile=prof
    )

    wf_solver._last_reference_time = -np.finfo(float).eps

    sim.run_simulation(
        beams=([beam]),
        turn_i_init=0,
        n_turns=N_TURNS,
        observe=(
            bunch_observation,
            profile_obs,
        ),
    )

    return bunch_observation, profile_obs


def plot_results(
    bunch_observation: BunchObservation_meta_params,
    profile_observation: StaticProfileObservation,
) -> None:
    plt.title("bunch length")
    plt.plot(bunch_observation.sigma_dt / 1e9)
    plt.xlabel("turns")
    plt.ylabel("bunch length [ns]")
    plt.show()

    plt.title("bunch centroid")
    plt.plot(bunch_observation.mean_dt)
    plt.xlabel("turns")
    plt.ylabel("bunch centroid [ns]")
    plt.show()

    plt.title("last profile")
    plt.plot(profile_observation.hist_y[-1])
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    N_turns = int(1e3)
    bunch_obs, profile_obs = run_and_sim(N_turns)

    plot_results(bunch_obs, profile_obs)
