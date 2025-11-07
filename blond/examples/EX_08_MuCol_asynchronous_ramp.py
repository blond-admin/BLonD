from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import c

from blond import (
    Beam,
    BiGaussian,
    DriftSimple,
    MagneticCycleByTime,
    Ring,
    Simulation,
    SingleHarmonicCavity,
    StaticProfile,
    mu_plus,
)
from blond.physics.energy_reference_kick import ReferenceEnergyChange

if TYPE_CHECKING:
    pass


def main():
    N_TURNS = 17
    # calculate parameters
    transition_gamma = 1 / np.sqrt(10.395e-4)
    N_SECTIONS = 1
    VOLTAGE_PER_SECTION = 865 * 30e6 / N_SECTIONS
    TIME_PER_TURN = 953.338 * 2 * np.pi / c

    # define energy ramp
    ENERGY_RAMP = np.linspace(63e9, 313.83e9 * 100, N_TURNS)
    PHI_S = 135 * np.pi / 180

    # initiate ring
    ring = Ring(circumference=953.338 * 2 * np.pi)

    energy_cycle = MagneticCycleByTime(
        reference_particle=mu_plus,
        base_time=np.linspace(0, 18 * TIME_PER_TURN, N_TURNS),
        base_values=ENERGY_RAMP,
        in_unit="momentum",
    )

    N_CAVITIES = N_SECTIONS

    one_turn_model = []
    for cavity_i in range(N_CAVITIES):
        cavity = SingleHarmonicCavity(
            section_index=cavity_i,
        )
        profile = StaticProfile(
            cut_left=0, cut_right=1, n_bins=256, section_index=cavity_i
        )
        cavity.voltage = VOLTAGE_PER_SECTION
        cavity.phi_rf = PHI_S
        cavity.harmonic = 25900

        one_turn_model.extend(
            [
                cavity,
                DriftSimple(
                    transition_gamma=transition_gamma,
                    orbit_length=ring.circumference / N_SECTIONS / 3,
                    section_index=cavity_i,
                ),
                ReferenceEnergyChange(section_index=cavity_i),
                DriftSimple(
                    transition_gamma=transition_gamma,
                    orbit_length=ring.circumference / N_SECTIONS / 3,
                    section_index=cavity_i,
                ),
                ReferenceEnergyChange(section_index=cavity_i),
                DriftSimple(
                    transition_gamma=transition_gamma,
                    orbit_length=ring.circumference / N_SECTIONS / 3,
                    section_index=cavity_i,
                ),
                profile,
            ]
        )

    ring.add_elements(one_turn_model, reorder=False)
    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
    sim.print_one_turn_execution_order()

    beam1 = Beam(
        intensity=2.7e12,
        particle_type=mu_plus,
    )

    zmax = ring.circumference / (2 * 25900)

    total_cavity = SingleHarmonicCavity(
        section_index=cavity_i,
    )

    total_cavity.voltage = VOLTAGE_PER_SECTION
    total_cavity.phi_rf = PHI_S
    total_cavity.harmonic = 25900

    sim.prepare_beam(
        beam=beam1,
        preparation_routine=BiGaussian(
            sigma_dt=zmax / 43,
            reinsertion=False,
            seed=1,
            n_macroparticles=1e5,
        ),
    )

    sim.run_simulation(
        beams=(beam1,),
        turn_i_init=0,
        n_turns=N_TURNS,
    )

    return


if __name__ == "__main__":
    main()
