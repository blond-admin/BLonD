import numpy as np
from scipy.constants import c

from blond import (
    Beam,
    BiGaussian,
    DriftSimple,
    MagneticCycleByTime,
    Ring,
    Simulation,
    SingleHarmonicRfStation,
    StaticProfile,
    mu_plus,
)
from blond.physics.energy_reference_kick import ReferenceEnergyChange


def main():
    n_turns = 17
    # calculate parameters
    transition_gamma = 1 / np.sqrt(10.395e-4)
    n_sections = 1
    voltage_per_section = 865 * 30e6 / n_sections
    time_per_turn = 953.338 * 2 * np.pi / c

    # define energy ramp
    energy_ramp = np.linspace(63e9, 313.83e9 * 100, n_turns)
    phi_s = 135 * np.pi / 180

    # initiate ring
    ring = Ring(circumference=953.338 * 2 * np.pi)

    energy_cycle = MagneticCycleByTime(
        reference_particle=mu_plus,
        base_time=np.linspace(0, 18 * time_per_turn, n_turns),
        base_values=energy_ramp,
        in_unit="momentum",
    )

    one_turn_model = []
    for cavity_i in range(n_sections):
        cavity = SingleHarmonicRfStation(
            section_index=cavity_i,
        )
        profile = StaticProfile(
            cut_left=0, cut_right=1, n_bins=256, section_index=cavity_i
        )
        cavity.voltage = voltage_per_section
        cavity.phi_rf = phi_s
        cavity.harmonic = 25900

        one_turn_model.extend(
            [
                cavity,
                DriftSimple(
                    transition_gamma=transition_gamma,
                    orbit_length=ring.circumference / n_sections / 3,
                    section_index=cavity_i,
                ),
                ReferenceEnergyChange(section_index=cavity_i),
                DriftSimple(
                    transition_gamma=transition_gamma,
                    orbit_length=ring.circumference / n_sections / 3,
                    section_index=cavity_i,
                ),
                ReferenceEnergyChange(section_index=cavity_i),
                DriftSimple(
                    transition_gamma=transition_gamma,
                    orbit_length=ring.circumference / n_sections / 3,
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

    zmax = ring.circumference / (2 * 25900)  # maximum bunch length z

    total_cavity = SingleHarmonicRfStation(
        section_index=cavity_i,
    )

    total_cavity.voltage = voltage_per_section
    total_cavity.phi_rf = phi_s
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
        n_turns=n_turns,
    )

    return


if __name__ == "__main__":
    main()  # pragma: no cover
