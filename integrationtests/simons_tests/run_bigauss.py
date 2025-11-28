"""Dev script, needs rework.

Authors
-------
Simon Lauber
"""

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicRfStation,
    backend,
)
from blond.core.beam.particle_types import ParticleType, c, e, m_p

backend.set_specials("cpp")
interactive = False


def main():
    """Runs a several small simulations to see if phi_s is calculated correctly.

    It is expected that the red line is at the stable point of the bunch.
    """
    splt_i = 0
    cycle_const = True
    for charge in (-1, 1):
        for momentum_compaction_factor in (-1, 1):
            plt.subplot(2, 2, 1 + splt_i)
            plt.title(f"{charge=}\n{momentum_compaction_factor=}")
            test_particle = ParticleType(
                mass=m_p * c**2 / e,
                charge=charge,
            )
            ring = Ring(circumference=20e3)
            rf_station = SingleHarmonicRfStation(
                voltage=1e6, harmonic=10, phi_rf=np.deg2rad(45)
            )
            beam = Beam(
                intensity=12,
                particle_type=test_particle,
                is_counter_rotating=False,
            )
            dt, dE = np.meshgrid(
                np.linspace(0, ring.circumference / c, 512),
                np.linspace(-2e8, 2e8),
            )

            drift = DriftSimple(
                orbit_length=ring.circumference,
                momentum_compaction_factor=momentum_compaction_factor,
            )

            reference_total_energy = test_particle.mass + 1e12

            ramp = np.linspace(
                reference_total_energy,
                1.005 * reference_total_energy,
                10000 + 1,
            )
            print(ramp[1] - ramp[0], "V")
            if not cycle_const:
                cycle = MagneticCyclePerTurn(
                    reference_particle=test_particle,
                    value_init=float(ramp[0]),
                    values_after_turn=ramp[1:],
                    in_unit="total energy",
                )
            else:
                cycle = ConstantMagneticCycle(
                    reference_particle=test_particle,
                    value=reference_total_energy,
                    in_unit="total energy",
                )
                simulation = Simulation.from_locals(locals())
            simulation.print_one_turn_execution_order()
            simulation.prepare_beam(
                beam=beam,
                preparation_routine=BiGaussian(
                    n_macroparticles=1000, sigma_dt=1e-6 / 10
                ),
            )
            T_rev = cycle.get_t_rev_init(
                circumference=ring.circumference,
                turn_i_init=0,
                t_init=0,
                particle_type=test_particle,
            )

            def plot_beam(simulation, beam):
                if simulation.turn_i.value % 100 == 0:  # Every 100 turns
                    plt.figure(11)
                    plt.clf()
                    plt.scatter(beam.read_partial_dt(), beam.read_partial_dE())
                    val = (
                        rf_station.phi_s  # NOQA:  B023
                        / (2 * np.pi)
                        * T_rev  # NOQA:  B023
                        / rf_station.harmonic  # NOQA:  B023
                    )

                    plt.axvline(
                        val,
                        color="red",
                        zorder=10,
                    )
                    plt.draw()
                    plt.pause(0.01)

            simulation.run_simulation(
                beams=(beam,),
                n_turns=10000,
                callback=plot_beam if interactive else None,
            )

            beam.plot_hist2d(range=((-2.8e-6, 5e-6), (-5e7, 5e7)))
            val = rf_station.phi_s / (2 * np.pi) * T_rev / rf_station.harmonic

            plt.axvline(
                val,
                color="red",
                zorder=10,
            )
            print(f"{rf_station.phi_s=}")
            print(f"{T_rev=}")
            print(f"{val=}")

            splt_i += 1
    # todo assertions
    plt.show()


if __name__ == "__main__":
    main()
