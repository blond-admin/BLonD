"""Dev script, needs rework.

Authors
-------
Simon Lauber
"""

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    DriftSimple,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    backend,
)
from blond.core.beam.particle_types import ParticleType, c, e, m_p

backend.set_specials("cpp")


def main():
    """Runs a several small simulations to see if phi_s is calculated correctly.

    It is expected that the red line is at the stable point of the bunch.
    """
    splt_i = 0
    for charge in (-1, 1):
        for momentum_compaction_factor in (-1, 1):
            plt.subplot(2, 2, 1 + splt_i)
            plt.title(f"{charge=}\n{momentum_compaction_factor=}")
            test_particle = ParticleType(
                mass=m_p * c**2 / e,
                charge=charge,
            )
            ring = Ring(circumference=20e3)
            rf_station = SingleHarmonicRFStation(
                voltage=1e6, harmonic=10, phi_rf=np.deg2rad(-90)
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

            beam.setup_beam(
                dt=dt.flatten(),
                dE=dE.flatten(),
                reference_total_energy=test_particle.mass + 1e12,
            )
            drift = DriftSimple(
                orbit_length=ring.circumference,
                momentum_compaction_factor=momentum_compaction_factor,
            )
            print(f"{ring.transition_gamma=}")
            print(f"{drift.momentum_compaction_factor=}")
            print(f"{drift.eta_0(beam.reference.gamma)=}")

            ramp = np.linspace(
                beam.reference.total_energy,
                1.005 * beam.reference.total_energy,
                10000 + 1,
            )
            print(ramp[1] - ramp[0], "V")
            print(f"{beam.reference.beta=}")
            print(f"{beam.reference.gamma=}")
            cycle = MagneticCyclePerTurn(
                reference_particle=test_particle,
                value_init=float(ramp[0]),
                values_after_turn=ramp[1:],
                in_unit="total energy",
            )
            """cycle = ConstantMagneticCycle(reference_particle=test_particle,
                                          value= beam.reference.total_energy,
                                          in_unit="total energy")"""
            simulation = Simulation.from_locals(locals())
            simulation.print_one_turn_execution_order()

            def plot_beam(simulation, beam):
                if simulation.turn_i.value % 100 == 0:  # Every 100 turns
                    plt.figure(11)
                    plt.clf()
                    plt.scatter(beam.read_partial_dt(), beam.read_partial_dE())
                    plt.draw()
                    plt.pause(0.01)

            simulation.run_simulation(
                beams=(beam,),
                n_turns=10000,
                # callback=plot_beam,
            )

            beam.plot_hist2d(
                range=(
                    (0, ring.circumference / c / rf_station.harmonic),
                    (-0.5e9, 0.5e9),
                )
            )
            T_rev = cycle.get_t_rev_init(
                circumference=ring.circumference,
                particle_type=test_particle,
            )
            val = rf_station.phi_s / (2 * np.pi) * T_rev / rf_station.harmonic
            print(f"{rf_station.phi_s=}")
            print(f"{T_rev=}")
            print(f"{val=}")
            plt.axvline(
                val,
                color="red",
                zorder=10,
            )
            splt_i += 1
    # todo assertions
    plt.show()


if __name__ == "__main__":
    main()
