"""Verify phi_s is at the correct stable phase for all sign combinations.

For each combination of charge sign and momentum compaction sign, the
synchronous phase phi_s must satisfy the longitudinal stability condition:

    cos(phi_s) * charge * momentum_compaction_factor < 0

Set DEV_DRAW=True to run the full phase-space simulation and visualise the
stable fixed point together with the calculated phi_s (red vertical line).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

_DEV_DRAW = os.getenv("DEV_DRAW", "False").lower() == "true"


@pytest.mark.integration
def test_phi_s():
    from blond import (
        Beam,
        DriftSimple,
        MagneticCyclePerTurn,
        Ring,
        Simulation,
        SingleHarmonicRFStation,
    )
    from blond.core.beam.particle_types import ParticleType, c, e, m_p

    if _DEV_DRAW:
        plt.figure(figsize=(10, 10))

    splt_i = 0
    for charge in (-1, 1):
        for momentum_compaction_factor in (-1, 1):
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
            ramp = np.linspace(
                test_particle.mass + 1e12,
                1.005 * (test_particle.mass + 1e12),
                10000 + 1,
            )
            beam.setup_beam(
                dt=np.array([0.0]),
                dE=np.array([0.0]),
                reference_total_energy=float(ramp[0]),
            )
            drift = DriftSimple(
                orbit_length=ring.circumference,
                momentum_compaction_factor=momentum_compaction_factor,
            )
            cycle = MagneticCyclePerTurn(
                reference_particle=test_particle,
                value_init=float(ramp[0]),
                values_after_turn=ramp[1:],
                in_unit="total energy",
            )
            ring.add_elements(
                (rf_station, drift), reorder=False, section_index=0
            )
            simulation = Simulation(ring=ring, magnetic_cycle=cycle)

            phi_s = rf_station.calc_phi_s_main_harmonic(beam=beam)

            # Stability condition: cos(phi_s) * charge * alpha < 0
            assert np.isfinite(phi_s), (
                f"phi_s is not finite for {charge=}, {momentum_compaction_factor=}"
            )
            stability = np.cos(phi_s) * charge * momentum_compaction_factor
            assert stability < 0, (
                f"Stability condition violated for {charge=}, "
                f"{momentum_compaction_factor=}: cos(phi_s)*charge*alpha = "
                f"{stability:.4f} (expected < 0)"
            )

            if _DEV_DRAW:
                dt_grid, dE_grid = np.meshgrid(
                    np.linspace(0, ring.circumference / c, 512),
                    np.linspace(-2e8, 2e8),
                )
                beam_vis = Beam(
                    intensity=12,
                    particle_type=test_particle,
                    is_counter_rotating=False,
                )
                beam_vis.setup_beam(
                    dt=dt_grid.flatten(),
                    dE=dE_grid.flatten(),
                    reference_total_energy=float(ramp[0]),
                )
                simulation.run_simulation(beams=(beam_vis,), n_turns=10000)
                plt.subplot(2, 2, 1 + splt_i)
                plt.title(f"{charge=}\n{momentum_compaction_factor=}")
                beam_vis.plot_hist2d(
                    range=(
                        (0, ring.circumference / c / rf_station.harmonic),
                        (-0.5e9, 0.5e9),
                    )
                )
                T_rev = cycle.get_t_rev_init(
                    circumference=ring.circumference,
                    particle_type=test_particle,
                )
                val = phi_s / (2 * np.pi) * T_rev / rf_station.harmonic
                plt.axvline(val, color="red", zorder=10)

            splt_i += 1

    if _DEV_DRAW:
        plt.show()
