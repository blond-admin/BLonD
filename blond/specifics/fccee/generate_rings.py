"""Basic simulation generation for the FCCee collider and high-energy booster."""

from typing import TYPE_CHECKING

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRfStation,
    positron,
)

if TYPE_CHECKING:  # pragma: no cover
    from blond._core.beam.particle_types import ParticleType


def generate_fccee_collider_basic_simulation(
    operation_mode: str = "Z",
    particle: ParticleType = positron,
) -> Simulation:
    """
    Function to generate a basic simulation for the FCCee collider.

    Parameters
    ----------
    operation_mode
        Operation mode to simulate. Can be ZZ, WW, ZH or tt.
    particle
        ParticleType object.

    Returns
    -------
    fccee_simulation
        Basic FCCee collider Simulation object.
    """
    # Parameters taken from the FCCee feasibility report of March 2025
    collider_circumference = 90.658509 * 1e3
    bending_radius = 10.021 * 1e3

    if operation_mode in {"Z", "ZZ"}:
        reference_energy = 45.6 * 1e9
        total_rf_voltage = 50 * 1e6
        momentum_compaction_factor = 28.6 * 1e-6
        radiation_integrals = None
    elif operation_mode in {"W", "WW"}:
        reference_energy = 80 * 1e9
        total_rf_voltage = 50 * 1e6
        momentum_compaction_factor = 28.6 * 1e-6
        radiation_integrals = None
    elif operation_mode in {"H", "ZH"}:
        reference_energy = 120 * 1e9
        total_rf_voltage = 50 * 1e6
        momentum_compaction_factor = 28.6 * 1e-6
        radiation_integrals = None
    elif operation_mode in {"ttbar", "tt"}:
        reference_energy = 182.5 * 1e9
        total_rf_voltage = 50 * 1e6
        momentum_compaction_factor = 28.6 * 1e-6
        radiation_integrals = None
    else:
        raise ValueError(
            f"Operation mode not recognised. Expected ZZ, WW, ZH or tt and "
            f"got {operation_mode}"
        )

    ring = Ring(
        circumference=collider_circumference,
        radiation_integrals=radiation_integrals,
    )
    drift = DriftSimple(
        orbit_length=collider_circumference,
        momentum_compaction_factor=momentum_compaction_factor,
    )

    # TODO: calculate initial total RF voltage required in the collider
    cavity = SingleHarmonicRfStation(
        harmonic=121200, voltage=total_rf_voltage, phi_rf=0
    )
    ring.add_elements([cavity, drift])

    magnetic_cycle = ConstantMagneticCycle(
        value=reference_energy,
        in_unit="total energy",
        reference_particle=particle,
        bending_radius=bending_radius,
    )

    beam = Beam(intensity=1e9, particle_type=particle)
    # TODO verify output
    fccee_simulation = Simulation.from_locals(locals())
    fccee_simulation.print_one_turn_execution_order()

    fccee_simulation.prepare_beam(
        beam=beam,
        preparation_routine=BiGaussian(
            sigma_dt=0.4e-9 / 4,
            sigma_dE=1e9 / 4,
            reinsertion=False,
            seed=1,
            n_macroparticles=10,
        ),
        turn_i=1,
    )

    return fccee_simulation
