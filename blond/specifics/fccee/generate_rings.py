from typing import TYPE_CHECKING, Optional

import numpy as np

from blond import (
    Beam,
    BiGaussian,
    BunchObservation,
    CavityPhaseObservation,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicCavity,
    positron,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable
    from typing import Any

    from numpy.typing import NDArray as NumpyArray

    from blond._core.beam.particle_types import ParticleType
    from blond._core.ring.beam_physics_relevant_elements import (
        BeamPhysicsRelevantElements,
    )
    from blond._core.simulation.simulation import Simulation

    from ...physics.drifts import DriftBaseClass


def generate_FCCee_ring(
    operation_mode: str = "Z",
    particle: ParticleType = positron,
    beam_intensity: float = 2e11,
    n_turns: int = 2000,
    momentum: NumpyArray | None = None,
):
    # Parameters taken from the FCCee feasibility report of March 2025
    collider_circumference = 90.658509 * 1e3
    bending_radius = 10.021 * 1e3

    if operation_mode == "Z" or operation_mode == "ZZ":
        reference_energy = 45.6 * 1e9
        total_rf_voltage = 50 * 1e6
        momentum_compaction_factor = 28.6 * 1e-6
        radiation_integrals = None
    elif operation_mode == "W" or operation_mode == "WW":
        reference_energy = 80 * 1e9
        total_rf_voltage = 50 * 1e6
        momentum_compaction_factor = 28.6 * 1e-6
        radiation_integrals = None
    elif operation_mode == "H" or operation_mode == "ZH":
        reference_energy = 120 * 1e9
        total_rf_voltage = 50 * 1e6
        momentum_compaction_factor = 28.6 * 1e-6
        radiation_integrals = None
    elif operation_mode == "ttbar" or operation_mode == "tt":
        reference_energy = 182.5 * 1e9
        total_rf_voltage = 50 * 1e6
        momentum_compaction_factor = 28.6 * 1e-6
        radiation_integrals = None
    else:
        raise ValueError(
            "Operation mode not recognised. Expected Z, W, "
            f"ZH or ttbar and got {operation_mode}"
        )

    ring = Ring(
        circumference=collider_circumference,
        radiation_integrals=radiation_integrals,
    )
    drift = DriftSimple(
        orbit_length=collider_circumference,
        momentum_compaction_factor=momentum_compaction_factor,
    )

    # TODO: calculate initial total RF votlage required in the collider
    cavity = SingleHarmonicCavity(
        harmonic=121200, voltage=total_rf_voltage, phi_rf=0
    )
    ring.add_elements([cavity, drift])

    magnetic_cycle = ConstantMagneticCycle(
        value=reference_energy,
        in_unit="total energy",
        reference_particle=particle,
        bending_radius=bending_radius,
    )

    beam1 = Beam(intensity=1e9, particle_type=particle)

    return FCCee


def generate_HEB_ring(
    op_mode="Z",
    particle: ParticleType = positron,
    Nturns=2000,
    momentum=None,
    rad_int=None,
    n_sections=1,
):
    # Parameters taken from the Booster feasibility report (July 2024)
    C = 90.65874532 * 1e3
    rho = 10.021 * 1e3
    alpha_0 = 7.120435962 * 1e-6
    Emin = 20 * 1e9
    if rad_int is None:
        rad_int = np.array(
            [
                0.646747216157,
                0.0005936549319,
                5.6814536525e-08,
                5.92870407301e-09,
                1.698280783e-11,
            ]
        )

    if op_mode == "Z":
        E = 45.6 * 1e9
        U0 = 0.0391 * 1e9
        ex_coll = 0.71 * 1e-9
        ey_coll = 1.9 * 1e-12
    elif op_mode == "W" or op_mode == "WW":
        E = 80 * 1e9
        U0 = 0.374 * 1e9
        ex_coll = 2.17 * 1e-9
        ey_coll = 2.2 * 1e-12
    elif op_mode == "H" or op_mode == "ZH":
        E = 120 * 1e9
        U0 = 1.88 * 1e9
        ex_coll = 0.67 * 1e-9
        ey_coll = 1.0 * 1e-12
    elif op_mode == "ttbar" or op_mode == "tt":
        E = 182.5 * 1e9
        U0 = 10.29 * 1e9
        ex_coll = 1.57 * 1e-9
        ey_coll = 1.6 * 1e-12
    else:
        print("Unexpected operation mode. Please use Z, W, H, or ttbar.")
        return
    if momentum is None or len(momentum) != Nturns + 1:
        if momentum is None:
            print(f"Warning, no momentum provided.")
        else:
            print(
                f"Warning, len(momentum) != Nturns+1 == {len(momentum) != Nturns + 1}"
            )
        # pinj = np.sqrt(Emin ** 2 - particle.mass ** 2)/c
        # pext = np.sqrt(E ** 2 - particle.mass ** 2)/c
        momentum = np.linspace(
            Emin, E, Nturns + 1
        )  # momentum in Blond, but energy in reality
    if n_sections == 1:
        HEB = Ring(
            C,
            alpha_0,
            momentum,
            particle,
            Nturns,
            n_sections=n_sections,
            synchronous_data_type="total energy",
            bending_radius=rho,
            radiation_integrals=rad_int,
        )
    else:
        energy = []
        dE_per_turn = np.diff(momentum, axis=0)
        for i in range(n_sections):
            energy.append(
                momentum
                + i * np.append(dE_per_turn, dE_per_turn[-1]) / n_sections
            )
        # HEB = Ring(np.ones(n_sections) * C / n_sections, np.tile(alpha_0, (1, n_sections)).T, np.tile(momentum, (n_sections, 1)), particle, Nturns,  n_sections= n_sections, synchronous_data_type='total energy',bending_radius=rho, rad_int=rad_int)
        HEB = Ring(
            np.ones(n_sections) * C / n_sections,
            np.tile(alpha_0, (1, n_sections)).T,
            energy,
            particle,
            Nturns,
            n_sections=n_sections,
            synchronous_data_type="total energy",
            bending_radius=rho,
            radiation_integrals=rad_int,
        )
    return HEB


def generate_HEB_ring_RPO(
    op_mode="Z",
    particle=positron,
    Nturns=2000,
    momentum=None,
    rad_int=None,
    n_sections=1,
    ncav=112,
    DN=16,
    voltage_per_cavity=None,
):
    # Parameters taken from the Booster feasibility report (July 2024)
    C = 90.65874532 * 1e3
    rho = 10.021 * 1e3
    alpha_0 = 7.120435962 * 1e-6
    Emin = 20 * 1e9
    if rad_int is None:
        rad_int = np.array(
            [
                0.646747216157,
                0.0005936549319,
                5.6814536525e-08,
                5.92870407301e-09,
                1.698280783e-11,
            ]
        )

    if op_mode == "Z":
        E = 45.6 * 1e9
        U0 = 0.0391 * 1e9
        ex_coll = 0.71 * 1e-9
        ey_coll = 1.9 * 1e-12
    elif op_mode == "W" or op_mode == "WW":
        E = 80 * 1e9
        U0 = 0.374 * 1e9
        ex_coll = 2.17 * 1e-9
        ey_coll = 2.2 * 1e-12
    elif op_mode == "H" or op_mode == "ZH":
        E = 120 * 1e9
        U0 = 1.88 * 1e9
        ex_coll = 0.67 * 1e-9
        ey_coll = 1.0 * 1e-12
    elif op_mode == "ttbar" or op_mode == "tt":
        E = 182.5 * 1e9
        U0 = 10.29 * 1e9
        ex_coll = 1.57 * 1e-9
        ey_coll = 1.6 * 1e-12
    else:
        print("Unexpected operation mode. Please use Z, W, H, or ttbar.")
        return
    if momentum is None or len(momentum) != Nturns + 1:
        if momentum is None:
            print(f"Warning, no momentum provided.")
        else:
            print(
                f"Warning, len(momentum) != Nturns+1 == {len(momentum) != Nturns + 1}"
            )
        # pinj = np.sqrt(Emin ** 2 - particle.mass ** 2)/c
        # pext = np.sqrt(E ** 2 - particle.mass ** 2)/c
        momentum = np.linspace(
            Emin, E, Nturns + 1
        )  # momentum in Blond, but energy in reality
    if n_sections == 1:
        HEB = Ring(
            C,
            alpha_0,
            momentum,
            particle,
            Nturns,
            n_sections=n_sections,
            synchronous_data_type="total energy",
            bending_radius=rho,
            radiation_integrals=rad_int,
        )
    else:
        energy = []
        dE_per_turn = np.diff(momentum, axis=0)
        for i in range(n_sections):
            energy.append(
                momentum
                + i * np.append(dE_per_turn, dE_per_turn[-1]) / n_sections
            )
        HEB = Ring(
            np.ones(n_sections) * C / n_sections,
            np.tile(alpha_0, (1, n_sections)).T,
            energy,
            particle,
            Nturns,
            n_sections=n_sections,
            synchronous_data_type="total energy",
            bending_radius=rho,
            radiation_integrals=rad_int,
        )
    return HEB
