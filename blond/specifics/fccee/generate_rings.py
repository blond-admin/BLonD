# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Basic simulation generation for the FCCee collider and high-energy booster."""

from __future__ import annotations

from typing import TYPE_CHECKING

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    positron,
)
from blond.physics.synchrotron_radiation.synchrotron_radiation_elements import (
    WigglerMagnet,
)
from blond.physics.synchrotron_radiation.synchrotron_radiation_master import (
    SynchrotronRadiationMaster,
)

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.particle_types import ParticleType


# TODO:SR integrals - once the optics are frozen
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
    collider_circumference = 90.658509e3
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
            f"Operation mode not recognised. Expected Z or ZZ, W or WW, "
            f"H or ZH, tt or ttbar and "
            f"got {operation_mode}"
        )

    ring = Ring(
        circumference=collider_circumference,
        synchrotron_radiation_integrals=radiation_integrals,
    )
    drift = DriftSimple(
        orbit_length=collider_circumference,
        momentum_compaction_factor=momentum_compaction_factor,
    )

    # TODO: calculate initial total RF voltage required in the collider
    cavity = SingleHarmonicRFStation(
        harmonic=121200, voltage=total_rf_voltage, phi_rf=0
    )
    ring.add_elements([cavity, drift])

    SRM = SynchrotronRadiationMaster()
    SRM.prepare_ring_for_synchrotron_radiation_tracking(ring=ring)

    magnetic_cycle = ConstantMagneticCycle(
        value=reference_energy,
        in_unit="total energy",
        reference_particle=particle,
        bending_radius=bending_radius,
    )

    beam = Beam(intensity=1e9, particle_type=particle)
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


def WigglerMagnetFCCee(
    number_of_wigglers: int = 2,
    section_index: int = 0,
) -> WigglerMagnet:
    """
    Damping wiggler magnets for the FCCee booster and collider.

    Parameters from the Feasibility Study Report.

    Parameters
    ----------
    number_of_wigglers
        Number of damping wigglers.
    section_index
        Section index.

    Returns
    -------
    damping_wiggler_magnet
        WigglerMagnet object.
    """
    return WigglerMagnet(
        name=f"DampingWiggler_{number_of_wigglers}",
        section_index=section_index,
        wiggler_type="sinusoidal",
        number_of_wigglers=1,
        peak_field=1.0,
        pole_length=0.095,
        number_of_poles=43,
    )
