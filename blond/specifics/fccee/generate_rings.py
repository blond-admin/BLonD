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

import numpy as np

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


def generate_fccee_booster_basic_simulation(
    operation_mode: str = "Z",
    particle: ParticleType = positron,
) -> Simulation:
    """
    Function to generate a basic simulation for the FCCee booster at 20 GeV.

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
    booster_circumference = 90.65874532 * 1e3
    bending_radius = 10.021 * 1e3
    radiation_integrals = np.array(
        [
            0.646747216157,
            0.0005936549319,
            5.6814536525e-08,
            5.92870407301e-09,
            1.698280783e-11,
        ]
    )
    if operation_mode in {"Z", "ZZ"}:
        injection_energy = 20 * 1e9
        extraction_energy = 45.6 * 1e9
        total_rf_voltage_injection = 50 * 1e6
        momentum_compaction_factor = 7.120435962 * 1e-6
    elif operation_mode in {"W", "WW"}:
        injection_energy = 20 * 1e9
        extraction_energy = 80 * 1e9
        total_rf_voltage_injection = 50 * 1e6
        momentum_compaction_factor = 7.120435962 * 1e-6
    elif operation_mode in {"H", "ZH"}:
        injection_energy = 20 * 1e9
        extraction_energy = 120 * 1e9
        total_rf_voltage_injection = 50 * 1e6
        momentum_compaction_factor = 7.120435962 * 1e-6
    elif operation_mode in {"ttbar", "tt"}:
        injection_energy = 20 * 1e9
        extraction_energy = 182.5 * 1e9
        total_rf_voltage_injection = 50 * 1e6
        momentum_compaction_factor = 7.120435962 * 1e-6
    else:
        raise ValueError(
            f"Operation mode not recognised. Expected Z or ZZ, W or WW, "
            f"H or ZH, tt or ttbar and "
            f"got {operation_mode}"
        )

    ring = Ring(
        circumference=booster_circumference,
        synchrotron_radiation_integrals=radiation_integrals,
    )
    drift = DriftSimple(
        orbit_length=booster_circumference,
        momentum_compaction_factor=momentum_compaction_factor,
    )

    # TODO: calculate initial total RF voltage required in the collider
    cavity = SingleHarmonicRFStation(
        harmonic=242400, voltage=total_rf_voltage_injection, phi_rf=0
    )
    SRM = SynchrotronRadiationMaster()

    magnetic_cycle = ConstantMagneticCycle(
        value=injection_energy,
        in_unit="total energy",
        reference_particle=particle,
        bending_radius=bending_radius,
    )

    beam = Beam(intensity=1e9, particle_type=particle)
    fccee_booster_simulation = Simulation.from_locals(locals())

    fccee_booster_simulation.prepare_beam(
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
    return fccee_booster_simulation


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
        number_of_wigglers=number_of_wigglers,
        peak_field=1.0,
        pole_length=0.095,
        number_of_poles=43,
    )
