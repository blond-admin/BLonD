# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Convenience functions to setup basic simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond import (
    ConstantMagneticCycle,
    DriftSimple,
    MultiHarmonicRFStation,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
)
from blond.core.base import ScheduledBaseClass
from blond.cycles.magnetic_cycle import (
    MagneticCyclePerTurn,
)

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.particle_types import ParticleType
    from blond.cycles.magnetic_cycle import (
        SynchronousDataTypes,
    )
    from blond.physics.impedances.base import WakeFieldSolver, WakeFieldSource


def single_section_simulation(  # noqa: PLR0912
    ring_circumference: float,
    cycle_values: float | NumpyArray,
    cycle_unit: SynchronousDataTypes,
    particle_type: ParticleType,
    ring_momentum_compaction_factor: float | ScheduledBaseClass,
    cavity_voltage: float | NumpyArray | ScheduledBaseClass,
    cavity_phi_rf: float | NumpyArray | ScheduledBaseClass,
    cavity_harmonic: float | NumpyArray | ScheduledBaseClass,
    cavity_n_harmonics: int,
    wakefield_impedance_sources: tuple[WakeFieldSource, ...] | None = None,
    wakefield_solver: WakeFieldSolver | None = None,
    wakefield_cutoff_frequency: float | None = None,
    cycle_bending_radius: float | None = None,
):
    """
    Convenience function to setup a simulation.

    Parameters
    ----------
    ring_circumference
        The reference circumference of the synchrotron, in [m].
        This value remains constant during simulation and is used to determine
        the RF frequency program. Note: While the actual orbit length may vary
        during simulation (e.g., due to energy changes), the circumference stays
        fixed. Orbit length changes result in timing delays but don't affect
        the RF frequency program.
    cycle_values
         Value(s) of the cycle in unit `in_unit`.
         This must be ``n_turns + 1`` values long.
    cycle_unit
        - 'momentum' [eV/c], (no conversion is done)
        - 'total energy' [eV],
        - 'kinetic energy' [eV], or
        - 'bending field' [T]
    particle_type
        Type of particles, e.g. protons.
    ring_momentum_compaction_factor
        Momentum compaction factor.
    cavity_voltage
        RF station's effective voltage, in [V].
    cavity_phi_rf
        RF station's design phase, in [rad].
    cavity_harmonic
        RF station's design harmonic [].
    cavity_n_harmonics
        Number of harmonics.
    wakefield_impedance_sources
        Impedance sources.
    wakefield_solver
        Solver to generate induced voltage from the `sources`.
    wakefield_cutoff_frequency
        Cutoff frequency of the beam profile, in [Hz].
    cycle_bending_radius
        To 'bending field' associated bending radius, in [m].

    Returns
    -------
    simulation
        The `Simulation` object ready for beam matching and simulation.
    """
    assert cavity_n_harmonics > 0, f"{cavity_n_harmonics=}"
    if isinstance(cycle_values, float):
        _cycle = ConstantMagneticCycle(
            reference_particle=particle_type,
            value=cycle_values,
            in_unit=cycle_unit,
            bending_radius=cycle_bending_radius,
        )
    elif isinstance(cycle_values, np.ndarray):
        _cycle = MagneticCyclePerTurn.init_from_linspace(
            reference_particle=particle_type,
            values=cycle_values,
            in_unit=cycle_unit,
            bending_radius=cycle_bending_radius,
        )
    else:
        raise TypeError(type(cycle_values))

    ring = Ring(circumference=ring_circumference)

    drift = DriftSimple(
        orbit_length=ring.closed_orbit_length,
    )
    if not isinstance(ring_momentum_compaction_factor, ScheduledBaseClass):
        drift.momentum_compaction_factor = ring_momentum_compaction_factor
    else:
        drift.schedule(
            "momentum_compaction_factor", ring_momentum_compaction_factor
        )
    ring.add_element(drift)

    if cavity_n_harmonics == 1:
        rf_station = SingleHarmonicRFStation()
    else:
        rf_station = MultiHarmonicRFStation(
            n_harmonics=cavity_n_harmonics,
            main_harmonic_idx=0,
        )

    if not isinstance(cavity_voltage, ScheduledBaseClass):
        rf_station.voltage = cavity_voltage
    else:
        rf_station.schedule("voltage", cavity_voltage)

    if not isinstance(cavity_phi_rf, ScheduledBaseClass):
        rf_station.phi_rf_design = cavity_phi_rf
    else:
        rf_station.schedule("phi_rf_design", cavity_phi_rf)

    if not isinstance(cavity_harmonic, ScheduledBaseClass):
        rf_station.harmonic = cavity_harmonic
    else:
        rf_station.schedule("harmonic", cavity_harmonic)

    ring.add_element(rf_station)

    if wakefield_impedance_sources is not None:
        assert wakefield_solver is not None, (
            "`wakefield_solver` must be given when using impedances."
        )
        assert wakefield_cutoff_frequency is not None, (
            "`wakefield_cutoff_frequency` must be given when using impedances."
        )
        profile = StaticProfile.from_cutoff(
            cut_left=0,
            cut_right=_cycle.get_t_rev_init(
                circumference=ring_circumference, particle_type=particle_type
            )
            / rf_station.harmonic,
            cutoff_frequency=wakefield_cutoff_frequency,
        )
        wakefield = WakeField(
            sources=wakefield_impedance_sources,
            solver=wakefield_solver,
            profile=profile,
        )
        ring.add_element(wakefield)

    simulation = Simulation(
        ring=ring,
        magnetic_cycle=_cycle,
    )
    return simulation
