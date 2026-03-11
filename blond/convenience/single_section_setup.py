# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Convenience functions to setup basic simulations."""

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

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

    from blond.core.base import ScheduledBaseClass
    from blond.core.beam.particle_types import ParticleType
    from blond.cycles.magnetic_cycle import (
        MagneticCyclePerTurn,
        SynchronousDataTypes,
    )
    from blond.physics.impedances.base import WakeFieldSolver, WakeFieldSource


def single_section_simulation(  # noqa: PLR0912
    circumference: float,
    cycle: float | NumpyArray,
    cycle_unit: SynchronousDataTypes,
    particle_type: ParticleType,
    momentum_compaction_factor: float | ScheduledBaseClass,
    voltage: float | ScheduledBaseClass,
    phi_rf: float | ScheduledBaseClass,
    harmonic: float | ScheduledBaseClass,
    n_hamonmics: int,
    sources: tuple[WakeFieldSource, ...] | None = None,
    solver: WakeFieldSolver | None = None,
    cutoff_frequency: float | None = None,
    cycle_bending_radius: float | None = None,
):
    """
    Convenience function to setup a simulation.

    Parameters
    ----------
    circumference
        The reference circumference of the synchrotron, in [m].
        This value remains constant during simulation and is used to determine
        the RF frequency program. Note: While the actual orbit length may vary
        during simulation (e.g., due to energy changes), the circumference stays
        fixed. Orbit length changes result in timing delays but don't affect
        the RF frequency program.
    cycle
         Value(s) of the cycle in unit `in_unit`.
         This must be ``n_turns + 1`` values long.
    cycle_unit
        - 'momentum' [eV/c], (no conversion is done)
        - 'total energy' [eV],
        - 'kinetic energy' [eV], or
        - 'bending field' [T]
    particle_type
        Type of particles, e.g. protons.
    momentum_compaction_factor
        Momentum compaction factor.
    voltage
        RF station's effective voltage, in [V].
    phi_rf
        RF station's design phase, in [rad].
    harmonic
        RF station's design harmonic [].
    n_hamonmics
        Number of harmonics.
    sources
        Impedance sources.
    solver
        Solver to generate induced voltage from the `sources`.
    cutoff_frequency
        Cutoff frequency of the beam profile, in [Hz].
    cycle_bending_radius
        To 'bending field' associated bending radius, in [m].

    Returns
    -------
    simulation
        The `Simulation` object ready for beam matching and simulation.
    """
    assert n_hamonmics > 0, f"{n_hamonmics=}"
    if isinstance(cycle, float):
        _cycle = ConstantMagneticCycle(
            reference_particle=particle_type,
            value=cycle,
            in_unit=cycle_unit,
            bending_radius=cycle_bending_radius,
        )
    elif isinstance(cycle, np.ndarray):
        _cycle = MagneticCyclePerTurn.init_from_linspace(
            reference_particle=particle_type,
            values=cycle,
            in_unit=cycle_unit,
            bending_radius=cycle_bending_radius,
        )
    else:
        raise TypeError(type(cycle))

    ring = Ring(circumference=circumference)

    drift = DriftSimple(
        orbit_length=ring.closed_orbit_length,
    )
    if not isinstance(momentum_compaction_factor, ScheduledBaseClass):
        drift.momentum_compaction_factor = momentum_compaction_factor
    else:
        drift.schedule(
            "momentum_compaction_factor", momentum_compaction_factor
        )
    ring.add_element(drift)

    if n_hamonmics == 1:
        rf_station = SingleHarmonicRFStation()
    else:
        rf_station = MultiHarmonicRFStation(
            n_harmonics=n_hamonmics,
            main_harmonic_idx=0,
        )

    if not isinstance(voltage, ScheduledBaseClass):
        rf_station.voltage = voltage
    else:
        rf_station.schedule("voltage", voltage)

    if not isinstance(phi_rf, ScheduledBaseClass):
        rf_station.phi_rf = phi_rf
    else:
        rf_station.schedule("phi_rf", phi_rf)

    if not isinstance(harmonic, ScheduledBaseClass):
        rf_station.harmonic = harmonic
    else:
        rf_station.schedule("harmonic", harmonic)

    ring.add_element(rf_station)

    if sources is not None:
        assert solver is not None
        assert cutoff_frequency is not None
        profile = StaticProfile.from_cutoff(
            cut_left=0,
            cut_right=_cycle.get_t_rev_init(
                circumference=circumference, particle_type=particle_type
            )
            / rf_station.harmonic,
            cutoff_frequency=cutoff_frequency,
        )
        wakefield = WakeField(sources=sources, solver=solver, profile=profile)
        ring.add_element(wakefield)

    simulation = Simulation(
        ring=ring,
        magnetic_cycle=_cycle,
    )
    return simulation
