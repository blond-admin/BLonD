"""Tests for the externally-driven reference energy cycle."""

from blond import proton
from blond.cycles.magnetic_cycle import (
    ExternalReferenceCycle,
    MagneticCycleBase,
)


def test_is_a_magnetic_cycle():
    cycle = ExternalReferenceCycle(
        reference_particle=proton, total_energy_init=450e9
    )
    assert isinstance(cycle, MagneticCycleBase)


def test_returns_initial_energy():
    cycle = ExternalReferenceCycle(
        reference_particle=proton, total_energy_init=450e9
    )
    energy = cycle.get_target_total_energy(
        turn_i=0, section_i=0, reference_time=0.0, particle_type=proton
    )
    assert energy == 450e9


def test_set_total_energy_overrides_value():
    cycle = ExternalReferenceCycle(
        reference_particle=proton, total_energy_init=450e9
    )
    cycle.set_total_energy(7000e9)
    energy = cycle.get_target_total_energy(
        turn_i=5, section_i=0, reference_time=1.0, particle_type=proton
    )
    assert energy == 7000e9
