"""Tests for injecting a magnetic cycle into a headless RF station."""

import unittest

from blond import SingleHarmonicRFStation, proton
from blond.cycles.magnetic_cycle import ExternalReferenceCycle


def _headless(**overrides):
    kwargs = dict(
        section_index=0,
        voltage=6e6,
        phi_rf=0.0,
        harmonic=35640,
        circumference=26658.883,
        beam_reference_beta=0.9999,
    )
    kwargs.update(overrides)
    return SingleHarmonicRFStation.headless(**kwargs)


def test_headless_uses_injected_magnetic_cycle():
    cycle = ExternalReferenceCycle(
        reference_particle=proton, total_energy_init=450e9
    )
    rf = _headless(magnetic_cycle=cycle)
    assert rf._magnetic_cycle is cycle


def test_injected_cycle_drives_reference_energy():
    cycle = ExternalReferenceCycle(
        reference_particle=proton, total_energy_init=450e9
    )
    rf = _headless(magnetic_cycle=cycle)
    cycle.set_total_energy(7000e9)
    assert (
        rf._magnetic_cycle.get_target_total_energy(0, 0, 0.0, proton) == 7000e9
    )


if __name__ == "__main__":
    unittest.main()
