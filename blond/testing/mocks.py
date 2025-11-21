"""Convenience functions for testing BLonD."""

from unittest.mock import Mock

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Simulation,
    SingleHarmonicRfStation,
    StaticProfile,
    WakeField,
)

simulation_mock = Mock(Simulation)

beam_mock = Mock(Beam)
static_profile_mock = Mock(StaticProfile)
wakefield_profile_mock = Mock(WakeField)
cycle_const_mock = Mock(ConstantMagneticCycle)

drift_simple_mock = Mock(DriftSimple)
single_harmonic_rf_station_mock = Mock(SingleHarmonicRfStation)
