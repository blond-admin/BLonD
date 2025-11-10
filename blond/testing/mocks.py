"""Convenience functions for testing BLonD."""

from unittest.mock import Mock

from blond import Beam, DriftSimple, Simulation, StaticProfile, WakeField

simulation_mock = Mock(Simulation)

beam_mock = Mock(Beam)
static_profile_mock = Mock(StaticProfile)
wakefield_profile_mock = Mock(WakeField)

drift_simple_mock = Mock(DriftSimple)
