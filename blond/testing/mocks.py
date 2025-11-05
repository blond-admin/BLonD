"""Convenience functions for testing BLonD."""

from unittest.mock import Mock

from blond import Beam, DriftSimple, Simulation

simulation_mock = Mock(Simulation)

beam_mock = Mock(Beam)

drift_simple_mock = Mock(DriftSimple)
