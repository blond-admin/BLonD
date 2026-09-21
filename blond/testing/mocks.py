# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Convenience functions for testing BLonD."""

from unittest.mock import Mock

from blond.core.beam.beams import Beam
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.core.simulation.simulation import Simulation
from blond.cycles.magnetic_cycle import ConstantMagneticCycle
from blond.physics.drifts import DriftSimple
from blond.physics.impedances.base import WakeField
from blond.physics.profiles import StaticProfile
from blond.physics.rf_station import SingleHarmonicRFStation

simulation_mock = Mock(Simulation)

beam_mock = Mock(Beam)
beam_mock.reference = Mock(ReferenceCoordinates)
static_profile_mock = Mock(StaticProfile)
wakefield_profile_mock = Mock(WakeField)
cycle_const_mock = Mock(ConstantMagneticCycle)

drift_simple_mock = Mock(DriftSimple)
single_harmonic_rf_station_mock = Mock(SingleHarmonicRFStation)
