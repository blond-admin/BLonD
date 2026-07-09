# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Convenience functions for testing BLonD."""

from unittest.mock import Mock

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
)
from blond.core.reference_clock.reference_clock import ReferenceCoordinates

simulation_mock = Mock(Simulation)

beam_mock = Mock(Beam)
beam_mock.reference = Mock(ReferenceCoordinates)
beam_mock.is_counter_rotating = False
# Direction-signed charge (see BeamBaseClass.signed_charge_with_direction):
# computed at call time from the mock's current particle_type / direction, so
# tests may reassign either and stay consistent with the real semantics.
beam_mock.signed_charge_with_direction = lambda: (
    beam_mock.particle_type.charge * -1
    if beam_mock.is_counter_rotating
    else beam_mock.particle_type.charge
)
static_profile_mock = Mock(StaticProfile)
wakefield_profile_mock = Mock(WakeField)
cycle_const_mock = Mock(ConstantMagneticCycle)

drift_simple_mock = Mock(DriftSimple)
single_harmonic_rf_station_mock = Mock(SingleHarmonicRFStation)
