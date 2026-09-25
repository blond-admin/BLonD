# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from unittest import mock

import numpy as np

from blond import Resonators, StaticProfile, WakeField, backend
from blond.physics.impedances.solvers import PeriodicFreqSolver
from blond.testing.backend_testing import BLonDTestCase


class TestWakeField(BLonDTestCase):
    def setUp(self):
        profile = StaticProfile(cut_left=0.0, cut_right=1e-6, n_bins=16)
        self.wakefield = WakeField(
            sources=(
                Resonators(np.array([1e4]), np.array([4e6]), np.array([3.0])),
            ),
            solver=PeriodicFreqSolver(t_periodicity=1e-6),
            profile=profile,
        )
        self.wakefield.track_profile = False
        self.wakefield.update_induced_voltage = False
        self.wakefield._induced_voltage = backend.linspace(
            0, 1, 16, dtype=backend.float
        )
        self.beam = mock.MagicMock()
        self.beam.signed_charge_with_direction.return_value = 1.0

    def test_track_kicks_with_induced_voltage_without_copying(self):
        """The induced voltage already has the backend dtype, so the
        per-turn kick must receive it as is, not a fresh copy.
        """
        with mock.patch.object(
            backend.specials, "kick_interpolated"
        ) as kick_interpolated:
            self.wakefield.track(beam=self.beam)

        voltage = kick_interpolated.call_args.kwargs["voltage"]
        self.assertIs(voltage, self.wakefield.induced_voltage)
