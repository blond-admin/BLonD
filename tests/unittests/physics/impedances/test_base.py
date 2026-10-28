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
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.impedances.base import (
    TimeDomain,
    TimeDomainCounterRotation,
)
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


class _RampCounterRotation(TimeDomainCounterRotation):
    """Counter-rotating source that keeps the default bin average."""

    def get_wake(self, time):
        return time

    def get_wake_counter_rotation(self, time):
        return 2.0 * time

    def get_impedance_from_wake_counter_rotation(
        self, time, simulation, beam, n_fft
    ):
        return self._impedance_from_wake_per_bin_counter_rotation(time, n_fft)


class TestTimeDomainCounterRotation(BLonDTestCase):
    def test_get_wake_per_bin_counter_rotation_default_is_the_stencil(self):
        """The default B-spline averages the counter-rotating wake.

        The symmetric stencil keeps a linear wake unchanged wherever all
        five taps lie inside the samples, and it averages
        ``get_wake_counter_rotation``, not ``get_wake``.
        """
        time = backend.arange(8, dtype=backend.float)

        wake_per_bin = copy_to_cpu(
            _RampCounterRotation().get_wake_per_bin_counter_rotation(time)
        )

        np.testing.assert_allclose(
            wake_per_bin[2:-2], 2.0 * np.arange(2, 6), rtol=1e-12
        )


class TestTimeDomain(BLonDTestCase):
    def test_subclass_without_impedance_from_wake_is_abstract(self):
        """A source must say how it turns into an impedance.

        Leaving it out must fail when the source is built, not at the
        first turn of a simulation.
        """

        class Empty(TimeDomain):
            pass

        with self.assertRaises(TypeError):
            Empty()

    def test_counter_rotation_without_impedance_is_abstract(self):
        """The same holds for the counter-rotating impedance."""

        class WakesOnly(TimeDomainCounterRotation):
            def get_wake(self, time):
                return time

            def get_wake_counter_rotation(self, time):
                return time

        with self.assertRaises(TypeError):
            WakesOnly()
