# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


import numpy as np
from scipy.constants import e

from blond import StaticProfile, WakeField
from blond.core.backends.backend import backend
from blond.physics.impedances.multipole.solver import MultiPoleSparseSolve
from blond.physics.impedances.sources import Resonators
from blond.testing.backend_testing import BLonDTestCase


class TestMultiPoleSparseInitialise(BLonDTestCase):
    """Unit tests for `MultiPoleSparseSolve` on hand-made bin axes."""

    BIN_DT = 1e-10

    def _make_solver(self, hist_x_in_bins):
        from types import SimpleNamespace
        from unittest.mock import Mock

        solver = MultiPoleSparseSolve()
        parent = Mock(WakeField)
        parent.sources = (
            Resonators(
                shunt_impedances=[1.0],
                center_frequencies=[0.7e9],
                quality_factors=[11.0],
            ),
        )
        hist_x = backend.array(
            self.BIN_DT * np.asarray(hist_x_in_bins, dtype=float),
            dtype=backend.float,
        )
        profile = Mock(spec=StaticProfile)
        profile.hist_x = hist_x
        profile.hist_y = backend.ones(len(hist_x), dtype=backend.float)
        profile.hist_step = self.BIN_DT
        profile.hist_y_to_density_factor = 1.0
        parent.profile = profile
        solver._parent_wakefield = parent
        solver._profile = profile
        # charge * e * intensity == 1, so the histogram factor is -1
        beam = SimpleNamespace(
            reference=SimpleNamespace(time=0.0),
            particle_type=SimpleNamespace(charge=1.0),
            intensity=1.0 / e,
            is_counter_rotating=False,
        )
        return solver, beam

    def test_rejects_gap_of_a_fractional_number_of_bins(self):
        """The near/far split only holds for gaps of whole bins."""
        for gap_bins in (1.3, 1.7, 2.5):
            with self.subTest(gap_bins=gap_bins):
                hist_x = [0, 1, 2, 3] + [3 + gap_bins + i for i in range(4)]
                solver, beam = self._make_solver(hist_x)
                with self.assertRaisesRegex(ValueError, "whole number"):
                    solver.calc_induced_voltage(beam=beam)

    def test_accepts_gap_of_whole_bins(self):
        hist_x = [0, 1, 2, 3] + [3 + 4 + i for i in range(4)]
        solver, beam = self._make_solver(hist_x)
        voltage = solver.calc_induced_voltage(beam=beam)
        self.assertEqual(len(voltage), len(hist_x))

    def test_single_bin_profile(self):
        """One bin sees only its own tap, at lag zero."""
        from blond.generals.cupy_.no_cupy_import import copy_to_cpu
        from blond.physics.impedances.wake_per_bin.bin_average import (
            triple_box_average_poles,
        )

        solver, beam = self._make_solver([5.0])
        voltage = copy_to_cpu(solver.calc_induced_voltage(beam=beam))
        source = solver._parent_wakefield.sources[0]
        poles, residues, _ = source.get_vectorfit()
        tap_0 = copy_to_cpu(
            triple_box_average_poles(
                backend.array([0.0], dtype=backend.float),
                backend.array(poles, dtype=backend.complex),
                backend.array(residues, dtype=backend.complex),
                self.BIN_DT,
            )
        )
        np.testing.assert_allclose(voltage, -tap_0, rtol=1e-12)

    def test_thread_scratch_grows_with_the_active_backend(self):
        """A backend with more threads than at the first call still runs."""
        from unittest.mock import patch

        solver, beam = self._make_solver(range(8))
        solver.calc_induced_voltage(beam=beam)
        more_threads = backend.specials.get_max_threads() + 3
        beam.reference.time = 1e-6
        with patch.object(
            backend.specials, "get_max_threads", return_value=more_threads
        ):
            solver.calc_induced_voltage(beam=beam)
        self.assertGreaterEqual(
            solver._voltage_threaded.shape[0], more_threads
        )
