# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


from unittest.mock import patch

import numpy as np

from blond.core.backends.backend import backend
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.impedances.wake_per_bin.bin_average import (
    triple_box_average_pole,
)
from blond.testing.backend_testing import BLonDTestCase


class TestIsFarFieldPole(BLonDTestCase):
    """The far-field selection evaluates all poles' near taps at once."""

    def setUp(self) -> None:
        from blond.physics.impedances.multipole import model as pole_model

        self.pole_model = pole_model
        bin_dt = 1e-10
        rng = np.random.default_rng(5)
        n_poles = 40
        is_real = np.arange(n_poles) % 4 == 1
        # damping from a fraction of a bin to thousands of bins, so the
        # selection keeps some poles and drops others
        poles = -np.logspace(6, 11, n_poles) + 1j * np.where(
            is_real, 0.0, rng.uniform(1e8, 9e9, n_poles)
        )
        residues = rng.normal(size=n_poles) + 1j * np.where(
            is_real, 0.0, rng.normal(size=n_poles)
        )
        poles = backend.array(poles, dtype=backend.complex)
        residues = backend.array(residues, dtype=backend.complex)
        pole_dt = poles * bin_dt
        self.model = pole_model._PoleModel(
            poles=poles,
            residues=residues,
            counterrotating_pole_signs=backend.ones(n_poles),
            pair_factors=2.0 - (poles.imag == 0),
            tail_residues=(
                residues
                * (backend.expm1(pole_dt) / pole_dt) ** 3
                * backend.exp(pole_dt / 2.0)
            ),
            taps=backend.zeros(3),
            adjacent=backend.zeros(0, dtype=bool),
            bin_dt=bin_dt,
            span_dt=1e-6,
        )
        self.tolerance = 1e-3

    def _per_pole_reference(self) -> np.ndarray:
        model = self.model
        lags = backend.array(
            [model.bin_dt, 0.0, -model.bin_dt], dtype=backend.float
        )
        near_field_peaks = np.empty(len(model.poles))
        for pole_i in range(len(model.poles)):
            pole = model.poles[pole_i : pole_i + 1]
            residue = model.residues[pole_i : pole_i + 1]
            in_phase = triple_box_average_pole(
                lags, pole, residue, model.bin_dt
            )
            quadrature = triple_box_average_pole(
                lags, pole, -1j * residue, model.bin_dt
            )
            near_field_peaks[pole_i] = np.max(
                np.hypot(copy_to_cpu(in_phase), copy_to_cpu(quadrature))
            )
        decay_per_bin = copy_to_cpu(
            backend.exp(model.poles.real * model.bin_dt)
        )
        tail_at_two_bins = copy_to_cpu(
            model.pair_factors * backend.abs(model.tail_residues)
        )
        return tail_at_two_bins >= (
            self.tolerance * near_field_peaks * (1.0 - decay_per_bin)
        )

    def test_matches_per_pole_selection(self) -> None:
        expected = self._per_pole_reference()
        # the case is only meaningful if it keeps some poles and drops some
        self.assertTrue(expected.any())
        self.assertFalse(expected.all())
        result = self.pole_model._is_far_field_pole(self.model, self.tolerance)
        np.testing.assert_array_equal(result, expected)

    def test_near_taps_of_all_poles_in_two_calls(self) -> None:
        with patch.object(
            self.pole_model,
            "triple_box_average_pole",
            wraps=self.pole_model.triple_box_average_pole,
        ) as single_pole:
            self.pole_model._is_far_field_pole(self.model, self.tolerance)
        # in phase and in quadrature, each for every pole at once
        self.assertEqual(single_pole.call_count, 2)
