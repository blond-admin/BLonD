# coding: utf8
# Copyright 2014-2026 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for llrf.cavity_loop_kernels

:Authors: **Lina Valle**
"""

import unittest

import numpy as np

from blond.llrf import cavity_loop_kernels
from blond.llrf.impulse_response import cavity_response_sparse_matrix


class TestCavityResponseGap(unittest.TestCase):
    """The gap kernel against the fine-grid solve it replaces: interpolated
    generator current and no beam current through the gap."""

    def setUp(self):
        rng = np.random.default_rng(1)
        self.n_coarse = 1212
        self.t_rf = 1.2475e-9
        self.T_s = 20 * self.t_rf
        self.rf_centers = (np.arange(self.n_coarse) + 0.5) * self.T_s
        self.I_gen_coarse = (1 + 0.5j) + 0.1 * (
            rng.normal(size=self.n_coarse)
            + 1j * rng.normal(size=self.n_coarse)
        )
        self.omega_rf = 2 * np.pi / self.t_rf
        self.R_over_Q = 315.2
        self.Q_L = 1e7
        self.detuning = -2.3e-6
        self.V_init = 1e6 * (0.3 - 0.8j)
        self.I_gen_init = 0.9 + 0.4j
        self.I_beam_init = 0.02 - 0.01j

    def _compare(self, slices, start_bucket, n_gap):
        bin_size = self.t_rf / slices
        samples = self.omega_rf * bin_size
        t_init = start_bucket * self.t_rf

        gap_centers = t_init + bin_size * np.arange(1, n_gap + 1)
        I_gen_gap = np.interp(gap_centers, self.rf_centers, self.I_gen_coarse)
        V_gap = cavity_response_sparse_matrix(
            I_beam=np.concatenate(
                (np.array([self.I_beam_init]), np.zeros(n_gap, dtype=complex))
            ),
            I_gen=np.concatenate((np.array([self.I_gen_init]), I_gen_gap)),
            n_samples=n_gap,
            V_ant_init=self.V_init,
            I_gen_init=self.I_gen_init,
            samples_per_rf=samples,
            R_over_Q=self.R_over_Q,
            Q_L=self.Q_L,
            detuning=self.detuning,
        )

        V_end, I_gen_end = cavity_loop_kernels.cavity_response_gap(
            complex(self.V_init),
            complex(self.I_gen_init),
            complex(self.I_beam_init),
            t_init,
            bin_size,
            n_gap,
            self.rf_centers,
            self.I_gen_coarse,
            0.5 * self.R_over_Q * samples,
            complex(
                1 - 0.5 * samples / self.Q_L + 1j * self.detuning * samples
            ),
        )
        # identical recursion with the numba kernels; rounding of the
        # sparse solver otherwise (BLOND_DISABLE_NUMBA_KERNELS)
        np.testing.assert_allclose(V_end, V_gap[-1], rtol=1e-9, atol=0)
        np.testing.assert_allclose(
            I_gen_end, I_gen_gap[-1], rtol=1e-13, atol=0
        )

    def test_gap_inside_coarse_grid(self):
        self._compare(slices=32, start_bucket=80.97, n_gap=19 * 32 - 1)

    def test_gap_over_many_coarse_samples(self):
        self._compare(slices=32, start_bucket=80.97, n_gap=12000 * 32 - 1)

    def test_single_bin_gap(self):
        self._compare(slices=1000, start_bucket=80.999, n_gap=1)

    def test_gap_starting_before_coarse_grid(self):
        self._compare(slices=100, start_bucket=-30.5, n_gap=70 * 100)

    def test_gap_ending_after_coarse_grid(self):
        self._compare(slices=100, start_bucket=24200.99, n_gap=100 * 100)

    def test_gap_over_the_whole_coarse_grid(self):
        self._compare(slices=10, start_bucket=-5.0, n_gap=24260 * 10)


if __name__ == "__main__":
    unittest.main()
