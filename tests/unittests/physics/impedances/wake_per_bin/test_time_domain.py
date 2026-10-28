# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import numpy as np

from blond import Resonators, backend
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.testing.backend_testing import BLonDTestCase


class TestImpedanceFromWakePerBin(BLonDTestCase):
    def setUp(self):
        self.n_bins = 16
        self.bin_step = 1e-10
        self.time = (
            backend.arange(self.n_bins, dtype=backend.float) * self.bin_step
        )
        self.source = Resonators(1e5, 200e6, 5.0)

    def _kernel(self, n_fft):
        impedance = self.source.get_impedance_from_wake(
            self.time, None, None, n_fft=n_fft
        )
        return copy_to_cpu(backend.fft.irfft(impedance, n=n_fft))

    def _reference(self):
        lags = backend.arange(-1, self.n_bins, dtype=backend.float)
        return copy_to_cpu(self.source.get_wake_per_bin(lags * self.bin_step))

    def test_kernel_reaches_the_last_bin(self):
        """Charge in the first bin must still reach the last one.

        The induced voltage of an ``n_bins`` profile needs the wake at lags
        0 to ``n_bins - 1``, plus the non-causal lag -1 of the bin average.
        """
        kernel = self._kernel(n_fft=2 * self.n_bins)
        reference = self._reference()
        scale = np.max(np.abs(reference))

        np.testing.assert_allclose(
            kernel[: self.n_bins], reference[1:], rtol=0, atol=1e-9 * scale
        )
        np.testing.assert_allclose(
            kernel[-1], reference[0], rtol=0, atol=1e-9 * scale
        )

    def test_kernel_does_not_wrap_onto_negative_lags(self):
        """Between lag ``n_bins - 1`` and lag -1 the kernel stays empty."""
        kernel = self._kernel(n_fft=2 * self.n_bins)
        scale = np.max(np.abs(self._reference()))

        np.testing.assert_allclose(
            kernel[self.n_bins : -1], 0.0, rtol=0, atol=1e-9 * scale
        )

    def test_without_n_fft_keeps_the_length_of_time(self):
        """``n_fft=None`` transforms exactly ``len(time)`` samples."""
        impedance = self.source.get_impedance_from_wake(
            self.time, None, None, n_fft=None
        )
        self.assertEqual(len(impedance), self.n_bins // 2 + 1)

    def test_cache_distinguishes_n_fft(self):
        """The same time axis with another ``n_fft`` is another spectrum."""
        short = self.source.get_impedance_from_wake(
            self.time, None, None, n_fft=2 * self.n_bins
        )
        long = self.source.get_impedance_from_wake(
            self.time, None, None, n_fft=4 * self.n_bins
        )
        self.assertEqual(len(short), self.n_bins + 1)
        self.assertEqual(len(long), 2 * self.n_bins + 1)
