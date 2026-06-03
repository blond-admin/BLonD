# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import os
import unittest

import numpy as np
import pytest

from blond.cycles.noise_generators import VariNoise
from blond.interfaces.rf_noise_cpp.wrap_rf_noise import (
    rf_noise_library_available,
)

# In CI the rf-noise-cpp library is provided, so library-backed tests must run
# (and fail loudly if it is missing) rather than silently skip. Locally, skip
# gracefully when the library is unavailable. GitLab sets ``CI=true``.
_RUN_RF_NOISE = os.environ.get("CI") == "true" or rf_noise_library_available()


#: Arbitrary, machine-agnostic playback frequency for tests, in [Hz].
_SAMPLING_RATE = 1.0e4


def _flat_spectrum(n: int = 16) -> np.ndarray:
    """A simple, machine-agnostic spectral shape for tests."""
    return np.ones(n, dtype=np.double)


class TestVariNoiseConfig(unittest.TestCase):
    """Configuration/validation behaviour that does not need the C++ library."""

    def _make(self, n_turns, **kwargs):
        kwargs.setdefault("gain_y", _flat_spectrum())
        kwargs.setdefault("sampling_rate", _SAMPLING_RATE)
        return VariNoise(
            frequency_high=np.full(n_turns, 200.0),
            frequency_low=np.full(n_turns, 100.0),
            **kwargs,
        )

    def test_gain_x_defaults_to_linspace(self):
        # gain_x defaults to linspace(0, 1, len(gain_y)) when not given.
        noise = self._make(10, gain_y=_flat_spectrum(8))

        self.assertEqual(len(noise.gain_x), len(noise.gain_y))
        self.assertAlmostEqual(float(noise.gain_x[0]), 0.0)
        self.assertAlmostEqual(float(noise.gain_x[-1]), 1.0)

    def test_get_noise_length_mismatch_raises(self):
        noise = self._make(10)
        with self.assertRaises(AssertionError):
            noise.get_noise(n_turns=11)

    def test_invalid_band_raises(self):
        # frequency_low must be strictly below frequency_high.
        n_turns = 10
        noise = VariNoise(
            frequency_high=np.full(n_turns, 100.0),
            frequency_low=np.full(n_turns, 200.0),
            gain_y=_flat_spectrum(),
            sampling_rate=_SAMPLING_RATE,
        )
        with self.assertRaises(AssertionError):
            noise.get_noise(n_turns=n_turns)

    def test_gain_x_out_of_range_raises(self):
        n_turns = 10
        noise = self._make(
            n_turns,
            gain_x=np.array([0.0, 0.5, 2.0]),  # 2.0 is out of [0, 1]
            gain_y=np.array([1.0, 1.0, 1.0]),
        )
        with self.assertRaises(AssertionError):
            noise.get_noise(n_turns=n_turns)


class TestVariNoiseLibrary(unittest.TestCase):
    """Behaviour that depends on the external C++ library."""

    def test_missing_library_raises_actionable_error(self):
        if rf_noise_library_available():
            self.skipTest("rf-noise-cpp library is available")
        n_turns = 10
        noise = VariNoise(
            frequency_high=np.full(n_turns, 200.0),
            frequency_low=np.full(n_turns, 100.0),
            gain_y=_flat_spectrum(),
            sampling_rate=_SAMPLING_RATE,
        )
        # Validation passes; failure comes from locating/building the library.
        with self.assertRaises((FileNotFoundError, RuntimeError)):
            noise.get_noise(n_turns=n_turns)

    @pytest.mark.skipif(
        not _RUN_RF_NOISE,
        reason="rf-noise-cpp library not available",
    )
    def test_get_noise_smoke(self):
        n_turns = 1024
        noise = VariNoise(
            frequency_high=np.full(n_turns, 200.0),
            frequency_low=np.full(n_turns, 100.0),
            gain_y=_flat_spectrum(),
            sampling_rate=_SAMPLING_RATE,
            r_seed=0,
            rms=1.0,
        )
        out = noise.get_noise(n_turns=n_turns)
        self.assertEqual(out.shape, (n_turns,))
        self.assertEqual(out.dtype, np.double)
        self.assertTrue(np.all(np.isfinite(out)))
