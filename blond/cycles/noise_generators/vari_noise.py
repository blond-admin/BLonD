# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
RF phase-noise generator backed by CERN's external ``VariNoise`` library.

This is a thin wrapper around the ``rf-noise-cpp`` ctypes binding (see
:mod:`blond.interfaces.rf_noise_cpp.wrap_rf_noise`) exposed through the
:class:`~blond.cycles.noise_generators.base.NoiseGenerator` interface.

Author: Simon Lauber
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.cycles.noise_generators.base import NoiseGenerator
from blond.interfaces.rf_noise_cpp.wrap_rf_noise import rf_noise

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


class VariNoise(NoiseGenerator):
    """
    Band-limited RF phase noise from CERN's ``VariNoise`` generator.

    The generated noise occupies, at each turn, the frequency band between
    ``frequency_low`` and ``frequency_high``; its spectral shape inside the
    band is given by ``gain_x``/``gain_y`` and is constant along time.

    The underlying algorithm is the external CERN ``rf-noise-cpp`` library,
    which must be available (prebuilt, or buildable from source). It is loaded
    lazily on the first :meth:`get_noise` call, so importing this class never
    requires the library to be present.

    Parameters
    ----------
    frequency_high
        Array of the frequency upper limit along time, in [Hz]. Its length
        defines the number of turns produced by :meth:`get_noise`.
    frequency_low
        Array of the frequency lower limit along time, in [Hz]. Same length as
        ``frequency_high``; element-wise smaller than it.
    gain_y
        Spectral density distribution between the low and high limits, constant
        along time. Must be supplied by the caller (e.g. a machine-specific
        spectrum).
    sampling_rate
        Play-back clock frequency, in [Hz] (machine-specific, e.g. the
        revolution frequency). Required.
    gain_x
        Array from 0 (``frequency_low``) to 1 (``frequency_high``) giving the
        positions of ``gain_y``. Defaults to ``linspace(0, 1, len(gain_y))``.
    n_source
        Minimum number of elementary harmonic noise sources.
    n_pnt_min
        Minimum number of steps to express the highest-frequency oscillation.
    r_seed
        If ``< 0`` use a clock seed (non-reproducible); if ``>= 0`` use it as a
        reproducible starting seed.
    rms
        RMS value of the total time-domain output stream, in [rad].
    """

    def __init__(
        self,
        frequency_high: NumpyArray,
        frequency_low: NumpyArray,
        gain_y: NumpyArray,
        sampling_rate: float,
        gain_x: NumpyArray | None = None,
        n_source: int = 2048,
        n_pnt_min: int = 8,
        r_seed: int = 0,
        rms: float = 1.0,
    ) -> None:
        super().__init__()
        self.frequency_high = np.asarray(frequency_high, dtype=np.double)
        self.frequency_low = np.asarray(frequency_low, dtype=np.double)

        self.gain_y = np.asarray(gain_y, dtype=np.double)

        if gain_x is None:
            gain_x = np.linspace(0.0, 1.0, len(self.gain_y))
        self.gain_x = np.asarray(gain_x, dtype=np.double)

        self.n_source = n_source
        self.n_pnt_min = n_pnt_min
        self.r_seed = r_seed
        self.sampling_rate = sampling_rate
        self.rms = rms

    def get_noise(self, n_turns: int) -> NumpyArray:
        """
        Generate ``n_turns`` of RF phase noise.

        Parameters
        ----------
        n_turns
            Number of turns to generate noise for. Must equal the length of
            ``frequency_high``/``frequency_low``.

        Returns
        -------
        noise
            Phase-noise array of length ``n_turns``, in [rad].
        """
        assert len(self.frequency_high) == n_turns, (
            f"{len(self.frequency_high)=} must equal {n_turns=}"
        )
        return rf_noise(
            frequency_high=self.frequency_high,
            frequency_low=self.frequency_low,
            gain_x=self.gain_x,
            gain_y=self.gain_y,
            n_source=self.n_source,
            n_pnt_min=self.n_pnt_min,
            r_seed=self.r_seed,
            sampling_rate=self.sampling_rate,
            rms=self.rms,
        )
