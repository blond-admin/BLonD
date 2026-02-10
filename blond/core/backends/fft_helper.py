# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Holds `RepeatedFftHelper`."""

from blond import backend
from blond.core.backends.backend import CupyBackend, NumpyBackend


class RepeatedFftHelper:
    """
    Helper class to repeatedly FFT the same array size.

    This class is only written to allow an backend agnostic
    access to the `rfft` and `irfft` routine, taking advantage
    of `out` as a buffer. At the time writing, this feature
    did not exist in Cupy.
    """

    def __init__(self):
        self.out_buffer = None

    def rfft(self, a):
        """
        Calculate the real-to-complex fourier transform.

        Parameters
        ----------
        a
            Input array to be transformed.

        Returns
        -------
        a_rfft
            The Fourier transform of `a`.
        """
        if isinstance(backend, NumpyBackend):
            if self.out_buffer is None:
                self.out_buffer = backend.rfft_parallel(a=a)
            else:
                backend.rfft_parallel(a=a, out=self.out_buffer)
            result = self.out_buffer
        elif isinstance(backend, CupyBackend):
            result = backend.rfft_parallel(a=a)  # out not supported (2025)
        else:
            raise RuntimeError(f"Unknown {type(backend)=}")
        return result

    def irfft(self, a):
        """
        Calculate the real-to-complex inverse Fourier transform.

        Parameters
        ----------
        a
            Input array to be transformed.

        Returns
        -------
        a_rfft
            The inverse Fourier transform of `a`.
        """
        if isinstance(backend, NumpyBackend):
            if self.out_buffer is None:
                self.out_buffer = backend.irfft_parallel(a=a)
            else:
                backend.irfft_parallel(a=a, out=self.out_buffer)
            result = self.out_buffer
        elif isinstance(backend, CupyBackend):
            result = backend.irfft_parallel(a=a)  # out not supported (2025)
        else:
            raise RuntimeError(f"Unknown {type(backend)=}")
        return result
