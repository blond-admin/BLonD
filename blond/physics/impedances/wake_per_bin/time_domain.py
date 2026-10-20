# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Bin-averaged wake of the time-domain source interfaces.

Mixed into :class:`~blond.physics.impedances.base.TimeDomain` and
:class:`~blond.physics.impedances.base.TimeDomainCounterRotation`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from blond.core.backends.backend import backend
from blond.generals.hashing_ import hash_linspace

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


def bspline_stencil(
    wake: NumpyArray | CupyArray,
) -> NumpyArray | CupyArray:
    """
    Average a piecewise-linear wake with the quadratic B-spline of one bin.

    On a uniform grid that is the stencil
    ``(w[n-2] + 76 w[n-1] + 230 w[n] + 76 w[n+1] + w[n+2]) / 384``; the
    edges repeat the boundary sample.

    Parameters
    ----------
    wake
        Wake sampled on a uniform grid, in [V].

    Returns
    -------
    wake_per_bin
        Bin-averaged wake, in [V].
    """
    previous_1 = backend.concatenate((wake[:1], wake[:-1]))
    previous_2 = backend.concatenate((wake[:1], wake[:1], wake[:-2]))
    next_1 = backend.concatenate((wake[1:], wake[-1:]))
    next_2 = backend.concatenate((wake[2:], wake[-1:], wake[-1:]))
    return (
        previous_2 + 76.0 * previous_1 + 230.0 * wake + 76.0 * next_1 + next_2
    ) / 384.0


def impedance_from_wake_per_bin(
    source: object,
    wake_per_bin: Callable[[NumpyArray | CupyArray], NumpyArray | CupyArray],
    time: NumpyArray | CupyArray,
    n_fft: int | None,
    cache_name: str,
) -> NumpyArray | CupyArray:
    """
    FFT of a bin-averaged wake, with the non-causal tap included.

    The bin-averaged wake reaches one bin before the charge, so it is
    sampled from ``time - dt`` and the spectrum is advanced by that one
    sample again to put lag zero where the caller expects it. The result is
    cached on ``source`` under ``cache_name`` (and ``cache_name + "_hash"``)
    until ``time`` changes.

    Parameters
    ----------
    source
        The wake source the cache lives on.
    wake_per_bin
        Function returning the bin-averaged wake at given times, in [V].
    time
        Time array to get the wake for, in [s].
    n_fft
        Number of points of the FFT; ``None`` transforms the wake as is.
    cache_name
        Attribute name of the cached impedance on ``source``.

    Returns
    -------
    impedance_from_wake
        Impedance array.
    """
    hash_ = hash_linspace(time)
    if hash_ == getattr(source, cache_name + "_hash", None):
        return getattr(source, cache_name)
    wake = wake_per_bin(time - (time[1] - time[0]))
    n_transform = len(wake) if n_fft is None else n_fft
    advance = backend.exp(
        1j
        * backend.twopi
        * backend.arange(n_transform // 2 + 1, dtype=backend.float)
        / n_transform
    )
    impedance_from_wake = backend.fft.rfft(wake, n=n_fft) * advance
    setattr(source, cache_name + "_hash", hash_)
    setattr(source, cache_name, impedance_from_wake)
    return impedance_from_wake


class TimeDomainWakePerBin:
    """Bin-averaged wake of a time-domain source, from its point wake."""

    def get_wake(self, time: NumpyArray | CupyArray) -> NumpyArray | CupyArray:
        """
        Point-charge wake sampled at ``time``, in [V].

        Parameters
        ----------
        time
            Time array at which the wake is evaluated, in [s].

        Returns
        -------
        wake
            Point-charge wake, in [V].
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a point-charge wake."
        )

    def get_wake_per_bin(
        self, time: NumpyArray | CupyArray
    ) -> NumpyArray | CupyArray:
        """
        Wake averaged with the quadratic B-spline of one bin width, in [V].

        Time-domain solvers apply this instead of :meth:`get_wake`: the
        induced voltage of a histogram is the wake integrated over the bin,
        and averaging with ``box * box * box`` keeps a resonance above the
        profile's Nyquist frequency from aliasing onto the bunch spectrum
        (see :mod:`blond.physics.impedances.bin_average`). The kernel is
        non-zero from ``-1.5 * dt``, so callers sample from one bin before
        the profile. The default treats :meth:`get_wake` as piecewise
        linear between its samples; sources with a closed form override it.

        Parameters
        ----------
        time
            Time array (bin centres, uniformly spaced) at which the wake is
            evaluated, in [s].

        Returns
        -------
        wake
            Bin-averaged wake, in [V].
        """
        return bspline_stencil(self.get_wake(time))

    def _impedance_from_wake_per_bin(
        self, time: NumpyArray | CupyArray, n_fft: int | None
    ) -> NumpyArray | CupyArray:
        """
        Impedance equivalent to the bin-averaged wake.

        Parameters
        ----------
        time
            Time array to get the wake for, in [s].
        n_fft
            Number of points of the FFT; ``None`` transforms the wake as is.

        Returns
        -------
        impedance_from_wake
            ``rfft`` of :meth:`get_wake_per_bin`, non-causal tap included.
        """
        return impedance_from_wake_per_bin(
            self,
            self.get_wake_per_bin,
            time,
            n_fft,
            "_cache_impedance_from_wake",
        )


class TimeDomainCounterRotationWakePerBin:
    """Bin-averaged wake of a source for the counter-rotating case."""

    def get_wake_per_bin_counter_rotation(
        self, time: NumpyArray | CupyArray
    ) -> NumpyArray | CupyArray:
        """
        Bin-averaged wake for the counter-rotating case, in [V].

        See
        :meth:`~blond.physics.impedances.base.TimeDomain.get_wake_per_bin`.

        Parameters
        ----------
        time
            Time array (bin centres, uniformly spaced) at which the wake is
            evaluated, in [s].

        Returns
        -------
        wake
            Bin-averaged wake, in [V].
        """
        return bspline_stencil(self.get_wake_counter_rotation(time))

    def _impedance_from_wake_per_bin_counter_rotation(
        self, time: NumpyArray | CupyArray, n_fft: int | None
    ) -> NumpyArray | CupyArray:
        """
        Impedance equivalent to the counter-rotating bin-averaged wake.

        Parameters
        ----------
        time
            Time array to get the wake for, in [s].
        n_fft
            Number of points of the FFT; ``None`` transforms the wake as is.

        Returns
        -------
        impedance_from_wake
            ``rfft`` of :meth:`get_wake_per_bin_counter_rotation`.
        """
        return impedance_from_wake_per_bin(
            self,
            self.get_wake_per_bin_counter_rotation,
            time,
            n_fft,
            "_cache_impedance_from_wake_counter_rotation",
        )
