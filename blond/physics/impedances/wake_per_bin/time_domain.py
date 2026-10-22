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
    salt: object = None,
) -> NumpyArray | CupyArray:
    """
    FFT of a bin-averaged wake, with the non-causal tap included.

    The bin-averaged wake reaches one bin before the charge, so it is
    sampled from ``time - dt`` and the spectrum is advanced by that one
    sample again to put lag zero where the caller expects it. One sample
    more than ``time`` holds is taken when the transform has room for it,
    so that the lags 0 to ``len(time) - 1`` are all kept next to lag -1.
    The result is cached on ``source`` under ``cache_name`` (and
    ``cache_name + "_hash"``) until ``time``, ``n_fft`` or ``salt``
    changes.

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
    salt
        Hashable summary of the source parameters the wake depends on.

    Returns
    -------
    impedance_from_wake
        Impedance array.
    """
    hash_ = hash_linspace(time, salt=(n_fft, salt))
    if hash_ == getattr(source, cache_name + "_hash", None):
        return getattr(source, cache_name)
    n_transform = len(time) if n_fft is None else n_fft
    n_samples = min(len(time) + 1, n_transform)
    bin_step = time[1] - time[0]
    lags = backend.arange(-1, n_samples - 1, dtype=backend.float)
    wake = wake_per_bin(time[0] + lags * bin_step)
    advance = backend.exp(
        1j
        * backend.twopi
        * backend.arange(n_transform // 2 + 1, dtype=backend.float)
        / n_transform
    )
    impedance_from_wake = backend.fft.rfft(wake, n=n_transform) * advance
    setattr(source, cache_name + "_hash", hash_)
    setattr(source, cache_name, impedance_from_wake)
    return impedance_from_wake


def provides_wake_per_bin(source: object) -> bool:
    """
    Whether ``source`` can return a bin-averaged wake.

    Every :class:`~blond.physics.impedances.base.TimeDomain` source inherits
    the generic :meth:`TimeDomainWakePerBin.get_wake_per_bin`, but that one
    only works on top of a point wake ``get_wake``.

    Parameters
    ----------
    source
        The wake source to check.

    Returns
    -------
    provides_wake_per_bin
        ``True`` if ``source`` overrides ``get_wake_per_bin``, or keeps the
        generic one and implements ``get_wake``.
    """
    wake_per_bin = getattr(source, "get_wake_per_bin", None)
    if wake_per_bin is None:
        return False
    if (
        getattr(wake_per_bin, "__func__", None)
        is not TimeDomainWakePerBin.get_wake_per_bin
    ):
        return True
    wake = getattr(source, "get_wake", None)
    return (
        wake is not None
        and getattr(wake, "__func__", None)
        is not TimeDomainWakePerBin.get_wake
    )


class _WakeParameters:
    """Cache invalidation shared by the bin-averaged wake mix-ins."""

    def _wake_parameters_salt(self) -> object:
        """
        Hashable summary of the parameters the wake depends on.

        The cached impedance of :func:`impedance_from_wake_per_bin` is
        recomputed whenever this changes. Sources whose parameters can be
        changed after construction override it.

        Returns
        -------
        salt
            ``None``: the wake is fixed once the source is built.
        """
        return None


class TimeDomainWakePerBin(_WakeParameters):
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
        (see :mod:`blond.physics.impedances.wake_per_bin.bin_average`). The kernel is
        non-zero from ``-1.5 * dt``, so callers sample from one bin before
        the profile.

        This default only samples :meth:`get_wake` at the bin width and
        treats it as piecewise linear between those samples, so it is
        second order in the bin width for a wake the bins resolve, but
        cannot suppress the aliasing of one they do not: that takes the
        exact averages of the overriding sources (``Resonators``,
        ``TravelingWaveCavity``, ``ImpedanceTableTime``).

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
            salt=self._wake_parameters_salt(),
        )


class TimeDomainCounterRotationWakePerBin(_WakeParameters):
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
            salt=self._wake_parameters_salt(),
        )
