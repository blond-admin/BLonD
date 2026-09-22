# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Collection of implementations to calculate the beam profile."""

from __future__ import annotations

import math
from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from blond.acc_math.empiric.empiric import gauss_fit, multi_gauss_fit
from blond.core.backends.backend import backend
from blond.core.base import BeamPhysicsRelevant, HasPropertyCache
from blond.core.helpers import int_from_float_with_warning
from blond.generals.cupy_.no_cupy_import import copy_to_cpu, is_cupy_array

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


class ProfileBaseClass(BeamPhysicsRelevant, HasPropertyCache):
    """
    Base class to implement calculation of beam profiles.

    Parameters
    ----------
    section_index
        Section index to group elements into sections.
    name
        User given name of the element.

    Attributes
    ----------
    hist_y_to_density_factor
        This factor is used to reproduce the behaviour
        of np.hist(..., density=True).
        Intended use: ``density = hist_y * hist_y_to_density_factor``
    """

    # The geometry (edges and the arrays sized by it) is only written by
    # `_set_window` and `_bind_arrays`, so it can never disagree.
    # `hist_y` changes every turn, but in place. See `__setattr__`.
    _GEOMETRY_FIELDS = frozenset(
        ("_cut_left", "_cut_right", "_hist_x", "_hist_y")
    )
    _cut_left: float | None = None
    _cut_right: float | None = None
    _hist_x: NumpyArray | CupyArray | None = None
    _hist_y: NumpyArray | CupyArray | None = None

    def __init__(
        self, section_index: int = 0, name: str | None = None
    ) -> None:
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self.hist_y_to_density_factor: float | None = None

        self._beam_spectrum_buffer: dict[int, NumpyArray] = {}

    def _set_window(
        self, cut_left: float, cut_right: float, n_bins: int
    ) -> None:
        """
        Set the histogram window and allocate the according arrays.

        This is the only place the geometry changes, so `cut_left`,
        `cut_right` and `hist_step` are stored instead of re-derived from
        `hist_x`, which would round the edges and cost device->host syncs.

        Parameters
        ----------
        cut_left
            Left outer edge of the histogram, in [s].
        cut_right
            Right outer edge of the histogram, in [s].
        n_bins
            Number of bins in the histogram.
        """
        hist_x, hist_y = ProfileBaseClass.get_arrays(
            cut_left=float(cut_left),
            cut_right=float(cut_right),
            n_bins=int(n_bins),
        )
        object.__setattr__(self, "_cut_left", float(cut_left))
        object.__setattr__(self, "_cut_right", float(cut_right))
        object.__setattr__(self, "_hist_x", hist_x)
        object.__setattr__(self, "_hist_y", hist_y)
        self.invalidate_cache()

    def _bind_arrays(
        self,
        hist_x: NumpyArray | CupyArray,
        hist_y: NumpyArray | CupyArray,
    ) -> None:
        """
        Replace the histogram arrays by others holding the same geometry.

        Intended to bind the profile to externally owned memory, e.g. a
        view into one continuous array.

        Parameters
        ----------
        hist_x
            X-axis of histogram, in [s], equal to the current `hist_x`.
        hist_y
            Y-axis of histogram.
        """
        assert len(hist_x) == len(hist_y) == self.n_bins
        assert hist_y.dtype == self._hist_y.dtype
        assert np.allclose(copy_to_cpu(hist_x), copy_to_cpu(self._hist_x)), (
            "`hist_x` must keep the geometry, use `_set_window` to change it"
        )
        object.__setattr__(self, "_hist_x", hist_x)
        object.__setattr__(self, "_hist_y", hist_y)
        self.invalidate_cache()

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Refuse direct writes to the geometry.

        Parameters
        ----------
        name
            Name of the attribute.
        value
            Value of the attribute.

        Raises
        ------
        AttributeError
            If `name` is part of the geometry.
        """
        # rebinding the same object, e.g. by `hist_y *= 2`, changes nothing
        if name in self._GEOMETRY_FIELDS and value is not getattr(
            self, name, None
        ):
            raise AttributeError(
                f"`{name}` is part of the profile geometry, set it via"
                " `_set_window` (or `_bind_arrays` for the same geometry)."
            )
        super().__setattr__(name, value)

    def on_init_simulation(self, simulation: Simulation, **kwargs) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        **kwargs
            Configure parameters collected by the MRO chain.
        """
        super().on_init_simulation(simulation=simulation, **kwargs)

    def configure(self, **kwargs) -> None:
        """
        Invalidate the geometry cache whenever configure is called.

        Parameters
        ----------
        **kwargs
            Passed to the next level in the MRO chain.
        """
        super().configure(**kwargs)
        self.invalidate_cache()

    def configure_run(
        self,
        *,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Validate histogram arrays and invalidate cache at run start.

        Parameters
        ----------
        beam
            The beam being simulated.
        n_turns
            Number of turns for this run.
        **kwargs
            Simulation-extracted values; passed to the next MRO level.
        """
        super().configure_run(beam=beam, n_turns=n_turns, **kwargs)
        assert self._hist_x is not None
        assert self._hist_y is not None
        self.invalidate_cache()

    def plot(self, **kwargs_plot: dict[str, Any]) -> list[Any]:
        """
        Plot the current histogram.

        Parameters
        ----------
        **kwargs_plot
            Keyword arguments for `matplotlib.pyplot.plot`.

        Returns
        -------
        artists
            The plotting artists.
        """
        from blond.generals.cupy_.no_cupy_import import AllowPlotting

        with AllowPlotting():
            artists = plt.plot(self.hist_x, self.hist_y, **kwargs_plot)
        return artists

    @property  # as readonly attributes
    def hist_x(self) -> NumpyArray | CupyArray:
        """
        Return x-axis of histogram, in [s], i.e. `bin_centers`.

        Returns
        -------
        hist_x
            X-axis of histogram, in [s], i.e. `bin_centers`.
        """
        return self._hist_x

    @property  # as readonly attributes
    def hist_y(self) -> NumpyArray | CupyArray:
        """
        Return y-axis of histogram.

        Returns
        -------
        hist_y
            Y-axis of histogram.
        """
        return self._hist_y

    @property  # as readonly attributes
    def n_bins(self) -> int:
        """
        Number of bins in the histogram.

        Returns
        -------
        n_bins
            Number of bins in the histogram.
        """
        # `_hist_x`, `_hist_x` could be None, which is not handled and
        # causes a MyPy type error,
        # This is intentionally ignored, we want to get an exception.
        return len(self._hist_x)  # type: ignore

    @cached_property
    def gradient_hist_y(self) -> NumpyArray | CupyArray:
        """
        Derivative of the histogram.

        Returns
        -------
        gradient_hist_y
            Derivative of the histogram.
        """
        return backend.gradient(self._hist_y, self.hist_step, edge_order=2)

    @property  # as readonly attributes
    def hist_step(self) -> float:
        """
        Size of a single histogram bin.

        Returns
        -------
        hist_step
            Size of a single histogram bin.
        """
        # `_cut_left`, `_cut_right` could be None, which is not handled and
        # causes a MyPy type error,
        # This is intentionally ignored, we want to get an exception.
        return (self._cut_right - self._cut_left) / self.n_bins  # type: ignore

    @property  # as readonly attributes
    def cut_left(self) -> float:
        """
        Left outer edge of the histogram.

        Returns
        -------
        cut_left
            Left outer edge of the histogram.
        """
        return self._cut_left  # type: ignore

    @property  # as readonly attributes
    def cut_right(self) -> float:
        """
        Right outer edge of the histogram.

        Returns
        -------
        cut_right
            Right outer edge of the histogram.
        """
        return self._cut_right  # type: ignore

    @property  # as readonly attributes
    def bin_edges(self) -> NumpyArray | CupyArray:
        """
        Get the edges from cut_left to cut_right of the histogram.

        Returns
        -------
        bin_edges
            Edges from cut_left to cut_right of the histogram.
        """
        # `_hist_x`, `_hist_x` could be None, which is not handled and
        # causes a MyPy type error,
        # This is intentionally ignored, we want to get an exception.
        return backend.linspace(
            self.cut_left,
            self.cut_right,
            len(self._hist_x) + 1,
            backend.float,  # type: ignore
        )

    def weighted_avg_dt(self) -> float:
        """
        Bunch center of weight, in [s].

        calculates the bunch position by calculating
        the average of `hist_x` (time coordinate)
        weighted by `hist_y` (number of particles).

        Returns
        -------
        weighted_avg_dt
            Bunch center of weight, in [s].
        """
        return backend.average(self._hist_x, weights=self._hist_y)

    def sigma_weighted_avg_dt(self) -> float:
        r"""
        Bunch length (:math:`1 \sigma`), in [s].

        Calculates the :math:`1 \sigma` bunch length by
        determining the std about the weighted average
        calculated as in `weighted_avg_dt`.

        Returns
        -------
        sigma_weighted_avg_dt
            Bunch length (:math:`1 \sigma`), in [s].
        """
        average = backend.average(self._hist_x, weights=self._hist_y)
        variance = backend.average(
            backend.square(self._hist_x - average), weights=self._hist_y
        )
        return backend.sqrt(variance)

    def singlebunch_gauss_fit(self) -> NumpyArray:
        """
        Perform a gaussian fit on a profile with a single bunches.

        Returns the amplitude, the mean and the standard deviation
        of the fitted gaussian curve the bunch.

        Returns
        -------
        params
            Amplitude, mean and standard deviation the bunch.
        """
        _hist_x = self._hist_x
        _hist_y = self._hist_y

        if is_cupy_array(self._hist_x):
            _hist_x = _hist_x.get()
            _hist_y = _hist_y.get()

        return gauss_fit(_hist_x, _hist_y)

    def multibunch_gauss_fit(self, n_bunches: int) -> NumpyArray:
        """
        Perform a gaussian fit on a profile with multiple bunches.

        Returns the amplitude, the mean and the standard deviation of the fitted
        gaussian curve for each bunch.

        Parameters
        ----------
        n_bunches
            Number of bunches.

        Returns
        -------
        params
            Amplitude, mean and standard deviation for each bunch.
            Shape (n_bunches, 3).
        """
        _hist_x = self._hist_x
        _hist_y = self._hist_y

        if is_cupy_array(self._hist_x):
            _hist_x = _hist_x.get()
            _hist_y = _hist_y.get()

        return multi_gauss_fit(_hist_x, _hist_y, n_bunches)

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        if beam.is_distributed:
            raise NotImplementedError(
                "Implement histogram on distributed array"
            )
        elif beam.common_array_size > 0:
            # `_hist_x`, `_hist_y` could be None, which is not handled and
            # causes a MyPy type error,
            # This is intentionally ignored, we want to get an exception.
            beam._dt.histogram(  # MPI aware histogram calculation
                len(self._hist_y),
                range=(
                    self.cut_left,
                    self.cut_right,
                ),
                out=self._hist_y,
            )
            # this factor is used to reproduce the behaviour
            # of np.hist(..., density=True)
            self.hist_y_to_density_factor = 1.0 / beam.common_array_size
        else:
            self._hist_y[:] = 0
            self.hist_y_to_density_factor = 0.0

        self.invalidate_cache()

    @staticmethod
    def get_arrays(
        cut_left: float, cut_right: float, n_bins: int
    ) -> tuple[NumpyArray, NumpyArray] | tuple[CupyArray, CupyArray]:
        """
        Helper method to initialize beam profiles.

        Parameters
        ----------
        cut_left
            Left outer edge of the histogram.
        cut_right
            Right outer edge of the histogram.
        n_bins
            Number of bins in the histogram.

        Returns
        -------
        hist_x
            X-axis of histogram, in [s], i.e. `bin_centers`.
        hist_y
            Y-axis of histogram.
        """
        step = (cut_right - cut_left) / n_bins
        offset = step / 2
        hist_x = backend.linspace(
            cut_left + offset, cut_right - offset, n_bins, dtype=backend.float
        )
        hist_y = backend.zeros(n_bins, dtype=backend.float)
        return hist_x, hist_y

    @property  # as readonly attributes
    def cutoff_frequency(self) -> float:
        """
        Cutoff frequency if the profile is fourier transformed, in [Hz].

        Returns
        -------
        cutoff_frequency
            Cutoff frequency if the profile is fourier transformed, in [Hz].
        """
        return 1 / (2 * self.hist_step)

    def beam_spectrum(self, n_fft: int | None) -> NumpyArray | CupyArray:
        """
        Calculate fourier transform of the profile.

        Parameters
        ----------
        n_fft
            Number of FFT points.

        Returns
        -------
        spectrum
            Fourier transform of the profile.
        """
        # `_hist_x`, `_hist_x` could be None, which is not handled and
        # causes a MyPy type error,
        # This is intentionally ignored, we want to get an exception.

        no_array_buffer = n_fft not in self._beam_spectrum_buffer
        if no_array_buffer:
            self._beam_spectrum_buffer[n_fft] = backend.fft.rfft(
                self._hist_y,  # type: ignore
                n_fft,
            )
        # recycle array, but overwrite data (preventing new array allocation)
        elif backend.is_gpu:
            # At the time of writing (2025), out is not a keyword argument
            # of cp.fft.rfft, but might be in future.
            self._beam_spectrum_buffer[n_fft] = backend.fft.rfft(
                self._hist_y,
                n_fft,
            )
        else:
            backend.fft.rfft(
                self._hist_y,
                n_fft,
                out=self._beam_spectrum_buffer[n_fft],  # type: ignore
            )

        return self._beam_spectrum_buffer[n_fft]

    def invalidate_cache(self) -> None:
        """Delete the stored values of functions with @cached_property."""
        self._invalidate_cache(props=("gradient_hist_y",))


class StaticProfile(ProfileBaseClass):
    """
    Calculation of beam profile that doesn't change its parameters.

    Parameters
    ----------
    cut_left
        Left outer edge of the histogram, in [s].
    cut_right
        Right outer edge of the histogram, in [s].
    n_bins
        Number of bins in the histogram.
    section_index
        Section index to group elements into sections.
    name
        User given name of the element.
    """

    def __init__(
        self,
        cut_left: float,
        cut_right: float,
        n_bins: int,
        section_index: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self._set_window(cut_left=cut_left, cut_right=cut_right, n_bins=n_bins)
        assert len(self._hist_x.shape) == 1

    @staticmethod
    def from_cutoff(
        cut_left: float,
        cut_right: float,
        cutoff_frequency: float,
        **static_profile_kwargs,
    ) -> StaticProfile:
        """
        Initialization method from `cutoff_frequency` in [Hz].

        Parameters
        ----------
        cut_left
            Left outer edge of the histogram.
        cut_right
            Right outer edge of the histogram.
        cutoff_frequency
            Cutoff frequency if the profile is fourier transformed, in [Hz].
        **static_profile_kwargs
            Additional keyword arguments for StaticProfile initialization.

        Returns
        -------
        static_profile
            Profile that doesn't change its parameters.
        """
        dt = 1 / (2 * cutoff_frequency)
        n_bins = int(math.ceil((cut_right - cut_left) / dt))
        return StaticProfile(
            cut_left=cut_left,
            cut_right=cut_right,
            n_bins=n_bins,
            **static_profile_kwargs,
        )

    @staticmethod
    def from_rad(
        cut_left_rad: float,
        cut_right_rad: float,
        n_bins: int,
        t_period: float,
        **static_profile_kwargs,
    ) -> StaticProfile:
        """
        Initialization method in [rad].

        Parameters
        ----------
        cut_left_rad
            Left outer edge of the histogram, in [rad].
        cut_right_rad
            Right outer edge of the histogram, in [rad].
        n_bins
            Number of bins in the histogram.
        t_period
            Period according to radian, in [s].
        **static_profile_kwargs
            Additional keyword arguments for StaticProfile initialization.

        Returns
        -------
        static_profile
            Profile that doesn't change its parameters.
        """
        rad_to_frac = 1 / (2 * np.pi)
        cut_left = cut_left_rad * rad_to_frac * t_period
        cut_right = cut_right_rad * rad_to_frac * t_period
        return StaticProfile(
            cut_left=cut_left,
            cut_right=cut_right,
            n_bins=n_bins,
            **static_profile_kwargs,
        )


class DynamicProfile(ProfileBaseClass):
    """
    Profile that can change its parameters during runtime.

    Parameters
    ----------
    section_index
        Section index to group elements into sections.
    name
        User given name of the element.
    """

    def __init__(
        self, section_index: int = 0, name: str | None = None
    ) -> None:
        super().__init__(
            section_index=section_index,
            name=name,
        )

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation `Beam` object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Simulation-extracted kwargs collected by the MRO chain.
        """
        super().on_run_simulation(simulation, beam, n_turns, **kwargs)

    def configure_run(
        self,
        *,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Update histogram limits from the beam at run start.

        Parameters
        ----------
        beam
            The beam being simulated.
        n_turns
            Number of turns for this run.
        **kwargs
            Simulation-extracted values; passed to the next MRO level.
        """
        self.update_attributes(beam=beam)
        # super call after attribute updates, because it also checks
        # whether the attributes are set correctly.
        super().configure_run(beam=beam, n_turns=n_turns, **kwargs)

    @abstractmethod  # pragma: no cover
    def update_attributes(self, beam: BeamBaseClass) -> None:
        """
        Update the histogram limits and according arrays.

        Parameters
        ----------
        beam
            Simulation `Beam` object.
        """
        pass

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        self.update_attributes(beam=beam)
        super()._track(beam=beam)


class DynamicProfileConstCutoff(DynamicProfile):
    """
    Profile that changes its width, keeping a constant cutoff frequency.

    Parameters
    ----------
    timestep
        Time step, in [s] to keep the cutoff constant.
    section_index
        Section index to group elements into sections.
    name
        User given name of the element.
    """

    def __init__(
        self,
        timestep: float,
        section_index: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self.timestep = timestep

    def update_attributes(self, beam: BeamBaseClass) -> None:
        """
        Update the histogram limits and according arrays.

        Parameters
        ----------
        beam
            Simulation `Beam` object.
        """
        cut_left = beam.dt_min  # TODO caching of attribute access
        cut_right = beam.dt_max  # TODO caching of attribute access
        timesteps = (cut_right - cut_left) / self.timestep
        # a whole number of timesteps must not get an extra bin
        # from float rounding
        if math.isclose(timesteps, round(timesteps)):
            n_bins = round(timesteps)
        else:
            n_bins = math.ceil(timesteps)
        self._set_window(cut_left=cut_left, cut_right=cut_right, n_bins=n_bins)


class DynamicProfileConstNBins(DynamicProfile):
    """
    Profile that changes its width, keeping a constant bin number.

    Parameters
    ----------
    n_bins
        Number of bins in the histogram.
    section_index
        Section index to group elements into sections.
    name
        User given name of the element.
    """

    _GEOMETRY_FIELDS = DynamicProfile._GEOMETRY_FIELDS | {"_n_bins"}

    def __init__(
        self, n_bins: int, section_index: int = 0, name: str | None = None
    ) -> None:
        super().__init__(
            section_index=section_index,
            name=name,
        )
        object.__setattr__(  # fixed, it sizes every window
            self,
            "_n_bins",
            int_from_float_with_warning(n_bins, warning_stacklevel=2),
        )

    @property  # as readonly attributes
    def n_bins(self) -> int:
        """
        Number of bins in the histogram, known before the first window.

        Returns
        -------
        n_bins
            Number of bins in the histogram.
        """
        return self._n_bins

    def update_attributes(self, beam: BeamBaseClass) -> None:
        """
        Update the histogram limits and according arrays.

        Parameters
        ----------
        beam
            Simulation `Beam` object.
        """
        cut_left = beam.dt_min  # TODO caching of attribute access
        cut_right = beam.dt_max  # TODO caching of attribute access
        self._set_window(
            cut_left=cut_left, cut_right=cut_right, n_bins=self.n_bins
        )
