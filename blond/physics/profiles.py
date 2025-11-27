# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Collection of implementations to calculate the beam profile.

Authors
-------
Simon Lauber
"""

from __future__ import annotations

import math
from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from blond.core.backends.backend import backend
from blond.core.base import BeamPhysicsRelevant, HasPropertyCache
from blond.core.helpers import int_from_float_with_warning

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


class ProfileBaseClass(BeamPhysicsRelevant, HasPropertyCache):
    """Base class to implement calculation of beam profiles.

    Parameters
    ----------
    section_index
        Section index to group elements into sections
    name
        User given name of the element

    Attributes
    ----------
    hist_y_to_density_factor
        This factor is used to reproduce the behaviour
        of np.hist(..., density=True).
        Intended use: ``density = hist_y * hist_y_to_density_factor``
    """

    def __init__(
        self, section_index: int = 0, name: str | None = None
    ) -> None:
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self._hist_x: NumpyArray | CupyArray | None = None
        self._hist_y: NumpyArray | CupyArray | None = None
        self.hist_y_to_density_factor: float | None = None

        self._beam_spectrum_buffer: dict[int, NumpyArray] = {}

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called.

        simulation
            Simulation context manager
        """
        pass

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called.

        simulation
            Simulation context manager
        beam
            Simulation `Beam` object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        assert self._hist_x is not None
        assert self._hist_y is not None
        self.invalidate_cache()

    def plot(self, **kwargs_plot: dict[str, Any]) -> None:
        """Plots the current histogram.

        Parameters
        ----------
        kwargs_plot
            Keyword arguments for `matplotlib.pyplot.plot`.

        """
        plt.plot(self.hist_x, self.hist_y, **kwargs_plot)

    @property  # as readonly attributes
    def hist_x(self) -> NumpyArray | CupyArray:
        """x-axis of histogram, in [s], i.e. `bin_centers`."""
        return self._hist_x

    @property  # as readonly attributes
    def hist_y(self) -> NumpyArray | CupyArray:
        """y-axis of histogram."""
        return self._hist_y

    @cached_property  # as readonly attributes
    def n_bins(self) -> int:
        """Number of bins in the histogram."""
        # `_hist_x`, `_hist_x` could be None, which is not handled and
        # causes a MyPy type error,
        # This is intentionally ignored, we want to get an exception.
        return len(self._hist_x)  # type: ignore

    @cached_property
    def gradient_hist_y(self) -> NumpyArray | CupyArray:
        """Derivative of the histogram."""
        return backend.gradient(self._hist_y, self.hist_step, edge_order=2)

    @cached_property
    def hist_step(self) -> float:
        """Size of a single histogram bin."""
        # `_hist_x`, `_hist_x` could be None, which is not handled and
        # causes a MyPy type error,
        # This is intentionally ignored, we want to get an exception.
        fist_hist_x = self._hist_x[0]  # type: ignore
        second_hist_x = self._hist_x[1]  # type: ignore
        if backend.is_gpu:
            fist_hist_x = fist_hist_x.get()
            second_hist_x = second_hist_x.get()
        return float(second_hist_x - fist_hist_x)  # type: ignore

    @cached_property
    def cut_left(self) -> float:
        """Left outer edge of the histogram."""
        # `_hist_x`, `_hist_x` could be None, which is not handled and
        # causes a MyPy type error,
        # This is intentionally ignored, we want to get an exception.
        fist_hist_x = self._hist_x[0]
        if backend.is_gpu:
            fist_hist_x = fist_hist_x.get()
        return float(fist_hist_x - self.hist_step / 2.0)  # type: ignore

    @cached_property
    def cut_right(self) -> float:
        """Right outer edge of the histogram."""
        # `_hist_x`, `_hist_x` could be None, which is not handled and
        # causes a MyPy type error,
        # This is intentionally ignored, we want to get an exception.
        last_hist_x = self._hist_x[-1]
        if backend.is_gpu:
            last_hist_x = last_hist_x.get()
        return float(last_hist_x + self.hist_step / 2.0)  # type: ignore

    @cached_property
    def bin_edges(self) -> NumpyArray | CupyArray:
        """Get the edges from cut_left to cut_right of the histogram."""
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
        """Bunch center of weight, in [s].

        calculates the bunch position by calculating
        the average of `hist_x` (time coordinate)
        weighted by `hist_y` (number of particles).
        """
        return backend.average(self._hist_x, weights=self._hist_y)

    def sigma_weighted_avg_dt(self) -> float:
        r"""Bunch length (:math:`1 \sigma`), in [s].

        Calculates the :math:`1 \sigma` bunch length by
        determining the std about the weighted average
        calculated as in `weighted_avg_dt`.
        """
        average = backend.average(self._hist_x, weights=self._hist_y)
        variance = backend.average(
            backend.square(self._hist_x - average), weights=self._hist_y
        )
        return backend.sqrt(variance)
    
    def gauss_fit(self) -> NumpyArray: 
        """Performs a gaussian fit on a profile with a single bunches.

        Returns the amplitude, the mean and the standard deviation of the fitted gaussian curve.
        """
        return gauss_fit(self._hist_x, self._hist_y)
    
    def multibunch_gauss_fit(self, n_bunches: int) -> NumpyArray: 
        """Performs a gaussian fit on a profile with multiple bunches.

        Returns the amplitude, the mean and the standard deviation of the fitted gaussian curve for each bunch.

        Parameters
        ----------
        n_bunches
            Number of bunches
        """
        return multi_gauss_fit(self._hist_x, self._hist_y, n_bunches)

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        if beam.is_distributed:
            raise NotImplementedError(
                "Implement histogram on distributed array"
            )
        else:
            # `_hist_x`, `_hist_x` could be None, which is not handled and
            # causes a MyPy type error,
            # This is intentionally ignored, we want to get an exception.
            backend.specials.histogram(
                array_read=beam.read_partial_dt(),
                array_write=self._hist_y,  # type: ignore
                start=self.cut_left,
                stop=self.cut_right,
            )
            # this factor is used to reproduce the behaviour
            # of np.hist(..., density=True)
            self.hist_y_to_density_factor = 1.0 / beam.common_array_size
        self.invalidate_cache()

    @staticmethod
    def get_arrays(
        cut_left: float, cut_right: float, n_bins: int
    ) -> tuple[NumpyArray, NumpyArray] | tuple[CupyArray, CupyArray]:
        """Helper method to initialize beam profiles.

        Parameters
        ----------
        cut_left
            Left outer edge of the histogram
        cut_right
            Right outer edge of the histogram
        n_bins
            Number of bins in the histogram

        Returns
        -------
        hist_x
            x-axis of histogram, in [s], i.e. `bin_centers`
        hist_y
            y-axis of histogram
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
        """Cutoff frequency if the profile is fourier transformed, in [Hz]."""
        return 1 / (2 * self.hist_step)

    def beam_spectrum(self, n_fft: int | None) -> NumpyArray:
        """Calculate fourier transform of the profile."""
        # `_hist_x`, `_hist_x` could be None, which is not handled and
        # causes a MyPy type error,
        # This is intentionally ignored, we want to get an exception.

        no_array_buffer = n_fft not in self._beam_spectrum_buffer
        if no_array_buffer:
            self._beam_spectrum_buffer[n_fft] = np.fft.rfft(
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
        self._invalidate_cache(
            props=(
                "gauss_fit_params",
                "beam_spectrum",
                "hist_step",
                "cut_left",
                "cut_right",
                "bin_edges",
                "n_bins",
            )
        )


class StaticProfile(ProfileBaseClass):
    """Calculation of beam profile that doesn't change its parameters.

    Parameters
    ----------
    cut_left
        Left outer edge of the histogram, in [s]
    cut_right
        Right outer edge of the histogram, in [s]
    n_bins
        Number of bins in the histogram
    section_index
        Section index to group elements into sections
    name
        User given name of the element
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
        self._hist_x, self._hist_y = ProfileBaseClass.get_arrays(
            cut_left=float(cut_left),
            cut_right=float(cut_right),
            n_bins=int(n_bins),
        )
        assert len(self._hist_x.shape) == 1

    @staticmethod
    def from_cutoff(
        cut_left: float,
        cut_right: float,
        cutoff_frequency: float,
        **static_profile_kwargs,
    ) -> StaticProfile:
        """Initialization method from `cutoff_frequency` in [Hz].

        Parameters
        ----------
        cut_left
            Left outer edge of the histogram
        cut_right
            Right outer edge of the histogram
        cutoff_frequency
            Cutoff frequency if the profile is fourier transformed, in [Hz]

        Returns
        -------
        static_profile
            Profile that doesn't change its parameters

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
        """Initialization method in [rad].

        Parameters
        ----------
        cut_left_rad
            Left outer edge of the histogram, in [rad]
        cut_right_rad
            Right outer edge of the histogram, in [rad]
        n_bins
            Number of bins in the histogram
        t_period
            Period according to radian, in [s]

        Returns
        -------
        static_profile
            Profile that doesn't change its parameters

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
    """Profile that can change its parameters during runtime.

    Parameters
    ----------
    section_index
        Section index to group elements into sections
    name
        User given name of the element
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
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called.

        simulation
            Simulation context manager
        beam
            Simulation `Beam` object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        self.update_attributes(beam=beam)

    @abstractmethod  # pragma: no cover
    def update_attributes(self, beam: BeamBaseClass) -> None:
        """Update the histogram limits and according arrays.

        Parameters
        ----------
        beam
            Simulation `Beam` object
        """
        pass

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        self.update_attributes(beam=beam)
        super().track(beam=beam)


class DynamicProfileConstCutoff(DynamicProfile):
    """Profile that changes its width, keeping a constant cutoff frequency.

    Parameters
    ----------
    timestep
        Time step, in [s] to keep the cutoff constant
    section_index
        Section index to group elements into sections
    name
        User given name of the element
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
        """Update the histogram limits and according arrays.

        Parameters
        ----------
        beam
            Simulation `Beam` object
        """
        cut_left = beam.dt_min  # TODO caching of attribute access
        cut_right = beam.dt_max  # TODO caching of attribute access
        n_bins = int(math.ceil((cut_right - cut_left) / self.timestep))
        self._hist_x, self._hist_y = ProfileBaseClass.get_arrays(
            cut_left=cut_left, cut_right=cut_right, n_bins=n_bins
        )


class DynamicProfileConstNBins(DynamicProfile):
    """Profile that changes its width, keeping a constant bin number.

    Parameters
    ----------
    n_bins
        Number of bins in the histogram
    section_index
        Section index to group elements into sections
    name
        User given name of the element
    """

    def __init__(
        self, n_bins: int, section_index: int = 0, name: str | None = None
    ) -> None:
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self.n_bins = int_from_float_with_warning(n_bins, warning_stacklevel=2)

    def update_attributes(self, beam: BeamBaseClass) -> None:
        """Update the histogram limits and according arrays.

        Parameters
        ----------
        beam
            Simulation `Beam` object
        """
        cut_left = beam.dt_min  # TODO caching of attribute access
        cut_right = beam.dt_max  # TODO caching of attribute access
        self._hist_x, self._hist_y = ProfileBaseClass.get_arrays(
            cut_left=cut_left, cut_right=cut_right, n_bins=self.n_bins
        )

def gauss_fit(hist_x: NumpyArray, hist_y: NumpyArray) -> NumpyArray:
    """Performs a gaussian fit on a profile with a single bunches.

        Returns the amplitude, the mean and the standard deviation of the fitted gaussian curve for each bunch.

    Parameter
    ----------
    hist_x
        X-axis of the histogram to perform the fitting on
    hist_y
        Y-axis of the histogram to perform the fitting on

    Returns
    -------
    params
        Amplitude, mean and standard deviation for each bunch
    """
    return multi_gauss_fit(hist_x, hist_y, n_bunches=1)[0]

def multi_gauss_fit(
    hist_x: NumpyArray, hist_y: NumpyArray, n_bunches: int
) -> NumpyArray:
    """Performs a gaussian fit on a profile with multiple bunches.

        Returns the amplitude, the mean and the standard deviation of the fitted gaussian curve for each bunch.

    Parameter
    ----------
    hist_x
        X-axis of the histogram to perform the fitting on
    hist_y
        Y-axis of the histogram to perform the fitting on
    n_bunches
        Number of bunches in the profile

    Returns
    -------
    params
        Amplitude, mean and standard deviation for each bunch
    """
    bucket_length = int(len(hist_x) / n_bunches)
    params = backend.zeros([n_bunches, 3])

    for bucket in range(n_bunches):
        bucket_hist_x = hist_x[
            bucket * bucket_length : (bucket + 1) * bucket_length
        ]
        bucket_hist_y = hist_y[
            bucket * bucket_length : (bucket + 1) * bucket_length
        ]

        p = [
            bucket_hist_y.max(),
            bucket_hist_x[np.argmax(bucket_hist_y)],
            bucket_hist_x[int(2 * bucket_length / 4)]
            - bucket_hist_x[int(bucket_length / 4)],
        ]

        params[bucket, :] = curve_fit(
            gauss, bucket_hist_x, bucket_hist_y, p
        )[0]

    return params

def gauss(x: NumpyArray, *p: NumpyArray) -> NumpyArray:
    r"""Returns a gaussian function.

    .. math::

    A\, e^{\frac{(x - x_0)^2}{2\sigma_x^2}}.

    Parameters
    ----------
    x
        Input array at which points to calculate the gaussian
    p
        Parameters necessary to calculate the Gaussian
        p = [A, x_0, \\sigma_x]
        A = Amplitude of the function
        x_0 = mean
        \\sigma_x = standard deviation
    """
    A, x_0, sigma_x = p

    return A * np.exp(-((x - x_0) ** 2) / 2.0 / sigma_x**2)
