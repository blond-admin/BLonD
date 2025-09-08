from __future__ import annotations

import math
from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from .._core.backends.backend import backend
from .._core.base import BeamPhysicsRelevant
from .._core.helpers import int_from_float_with_warning

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any, Dict
    from typing import Optional
    from typing import Optional as LateInit
    from typing import Tuple

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from .._core.beam.base import BeamBaseClass
    from .._core.simulation.simulation import Simulation


class ProfileBaseClass(BeamPhysicsRelevant):
    def __init__(
        self, section_index: int = 0, name: Optional[str] = None
    ) -> None:
        """
        Base class to implement calculation of beam profiles


        Parameters
        ----------
        section_index
            Section index to group elements into sections
        name
            User given name of the element

        """
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self._hist_x: LateInit[NumpyArray | CupyArray] = None
        self._hist_y: LateInit[NumpyArray | CupyArray] = None

        self._beam_spectrum_buffer: Dict[int, NumpyArray] = {}

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

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
        **kwargs: Dict[str, Any],
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        beam
            Simulation beam object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        assert self._hist_x is not None
        assert self._hist_y is not None
        self.invalidate_cache()

    @property  # as readonly attributes
    def hist_x(self) -> NumpyArray | CupyArray:
        """x-axis of histogram, in [s], i.e. `bin_centers`"""
        return self._hist_x

    @property  # as readonly attributes
    def hist_y(self) -> NumpyArray | CupyArray:
        """y-axis of histogram"""
        return self._hist_y

    @cached_property  # as readonly attributes
    def n_bins(self) -> int:
        """Number of bins in the histogram"""
        # `_hist_x`, `_hist_x` could be None, which is not handled and
        # causes a MyPy type error,
        # This is intentionally ignored, we want to get an exception.
        return len(self._hist_x)  # type: ignore

    @cached_property
    def diff_hist_y(self) -> NumpyArray | CupyArray:
        """Derivative of the histogram"""
        return backend.gradient(self._hist_y, self.hist_step, edge_order=2)

    @cached_property
    def hist_step(self) -> np.float32 | np.float64:
        """Size of a single histogram bin"""
        # `_hist_x`, `_hist_x` could be None, which is not handled and
        # causes a MyPy type error,
        # This is intentionally ignored, we want to get an exception.
        fist_hist_x = self._hist_x[0]  # type: ignore
        second_hist_x = self._hist_x[1]  # type: ignore
        if backend.is_gpu:
            fist_hist_x = fist_hist_x.get()
            second_hist_x = second_hist_x.get()
        return backend.float(second_hist_x - fist_hist_x)  # type: ignore

    @cached_property
    def cut_left(self) -> np.float32 | np.float64:
        """Left outer edge of the histogram"""
        # `_hist_x`, `_hist_x` could be None, which is not handled and
        # causes a MyPy type error,
        # This is intentionally ignored, we want to get an exception.
        fist_hist_x = self._hist_x[0]
        if backend.is_gpu:
            fist_hist_x = fist_hist_x.get()
        return backend.float(fist_hist_x - self.hist_step / 2.0)  # type: ignore

    @cached_property
    def cut_right(self) -> np.float32 | np.float64:
        """Right outer edge of the histogram"""
        # `_hist_x`, `_hist_x` could be None, which is not handled and
        # causes a MyPy type error,
        # This is intentionally ignored, we want to get an exception.
        last_hist_x = self._hist_x[-1]
        if backend.is_gpu:
            last_hist_x = last_hist_x.get()
        return backend.float(last_hist_x + self.hist_step / 2.0)  # type: ignore

    @cached_property
    def bin_edges(self) -> NumpyArray | CupyArray:
        """Get the edges from cut_left to cut_right of the histogram"""
        # `_hist_x`, `_hist_x` could be None, which is not handled and
        # causes a MyPy type error,
        # This is intentionally ignored, we want to get an exception.
        return backend.linspace(
            self.cut_left,
            self.cut_right,
            len(self._hist_x) + 1,
            backend.float,  # type: ignore
        )

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

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
        self.invalidate_cache()

    @staticmethod
    def get_arrays(
        cut_left: float, cut_right: float, n_bins: int
    ) -> Tuple[NumpyArray, NumpyArray] | Tuple[CupyArray, CupyArray]:
        """
        Helper method to initialize beam profiles

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
    def cutoff_frequency(self) -> np.float32 | np.float64:
        """Cutoff frequency if the profile is fourier transformed, in [Hz]"""
        return backend.float(1 / (2 * self.hist_step))

    def _calc_gauss(self) -> None:
        raise NotImplementedError
        return

    @cached_property
    def gauss_fit_params(self) -> None:
        raise NotImplementedError
        return self._calc_gauss()

    def beam_spectrum(self, n_fft: int) -> NumpyArray:
        """Calculate fourier transform of the profile"""
        # `_hist_x`, `_hist_x` could be None, which is not handled and
        # causes a MyPy type error,
        # This is intentionally ignored, we want to get an exception.

        if n_fft not in self._beam_spectrum_buffer.keys():
            self._beam_spectrum_buffer[n_fft] = np.fft.rfft(
                self._hist_y,  # type: ignore
                n_fft,
            )
        else:
            if backend.is_gpu:
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
        """Delete the stored values of functions with @cached_property"""
        for attribute in (
            "gauss_fit_params",
            "beam_spectrum",
            "hist_step",
            "cut_left",
            "cut_right",
            "bin_edges",
            "n_bins",
        ):
            self.__dict__.pop(attribute, None)


class StaticProfile(ProfileBaseClass):
    def __init__(
        self,
        cut_left: float,
        cut_right: float,
        n_bins: int,
        section_index: int = 0,
        name: Optional[str] = None,
    ) -> None:
        """
        Calculation of beam profile that doesn't change its parameters

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
        cut_left: float, cut_right: float, cutoff_frequency: float
    ) -> StaticProfile:
        """
        Initialization method from `cutoff_frequency` in Hz

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
            cut_left=cut_left, cut_right=cut_right, n_bins=n_bins
        )

    @staticmethod
    def from_rad(
        cut_left_rad: float, cut_right_rad: float, n_bins: int, t_period: float
    ) -> StaticProfile:
        """
        Initialization method in radian

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
            cut_left=cut_left, cut_right=cut_right, n_bins=n_bins
        )


class DynamicProfile(ProfileBaseClass):
    def __init__(
        self, section_index: int = 0, name: Optional[str] = None
    ) -> None:
        """
        Profile that can change its parameters during runtime

        Parameters
        ----------
        section_index
            Section index to group elements into sections
        name
            User given name of the element
        """
        super().__init__(
            section_index=section_index,
            name=name,
        )

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        super().on_init_simulation(simulation=simulation)

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        beam
            Simulation beam object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        self.update_attributes(beam=beam)

    @abstractmethod  # pragma: no cover
    def update_attributes(self, beam: BeamBaseClass) -> None:
        """Method to update the attributes"""
        pass

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        self.update_attributes(beam=beam)
        super().track(beam=beam)


class DynamicProfileConstCutoff(DynamicProfile):
    def __init__(
        self,
        timestep: float,
        section_index: int = 0,
        name: Optional[str] = None,
    ) -> None:
        """
        Profile that changes its width, keeping a constant cutoff frequency

        Parameters
        ----------
        timestep
            Time step, in [s] to keep the cutoff constant
        section_index
            Section index to group elements into sections
        name
            User given name of the element
        """
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self.timestep = timestep

    def update_attributes(self, beam: BeamBaseClass) -> None:
        cut_left = beam.dt_min  # TODO caching of attribute access
        cut_right = beam.dt_max  # TODO caching of attribute access
        n_bins = int(math.ceil((cut_right - cut_left) / self.timestep))
        self._hist_x, self._hist_y = ProfileBaseClass.get_arrays(
            cut_left=cut_left, cut_right=cut_right, n_bins=n_bins
        )


class DynamicProfileConstNBins(DynamicProfile):
    def __init__(
        self, n_bins: int, section_index: int = 0, name: Optional[str] = None
    ) -> None:
        """
        Profile that changes its width, keeping a constant bin number

        Parameters
        ----------
        n_bins
            Number of bins in the histogram
        section_index
            Section index to group elements into sections
        name
            User given name of the element
        """
        super().__init__(
            section_index=section_index,
            name=name,
        )
        self.n_bins = int_from_float_with_warning(n_bins, warning_stacklevel=2)

    def update_attributes(self, beam: BeamBaseClass) -> None:
        cut_left = beam.dt_min  # TODO caching of attribute access
        cut_right = beam.dt_max  # TODO caching of attribute access
        self._hist_x, self._hist_y = ProfileBaseClass.get_arrays(
            cut_left=cut_left, cut_right=cut_right, n_bins=self.n_bins
        )
