# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Collection of abstract classes to handle the calculation of wake potentials."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from scipy.constants import elementary_charge as e

from blond.core.backends.backend import backend
from blond.core.base import BeamPhysicsRelevant
from blond.core.ring.helpers import requires
from blond.experimental.physics.kick_pooling import (
    SupportsPooledInterpolationKickMixIn,
)

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation
    from blond.experimental.physics.kick_pooling import (
        PooledInterpolationKick,
    )
    from blond.physics.profiles import ProfileBaseClass


def _get_hash_linspace(
    array1d: NumpyArray | CupyArray, *, salt: Any = None
) -> int:
    """
    Compute a lightweight, approximate hash value for a 1D NumPy array.

    The function samples a few representative elements of the input array
    (first, second, middle, and last), along with the array length, and computes
    a Python built-in hash from this tuple. The result is intended for quick,
    approximate identification of arrays rather than exact equality or integrity
    verification.

    Parameters
    ----------
    array1d : numpy.ndarray
        One-dimensional NumPy array of numeric values.
    salt
        Additional information to generate a hash.

    Returns
    -------
    int
        An integer hash value derived from selected elements of the array.

    Warnings
    --------
    - This function is **not collision-resistant**. Different arrays may yield
      identical hash values, especially if they share similar boundary values or
      lengths.
    - Not suitable for **data integrity**, **deduplication**, or **security**
      purposes. Use `hashlib` (e.g., SHA-256) for robust, deterministic hashing.
    - Assumes a 1D numeric array; no validation is performed. Multi-dimensional
      or non-numeric inputs may cause unexpected behavior or errors.
    - Python’s built-in hash is **not stable across sessions** due to hash
      randomization (unless `PYTHONHASHSEED` is fixed).

    Notes
    -----
    - **Time complexity:** O(1) — the function samples only four elements
      regardless of array size.
    - **Memory usage:** O(1) — constant space overhead.
    - Designed for fast, approximate fingerprinting in performance-sensitive
      contexts where occasional collisions are acceptable.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> get_hash(arr)
    2398472938472938  # Example output (varies by session)
    """
    len_ = len(array1d)
    return hash(
        (
            float(array1d[0]),
            float(array1d[1]),
            float(array1d[int(len_ // 2)]),
            float(array1d[-1]),
            len_,
            salt,
        )
    )


class WakeFieldSolver:
    """Abstract class for a solver that generates wake fields based on beam profiles."""

    @abstractmethod  # pragma: no cover
    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ) -> None:
        """
        Lateinit method when :class:`blond.physics.impedances.base.WakeField` is late-initialized.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        parent_wakefield
            Wakefield that this solver affiliated to.
        """
        pass

    @abstractmethod  # pragma: no cover
    def calc_induced_voltage(
        self, beam: BeamBaseClass
    ) -> NumpyArray | CupyArray:
        """
        Calculate the induced voltage based on the beam profile and beam parameters.

        Parameters
        ----------
        beam
            Simulation object of a particle beam.

        Returns
        -------
        induced_voltage
            Induced voltage, in [V].
        """
        pass

    def _hist_y_to_intensity_factor(
        self,
        beam: BeamBaseClass,
        profile: ProfileBaseClass,
    ) -> float:
        """
        Calculate a conversion factor between histogram values and physical wakefield intensity.

        This factor converts quantities based on macroparticles in a simulation
        to their equivalent real-particle values, taking into account the particle charge,
        beam intensity, and profile scaling.

        Parameters
        ----------
        beam
            `Simulation` object of a particle beam.
        profile
            Beam profile object.

        Returns
        -------
        hist_y_to_intensity_factor
            Factor converting between wakefield
            (macroparticles vs. real particles).
        """
        # TODO this might fail with MOI?
        _factor = (-1 * beam.particle_type.charge * e) * (
            beam.intensity * profile.hist_y_to_density_factor
        )
        return _factor


class WakeFieldSource(ABC):
    """
    General abstract class for wake fields.

    Parameters
    ----------
    is_dynamic
        Whether the wake field source changes with time.
    """

    def __init__(self, is_dynamic: bool):
        self.is_dynamic = is_dynamic


class TimeDomain(ABC):
    """Indication of a source is defined in time domain."""

    def get_wake_per_particle(
        self, time: NumpyArray | CupyArray, counter_rotating: bool = False
    ) -> NumpyArray | CupyArray:
        """
        Point-charge wake (Green's function) sampled at ``time``, in [V].

        Kernel sources override this. Sources that define their impedance
        another way (e.g. InductiveImpedance) do not implement it and instead
        override get_impedance_from_wake.

        Parameters
        ----------
        time
            Time array at which the wake is evaluated, in [s].
        counter_rotating
            If ``True``, use the counter-rotating wake instead of the
            co-rotating one.

        Returns
        -------
        wake
            Point-charge wake, in [V].
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a point-charge wake."
        )

    def get_wake_per_bin(
        self, time: NumpyArray | CupyArray, counter_rotating: bool = False
    ) -> NumpyArray | CupyArray:
        """
        Wake averaged over each sample bin ``[t - dt/2, t + dt/2]``.

        A profile is a histogram, so its induced voltage is the wake
        integrated over each bin, not the wake sampled at the bin centre.
        Point-sampling aliases badly when the wake oscillates several times
        within a bin (the low-Q / broadband resonator bug); bin-averaging
        removes it. Time-domain solvers use this instead of
        :func:`get_wake_per_particle`.

        The default here bin-averages the piecewise-linear interpolant
        through :func:`get_wake_per_particle`, which on a uniform grid
        reduces to the stencil ``(w[n-1] + 6 w[n] + w[n+1]) / 8`` (edges
        extrapolate the boundary value). Exact for a tabulated wake; sources
        with an analytic wake (e.g.
        :class:`~blond.physics.impedances.sources.Resonators`) override this
        with the exact closed-form bin-average instead.

        Parameters
        ----------
        time
            Time array (bin centres) at which the wake is evaluated, in [s].
        counter_rotating
            If ``True``, use the counter-rotating wake instead of the
            co-rotating one.

        Returns
        -------
        wake
            Bin-averaged wake, in [V].
        """
        w = self.get_wake_per_particle(time, counter_rotating)
        wake_prev = backend.concatenate((w[:1], w[:-1]))
        wake_next = backend.concatenate((w[1:], w[-1:]))
        return (wake_prev + 6.0 * w + wake_next) / 8.0

    def _assert_wake_time_resolves_resonances(  # noqa: B027
        self, time: NumpyArray | CupyArray
    ) -> None:
        """
        No-op hook consulted by :func:`get_impedance_from_wake` before FFT-ing the wake.

        Sources whose bin-averaged wake can alias if the sampling grid is too
        coarse (e.g. :class:`~blond.physics.impedances.sources.Resonators`)
        override this to raise instead.

        Parameters
        ----------
        time
            Time array to get wake, in [s].
        """
        pass

    def get_impedance_from_wake(
        self,
        time: NumpyArray | CupyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_fft: int,
        counter_rotating: bool = False,
    ) -> NumpyArray | CupyArray:
        """
        Impedance from the bin-averaged wake: ``rfft(get_wake_per_bin(...))``.

        Keeps a single cached ``(hash, impedance)`` slot per rotation
        direction (co- and counter-rotating), recomputing and overwriting
        that slot whenever ``time`` changes. This bounds the cache to at
        most two entries regardless of how many distinct ``time`` arrays are
        seen over a simulation (e.g. one new array per turn with a dynamic
        profile). Sources whose impedance is not a wake FFT (e.g.
        InductiveImpedance) override this.

        Parameters
        ----------
        time
            Time array to get wake, in [s].
        simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation `Beam` object.
        n_fft
            Number of points to be used in the fft.
        counter_rotating
            If ``True``, use the counter-rotating wake instead of the
            co-rotating one.

        Returns
        -------
        impedance_from_wake
            Impedance array.
        """
        try:
            cache = self._impedance_from_wake_cache
        except AttributeError:
            cache = self._impedance_from_wake_cache = {}
        key = bool(counter_rotating)
        hash_ = _get_hash_linspace(time)
        cached = cache.get(key, (float("nan"), None))
        if cached[0] == hash_:
            impedance = cached[1]
        else:
            self._assert_wake_time_resolves_resonances(time)
            wake = self.get_wake_per_bin(time, counter_rotating)
            impedance = backend.fft.rfft(wake, n=n_fft)
            cache[key] = (hash_, impedance)
        return impedance


class FreqDomain(ABC):
    """Indication of a source is defined in frequency domain."""

    @abstractmethod  # pragma: no cover
    def get_impedance(
        self,
        freq_x: NumpyArray | CupyArray,
        simulation: Simulation,
        beam: BeamBaseClass,
        hist_step: float | None = None,
    ) -> NumpyArray | CupyArray:
        """
        Return the impedance in the frequency domain.

        Parameters
        ----------
        freq_x
            Frequency axis, in [Hz].
        simulation
            Simulation object containing turn index and RF info.
        beam
            Simulation `Beam` object.
        hist_step
            Bin width of the time-domain signal the impedance will be
            applied to, in [s]. `freq_x` alone is ambiguous about the
            signal length (odd vs. even), so sources whose impedance
            depends on the discrete time grid need this; analytic
            sources may ignore it.

        Returns
        -------
        impedance
            Complex impedance array.
        """
        pass


class ImpedanceBaseClass(BeamPhysicsRelevant):
    """
    Abstract class on how to calculate induced voltages.

    Parameters
    ----------
    section_index
        Section index to group elements into sections.
    profile
        Object for calculation of beam profiles.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        section_index: int = 0,
        profile: ProfileBaseClass | None = None,
        **kwargs,
    ):
        super().__init__(section_index=section_index, **kwargs)
        self._profile = profile

    @property  # as readonly attributes
    def profile(self) -> ProfileBaseClass:
        """
        The reference profile that is causing the wake.

        Returns
        -------
        profile
            The reference profile object.
        """
        return self._profile

    @abstractmethod  # pragma: no cover
    def calc_induced_voltage(
        self, beam: BeamBaseClass
    ) -> NumpyArray | CupyArray:
        """
        Calculate the induced voltage based on the beam profile and beam parameters.

        Parameters
        ----------
        beam
            Simulation object of a particle beam.

        Returns
        -------
        induced_voltage
            Induced voltage, in [V].
        """
        pass

    @requires(
        [
            "BeamPhysicsRelevantElements",  # for .section_index,
        ]
    )
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
        from blond.physics.profiles import (
            ProfileBaseClass,  # prevent cyclic import
        )

        if self._profile is None:
            profiles = simulation.ring.elements.get_elements(
                ProfileBaseClass, section_i=self.section_index, recursive=False
            )
            assert len(profiles) == 1, (
                f"Found {len(profiles)} profiles in "
                f"{self.section_index=}, but can only handle one. Set the attribute "
                f"`your_impedance.profile` in advance or remove the second "
                f"profile from this group."
            )
            profile = profiles[0]
        else:
            profile = self._profile
        super().on_init_simulation(simulation, profile=profile, **kwargs)

    def configure(self, *, profile: ProfileBaseClass, **kwargs) -> None:
        """
        Store the profile used for induced-voltage calculations.

        Parameters
        ----------
        profile
            Profile object that provides the beam histogram.
        **kwargs
            Passed to the next level in the MRO chain.
        """
        super().configure(**kwargs)
        self._profile = profile


class WakeField(ImpedanceBaseClass, SupportsPooledInterpolationKickMixIn):
    """
    Manager class to calculate wake-fields.

    Parameters
    ----------
    sources
        List of sources that cause wake-fields.
    solver
        Solver to calculate the induced voltage from the sources.
    section_index
        Section index to group elements into sections.
    profile
        Object for calculation of beam profiles.
    delayed_kick
        The common interface to apply the kick later.
        `PooledInterpolationKick.track(...)` must be executed elsewhere.

    Attributes
    ----------
    sources
        List of sources that cause wake-fields.
    solver
        Solver to calculate the induced voltage from the sources.
    update_induced_voltage
        If ``False``, will not update the induced voltage based on the profile,
        but rather re-use the previously calculated induced voltage.

    See Also
    --------
    blond.physics.impedances.solvers.PeriodicFreqSolver : General solver for wakes in frequency domain.
    blond.physics.impedances.solvers.TimeDomainFftSolver : General solver for wakes in timedomain.
    blond.physics.impedances.solvers.ContinuousMultiTurnTimeDomainSolver : General solver for multi-turn wakes.
    blond.physics.impedances.solvers.InductiveImpedanceSolver : Specialized solver for inductive impedance.
    blond.physics.impedances.solvers.SingleTurnResonatorConvolutionSolver : Specialized solver for `Resonators`.
    blond.physics.impedances.solvers.MultiPassResonatorSolver : Special solver for multi-turn wakes with resonators.

    Examples
    --------
    >>> from blond import StaticProfile, WakeField
    >>> from blond.physics.impedances.solvers import TimeDomainFftSolver
    >>> from blond.physics.impedances.sources import Resonators
    >>>
    >>> profile = StaticProfile(...)
    >>> induced_voltage = WakeField(
    ...     sources=(Resonators(...),),
    ...     solver=TimeDomainFftSolver(),
    ...     profile=profile,
    ... )
    """

    def __init__(
        self,
        sources: tuple[WakeFieldSource, ...],
        solver: WakeFieldSolver,
        section_index: int = 0,
        profile: ProfileBaseClass | None = None,
        delayed_kick: PooledInterpolationKick | None = None,
    ):
        super().__init__(
            section_index=section_index,
            profile=profile,
            delayed_kick=delayed_kick,
        )

        self.solver = solver
        self.sources = sources
        self.update_induced_voltage = True
        self._induced_voltage = None
        self.track_profile = True

    def info_string(self, prefix="") -> str:
        """
        Inform that the profile is also executed within the track method.

        Parameters
        ----------
        prefix
            Prefix string for formatting.

        Returns
        -------
        str
            Information string.
        """
        if self.track_profile:
            content = (
                f"{self.profile.info_string(prefix=prefix + ' ↓ ')}\n"
                f"{super().info_string(prefix=prefix)}"
            )
        else:
            content = super().info_string(prefix=prefix)
        return content

    @property
    def induced_voltage(self) -> NumpyArray | CupyArray:
        """
        Induced voltage in [V] from given beam profile and sources.

        Returns
        -------
        NumpyArray | CupyArray
            Induced voltage array.
        """
        if self._induced_voltage is None:
            raise AttributeError("Use `calc_induced_voltage` first!")
        return self._induced_voltage

    @requires(["MagneticCycleBase"])
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
        assert len(self.sources) > 0, (
            "Provide for at least one `WakeFieldSource`"
        )
        self.solver.on_wakefield_init_simulation(
            simulation=simulation, parent_wakefield=self
        )

    def calc_induced_voltage(
        self, beam: BeamBaseClass
    ) -> NumpyArray | CupyArray:
        """
        Calculate induced voltage from all sources.

        Parameters
        ----------
        beam
            Simulation object of a particle beam.

        Returns
        -------
        induced_voltage
            Induced voltage along the profile, in [V].
        """
        induced_voltage = self.solver.calc_induced_voltage(beam=beam)
        assert len(induced_voltage) >= self.profile.n_bins, (
            f"{type(self.solver).__name__} returned only"
            f" {len(induced_voltage)} samples, but the profile"
            f" has {self.profile.n_bins} bins."
        )
        # Some solvers (e.g. FFT-based convolution) zero-pad to a
        # convenient transform length and return more samples than
        # there are profile bins; only the leading `n_bins` samples
        # correspond to the profile and are physically meaningful.
        self._induced_voltage = induced_voltage[: self.profile.n_bins]
        # the induced voltage has to be provided with the backend precision
        # because the track() method below requires it by calling the backend.
        return self.induced_voltage

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Calculate induced voltage and apply this voltage to the beam.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        if self.profile.active and self.track_profile:
            self.profile.track(beam=beam)
        if self.update_induced_voltage:
            self.calc_induced_voltage(beam=beam)
        induced_voltage = self.induced_voltage
        assert induced_voltage.dtype == backend.float, (
            f"{induced_voltage.dtype}"
        )
        voltage = induced_voltage.astype(backend.float)
        bin_centers = self.profile.hist_x  # base for induced voltage
        if self._delayed_kick is not None:
            # Relies on PooledInterpolationKick.track()
            # being called later.
            self._delayed_kick.register(
                time_axis=bin_centers,
                voltage=voltage,
            )
        else:
            backend.specials.kick_interpolated(
                dt=beam.read_partial_dt(),
                dE=beam.write_partial_dE(),
                # TODO improve induced_voltage calculation data type for speedup
                voltage=voltage,
                bin_centers=bin_centers,  # base for induced voltage
                charge=beam.signed_charge_with_direction(),
                acceleration_kick=0.0,
            )

    @staticmethod
    def headless(
        beam: BeamBaseClass,
        sources: tuple[WakeFieldSource, ...],
        solver: WakeFieldSolver,
        section_index: int = 0,
        profile: ProfileBaseClass | None = None,
    ):
        """
        Initialize the full class.

        Parameters
        ----------
        beam
            The `Beam` object which state will be updated by this element.
        sources
            List of sources that cause wake-fields.
        solver
            Solver to calculate the induced voltage from the sources.
        section_index
            Section index to group elements into sections.
        profile
            Object for calculation of beam profiles.

        Returns
        -------
        wakefield
            Instance with lateinit methods executed.
        """
        wf = WakeField(
            sources=sources,
            solver=solver,
            section_index=section_index,
            profile=profile,
        )
        from unittest.mock import Mock

        from blond.core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        wf.on_init_simulation(simulation=simulation)
        wf.on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=1,
        )
        return wf


class SupportsVectorFittedModel(ABC):
    """
    Mixin to define sources with poles.

    See Also
    --------
    blond.physics.impedances.solvers.MultiPoleSparseSolve : The corresponding wakefield solver.
    """

    @abstractmethod  # pragma: no cover
    def get_vectorfit(self) -> tuple[NumpyArray, NumpyArray, NumpyArray]:
        """
        Derive the poles and residues as in vector-fitting.

        Returns
        -------
        poles
            Complex poles of an equivalent circuit model.
        residues
            Complex residues of an equivalent circuit model.
        counterrotation_signs
            Signs of the poles to deal with higher order oscillators
            in counterrotation. Default is ``1``.
        """
        pass
