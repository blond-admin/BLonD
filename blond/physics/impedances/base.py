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
from blond.generals.hashing_ import hash_linspace
from blond.physics.profiles_sparse import EquidistantMultiProfile

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation
    from blond.experimental.physics.kick_pooling import (
        PooledInterpolationKick,
    )
    from blond.physics.profiles import ProfileBaseClass


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
        Wake averaged over the source bin, the observation bin and one more.

        A profile is a histogram, so its induced voltage is the wake
        integrated over each bin, not the wake sampled at the bin centre.
        Point-sampling aliases badly when the wake oscillates several times
        within a bin (the low-Q / broadband resonator bug). Averaging over
        both the source and the observation bin removes that amplitude error
        but leaves a half-bin lag, because it models the beam as a staircase
        whose derivative sits exactly on the bin edges; averaging over a third
        box -- equivalently, reconstructing the line density as piecewise
        linear rather than as a staircase -- removes the lag as well. The wake
        is therefore weighted with the quadratic B-spline
        ``box * box * box`` (``sinc(f dt)**3``), whose support straddles the
        causal onset symmetrically. See
        ``Resonators._wake_bin_average``
        for the full argument and the numbers. Time-domain solvers use this
        instead of :func:`get_wake_per_particle`.

        The kernel is non-zero from ``-1.5 * dt``, i.e. it has one non-causal
        tap: the voltage of a bin depends on the charge of the next one. That
        is an artefact of interpolating the line density, not acausality.
        Callers must sample ``time`` from ``-dt`` to pick the tap up;
        :func:`get_impedance_from_wake` does.

        The default here B-spline-averages the piecewise-linear interpolant
        through :func:`get_wake_per_particle`, which on a uniform grid reduces
        to the stencil ``(w[n-2] + 76 w[n-1] + 230 w[n] + 76 w[n+1]
        + w[n+2]) / 384`` (edges extrapolate the boundary value). It is
        therefore exact only where the wake really is piecewise linear
        between the samples spanned by the stencil -- e.g. for a tabulated
        wake sampled on the same grid, away from any step. Across a
        discontinuity, in particular the causal onset of a resonator wake, the
        stencil smears the step over its support and the result is off by a
        few percent of the wake amplitude. Sources with an analytic wake (e.g.
        :class:`~blond.physics.impedances.sources.Resonators`) override this
        with the exact closed-form result instead.

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
        prev_1 = backend.concatenate((w[:1], w[:-1]))
        prev_2 = backend.concatenate((w[:1], w[:1], w[:-2]))
        next_1 = backend.concatenate((w[1:], w[-1:]))
        next_2 = backend.concatenate((w[2:], w[-1:], w[-1:]))
        return (
            prev_2 + 76.0 * prev_1 + 230.0 * w + 76.0 * next_1 + next_2
        ) / 384.0

    def _assert_wake_time_resolves_resonances(  # noqa: B027
        self, time: NumpyArray | CupyArray
    ) -> None:
        """
        No-op hook consulted by :func:`get_impedance_from_wake` before FFT-ing the wake.

        Extension point for a source that wants to reject a sampling grid it
        cannot represent (e.g. one too coarse for its bin-averaged wake, which
        would alias). No source currently overrides it, so the call is
        unconditionally a no-op.

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
        Impedance from the bin-averaged wake, ``rfft(get_wake_per_bin(...))``.

        The wake is sampled from ``time - dt`` rather than from ``time``, so
        that the kernel's one non-causal tap (see :func:`get_wake_per_bin`) is
        picked up, and the resulting spectrum is advanced by one sample again
        to put lag zero back where the caller expects it. The convolution the
        caller performs is zero-padded, so the sample the advance pulls in
        past the profile's last bin is padding, not a wrap-around.

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
        hash_ = hash_linspace(time)
        cached = cache.get(key, (float("nan"), None))
        if cached[0] == hash_:
            impedance = cached[1]
        else:
            self._assert_wake_time_resolves_resonances(time)
            bin_step = time[1] - time[0]
            wake = self.get_wake_per_bin(time - bin_step, counter_rotating)
            # `n_fft=None` means "transform the wake as it is".
            n_transform = len(wake) if n_fft is None else n_fft
            # Undo the one-sample delay the shifted sampling introduced.
            advance = backend.exp(
                1j
                * backend.twopi
                * backend.arange(n_transform // 2 + 1, dtype=backend.float)
                / n_transform
            )
            impedance = backend.fft.rfft(wake, n=n_fft) * advance
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
        sparse_metadata = (
            self.profile.sparse_kick_metadata
            if isinstance(self.profile, EquidistantMultiProfile)
            else None
        )
        if self._delayed_kick is not None:
            # Relies on PooledInterpolationKick.track()
            # being called later.
            self._delayed_kick.register(
                time_axis=bin_centers,
                voltage=voltage,
                sparse_metadata=sparse_metadata,
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
                **(sparse_metadata or {}),
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
