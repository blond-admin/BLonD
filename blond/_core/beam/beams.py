from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from ..._generals.cupy.no_cupy_import import is_cupy_array
from ..backends.backend import backend
from .base import BeamBaseClass, BeamFlags

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from ... import Simulation
    from ..beam.particle_types import ParticleType


class Beam(BeamBaseClass):
    def __init__(
        self,
        intensity: int | float,
        particle_type: ParticleType,
        is_counter_rotating: bool = False,
    ) -> None:
        """Initialize a beam of particles for simulation.

        The Beam class represents a collection of macro-particles that model
        the behavior of a real particle beam in an accelerator. Each macro-particle
        represents many real particles and has coordinates in time (dt) and
        energy (dE) space.

        Parameters
        ----------
        intensity
            The total number of real particles in the beam (beam intensity).
            This is distinct from the number of macro-particles used in the
            simulation, which is typically much smaller.
        particle_type
            The type of particle in the beam (e.g., protons, electrons).
            This determines properties like mass and charge.
        is_counter_rotating
            Whether this beam rotates in the opposite direction to the main beam.
            Default is False (co-rotating beam).
        """
        super().__init__(
            intensity=intensity,
            particle_type=particle_type,
            is_counter_rotating=is_counter_rotating,
            is_distributed=False,
        )

    def on_init_simulation(self, simulation: Simulation) -> None:
        """Initialize beam parameters when the simulation is created.

        This method is automatically called during simulation initialization
        to set up the beam within the simulation context.

        Parameters
        ----------
        simulation : Simulation
            The simulation object that manages this beam.
        """
        super().on_init_simulation(simulation=simulation)

    def setup_beam(
        self,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: NumpyArray | CupyArray | None = None,
        reference_time: float | None = None,
        reference_total_energy: float | None = None,
    ) -> None:
        """Configure the beam with initial particle distributions.

        This method sets the time and energy coordinates for all macro-particles
        in the beam. It must be called before running a simulation to initialize
        the particle distribution.

        Parameters
        ----------
        dt : array_like
            Time coordinates of each macro-particle relative to the reference
            time, in seconds [s]. The array length determines the number of
            macro-particles in the simulation.
        dE : array_like
            Energy coordinates of each macro-particle relative to the reference
            energy, in electron-volts [eV]. Must have the same length as dt.
        flags : array_like, optional
            Status flags for each macro-particle (e.g., active, lost).
            If not provided, all particles are set to ACTIVE by default.
        reference_time : float, optional
            The reference time (t=0) for the coordinate system, in seconds [s].
            Particle times (dt) are measured relative to this reference.
        reference_total_energy : float, optional
            The reference total energy (E=0) for the coordinate system, in
            electron-volts [eV]. Particle energies (dE) are measured relative
            to this reference.
        """
        assert len(dt) == len(dE), f"{len(dt)} != {len(dE)}"
        n_macroparticles = len(dt)
        if flags is None:
            flags = backend.int(BeamFlags.ACTIVE.value) * backend.ones(
                n_macroparticles, dtype=backend.int
            )
        else:
            assert flags.max() <= BeamFlags.ACTIVE.value
            assert len(dt) == len(flags)

        self._dE: NumpyArray | CupyArray = backend.array(
            dE, dtype=backend.float
        )
        self._dt: NumpyArray | CupyArray = backend.array(
            dt, dtype=backend.float
        )

        # intentionally 32 bit, this should be enough for all thinkable flags
        self._flags: NumpyArray | CupyArray = flags.astype(np.int32)

        self._ids: NumpyArray | CupyArray = backend.arange(
            len(dt), dtype=backend.int
        )

        if reference_time:
            self.reference_time = reference_time
        if reference_total_energy:
            self.reference_total_energy = reference_total_energy

        self.invalidate_cache()

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """Prepare the beam before the simulation starts running.

        This method is automatically called when `simulation.run_simulation()`
        is invoked, allowing the beam to perform any necessary setup before
        the turn-by-turn tracking begins.

        Parameters
        ----------
        simulation : Simulation
            The simulation object managing the beam dynamics.
        beam : BeamBaseClass
            The beam object being simulated (typically this beam itself).
        n_turns : int
            The total number of turns (revolutions) to simulate.
        turn_i_init : int
            The starting turn number for the simulation.
        **kwargs : dict
            Additional keyword arguments for simulation setup.
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
            turn_i_init=turn_i_init,
        )

    @property
    def ratio(self) -> float:
        """Number of real particles represented by each macro-particle.

        This is the ratio of the total beam intensity (real particles) to the
        number of macro-particles in the simulation. For example, if the beam
        has 1e11 real particles and 1e6 macro-particles, each macro-particle
        represents 1e5 real particles.

        Returns
        -------
        float
            The number of real particles per macro-particle.
        """
        # As there are no weights, lets assume all weights are 1,
        # The sum over all macro-particles with weight 1
        # is thus `common_array_size`.
        return self.intensity / self.common_array_size

    @cached_property
    def dt_min(self) -> np.int32 | np.int64:
        """Minimum time coordinate among all macro-particles in the beam.

        Returns
        -------
        float
            Earliest time position in seconds [s], relative to the reference time.
        """
        return self._dt.min()

    @cached_property
    def dt_max(self) -> np.int32 | np.int64:
        """Maximum time coordinate among all macro-particles in the beam.

        Returns
        -------
        float
            Latest time position in seconds [s], relative to the reference time.
        """
        return self._dt.max()

    @cached_property
    def dE_min(self) -> np.int32 | np.int64:
        """Minimum energy coordinate among all macro-particles in the beam.

        Returns
        -------
        float
            Lowest energy in electron-volts [eV], relative to the reference energy.
        """
        return self._dE.min()

    @cached_property
    def dE_max(self) -> np.int32 | np.int64:
        """Maximum energy coordinate among all macro-particles in the beam.

        Returns
        -------
        float
            Highest energy in electron-volts [eV], relative to the reference energy.
        """
        return self._dE.max()

    @cached_property
    def common_array_size(self) -> int:
        """Total number of macro-particles in the beam.

        This property returns the size of the particle arrays (dt, dE, flags).
        For distributed beams, this accounts for particles across all processes.

        Returns
        -------
        int
            The number of macro-particles being tracked in the simulation.
        """
        return len(self._dt)

    def plot_hist2d(self, **kwargs) -> None:
        """Plot a 2D histogram of the beam distribution in phase space.

        Creates a visualization showing the distribution of macro-particles in
        the time-energy phase space (dt vs dE). This is useful for visualizing
        the beam shape, density, and any structures in the distribution.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to ``matplotlib.pyplot.hist2d``.
            Common options include:
            - bins: number of bins (default: 256)
            - cmap: colormap (default: 'viridis')
            - range: data range [[xmin, xmax], [ymin, ymax]]

        Notes
        -----
        The x-axis represents time (dt) and the y-axis represents energy (dE).
        """
        if self._dt is None or self._dE is None:
            raise ValueError(
                "Beam `dt` and `dE` coordinates are not initialized!"
            )
        if "cmap" not in kwargs:
            kwargs["cmap"] = "viridis"
        if "bins" not in kwargs:
            kwargs["bins"] = 256
        if is_cupy_array(self._dt):
            # variables below are just for the type hints to function correctly
            dE: CupyArray = self._dE
            dt: CupyArray = self._dt
            plt.hist2d(dt.get(), dE.get(), **kwargs)
        else:
            plt.hist2d(self._dt, self._dE, **kwargs)

    def plot_scatter(self, **kwargs) -> None:
        """Scatter-plot of beam coordinates.

        Parameters
        ----------
        kwargs
            Keyword arguments for ``matplotlib.pyplot.scatter``
        """
        if self._dt is None or self._dE is None:
            raise ValueError(
                "Beam `dt` and `dE` coordinates are not initialized!"
            )
        if is_cupy_array(self._dt):
            # variables below are just for the type hints to function correctly
            dE: CupyArray = self._dE
            dt: CupyArray = self._dt
            plt.scatter(dt.get(), dE.get(), **kwargs)
        else:
            plt.scatter(self._dt, self._dE, **kwargs)

    def plot_hist(self, axis=0, **kwargs) -> None:
        """Plot a 1D histogram of beam coordinates along a single axis.

        Creates a histogram showing the distribution of macro-particles projected
        onto either the time axis or the energy axis.

        Parameters
        ----------
        axis : int, optional
            Which coordinate to plot:
            - 0: Plot time coordinate (dt) distribution
            - 1: Plot energy coordinate (dE) distribution
            Default is 0 (time).
        **kwargs
            Additional keyword arguments passed to ``matplotlib.pyplot.hist``.
            Common options include:
            - bins: number of bins (default: 256)
            - range: data range (min, max)
            - density: if True, normalize to form a probability density

        """
        if self._dt is None or self._dE is None:
            raise ValueError(
                "Beam `dt` and `dE` coordinates are not initialized!"
            )
        if "bins" not in kwargs:
            kwargs["bins"] = 256
        if is_cupy_array(self._dt):
            # variables below are just for the type hints to function correctly
            dE: CupyArray = self._dE
            dt: CupyArray = self._dt
            if axis == 0:
                xs = dt.get()
            elif axis == 1:
                xs = dE.get()
            else:
                raise ValueError(f"{axis=}")
        elif axis == 0:
            xs = self._dt
        elif axis == 1:
            xs = self._dE
        else:
            raise ValueError(f"{axis=}")
        plt.hist(xs, **kwargs)


class ProbeBeam(Beam):
    def __init__(
        self,
        particle_type: ParticleType,
        dt: NumpyArray | None = None,
        dE: NumpyArray | None = None,
        reference_time: float | None = None,
        reference_total_energy: float | None = None,
        intensity: int = 0,
    ) -> None:
        """Create a test beam for probing simulation dynamics without collective effects.

        A ProbeBeam is a special beam type with zero (or negligible) intensity,
        designed for testing and analysis purposes. Since it has no intensity,
        it doesn't generate collective effects (space charge, wake fields) but
        still experiences the accelerator's fields. This is useful for:

        - Testing single-particle dynamics
        - Exploring phase space acceptance
        - Creating specialized distributions (e.g., only time or only energy coordinates)

        At least one of `dt` or `dE` must be provided. If only one is given,
        the other coordinate is automatically set to zero for all particles.

        Parameters
        ----------
        particle_type : ParticleType
            The type of particle in the beam (e.g., protons, electrons).
            This determines properties like mass and charge.
        dt : array_like, optional
            Time coordinates for the macro-particles, in seconds [s].
            If only dt is provided, dE will be set to zeros.
            If neither dt nor dE is provided, an error is raised.
        dE : array_like, optional
            Energy coordinates for the macro-particles, in electron-volts [eV].
            If only dE is provided, dt will be set to zeros.
            If neither dt nor dE is provided, an error is raised.
        reference_time : float, optional
            The reference time (t=0) for the coordinate system, in seconds [s].
        reference_total_energy : float, optional
            The reference total energy (E=0) for the coordinate system, in
            electron-volts [eV].
        intensity : int, optional
            The beam intensity (number of real particles). Default is 0,
            meaning no collective effects.

        Raises
        ------
        ValueError
            If neither dt nor dE is provided.
        """
        super().__init__(
            intensity=intensity,
            particle_type=particle_type,
        )
        if dt is not None:
            dE = backend.zeros_like(dt)
        elif dE is not None:
            dt = backend.zeros_like(dE)
        elif (dE is None) and (dt is None):
            raise ValueError("dE or dt must be given!")

        else:
            raise RuntimeError(f"{dE=} {dt=}")

        self.setup_beam(
            dt=dt,
            dE=dE,
            reference_time=reference_time,
            reference_total_energy=reference_total_energy,
        )
