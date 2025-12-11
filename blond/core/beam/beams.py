# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Beam definitions based on `BeamBaseClass`."""

from __future__ import annotations

import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from blond.core.backends.backend import backend
from blond.core.beam.base import BeamBaseClass, BeamFlags
from blond.generals.cupy.no_cupy_import import is_cupy_array

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from blond import Simulation
    from blond.core.beam.particle_types import ParticleType


class Beam(BeamBaseClass):
    """
    Initialize a beam of particles for simulation.

    The Beam class represents a collection of macro-particles that model
    the behavior of a real particle beam in an accelerator. Each macro-particle
    represents many real particles and has relative coordinates in time `dt` and
    energy `dE` space.

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

    def __init__(
        self,
        intensity: int | float,
        particle_type: ParticleType,
        is_counter_rotating: bool = False,
    ) -> None:
        super().__init__(
            intensity=intensity,
            particle_type=particle_type,
            is_counter_rotating=is_counter_rotating,
            is_distributed=False,
        )

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Initialize beam parameters when the simulation is created.

        This method is automatically called during simulation initialization
        to set up the beam within the simulation context.

        Parameters
        ----------
        simulation
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
        """
        Configure the beam with an initial particle distributions.

        This method sets the time and energy coordinates for all macro-particles
        in the beam. It must be called before running a simulation to initialize
        the particle distribution.

        Parameters
        ----------
        dt
            Time coordinates of each macro-particle relative to the reference
            time, in [s].
        dE
            Energy coordinates of each macro-particle relative to the reference
            energy, in [eV]. Must have the same length as `dt`.
        flags
            Status flags for each macro-particle (e.g., active, lost).
            If not provided, all particles are set to `ACTIVE` by default.
        reference_time
            The absolute reference time for the coordinate system,
            in [s]. Particle times `dt` are relative to
            this reference.
        reference.total_energy
            The reference total energy for the coordinate system, in [eV].
            Particle energies `dE` are relative to this reference.
        """
        assert len(dt) == len(dE), f"{len(dt)} != {len(dE)}"
        n_macroparticles = len(dt)
        if flags is None:
            flags = np.int32(BeamFlags.ACTIVE.value) * backend.ones(
                n_macroparticles, dtype=np.int32
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
            len(dt), dtype=np.int32
        )

        if reference_time:
            self.reference.time = reference_time
        if reference_total_energy:
            self.reference.total_energy = reference_total_energy

        self.invalidate_cache()

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Prepare the beam before the simulation starts running.

        This method is automatically called when `simulation.run_simulation()`
        is invoked, allowing the beam to perform any necessary setup before
        the turn-by-turn tracking begins.

        Parameters
        ----------
        simulation
            The simulation object managing the beam dynamics.
        beam
            The beam object being simulated (typically this beam itself).
        n_turns
            The total number of turns (revolutions) to simulate.
        turn_i_init
            The starting turn number for the simulation.
        **kwargs
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
        """
        Number of real particles represented by each macro-particle.

        This is the ratio of the total beam intensity (real particles) to the
        number of macro-particles in the simulation. For example, if the beam
        has 1e11 real particles and 1e6 macro-particles, each macro-particle
        represents 1e5 real particles.

        Returns
        -------
        ratio
            The number of real particles per macro-particle.
        """
        warnings.warn(
            "`ratio` might be removed in future.",
            DeprecationWarning,
            stacklevel=1,
        )
        # As there are no weights, lets assume all weights are 1,
        # The sum over all macro-particles with weight 1
        # is thus `common_array_size`.
        return self.intensity / self.common_array_size

    @cached_property
    def dt_min(self) -> float:
        """
        Minimum time coordinate among all macro-particles in the beam in [s].

        Returns
        -------
        dt_min
            Earliest time position in [s], relative to the reference time.
        """
        return self._dt.min()

    @cached_property
    def dt_max(self) -> float:
        """
        Maximum time coordinate among all macro-particles in the beam in [s].

        Returns
        -------
        dt_max
            Latest time position in [s], relative to the reference time.
        """
        return self._dt.max()

    @cached_property
    def dE_min(self) -> float:
        """
        Minimum energy coordinate among all macro-particles in the beam in [eV].

        Returns
        -------
        dE_min
            Lowest energy in [eV], relative to the reference energy.
        """
        return self._dE.min()

    @cached_property
    def dE_max(self) -> float:
        """
        Maximum energy coordinate among all macro-particles in the beam in [eV].

        Returns
        -------
        dE_max
            Highest energy in [eV], relative to the reference energy.
        """
        return self._dE.max()

    @cached_property
    def common_array_size(self) -> int:
        """
        Total number of macro-particles in the beam regardless of `flags` state.

        This property returns the size of the particle arrays (`dt`, `dE`, `flags`).
        For distributed beams, this accounts for particles across all processes.

        Returns
        -------
        common_array_size
            The number of macro-particles being tracked in the simulation.

        Notes
        -----
        Particles that are labeled LOST will be nevertheless counted,
        as they still exist in the array.
        """
        return len(self._dt)

    def plot_hist2d(self, **kwargs) -> None:
        """
        Plot a 2D histogram of the beam distribution.

        Creates a visualization showing the distribution of macro-particles in
        the time-energy distribution (`dt` vs `dE`). This is useful for visualizing
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
        The x-axis represents time `dt` and the y-axis represents energy `dE`.
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
        """
        Scatter-plot of beam coordinates.

        Parameters
        ----------
        **kwargs
            Keyword arguments for ``matplotlib.pyplot.scatter``.
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
        """
        Plot a 1D histogram of beam coordinates along a single axis.

        Creates a histogram showing the distribution of macro-particles projected
        onto either the time axis or the energy axis.

        Parameters
        ----------
        axis
            Which coordinate to plot:
            - 0: Plot time coordinate `dt` distribution
            - 1: Plot energy coordinate `dE` distribution
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
    """
    Create a test beam for probing simulation dynamics.

    A ProbeBeam is a special beam type, designed for testing and
    analysis purposes.

    At least one of `dt` or `dE` must be provided. If only one is given,
    the other coordinate is automatically set to zero for all particles.

    Parameters
    ----------
    particle_type
        The type of particle in the beam (e.g., protons, electrons).
        This determines properties like mass and charge.
    dt
        Time coordinates for the macro-particles, in [s].
        If only `dt` is provided, `dE` will be set to zeros.
        If neither `dt` nor `dE` is provided, an error is raised.
    dE
        Energy coordinates for the macro-particles, in [eV].
        If only `dE` is provided, dt will be set to zeros.
        If neither `dt` nor `dE` is provided, an error is raised.
    reference_time
        The reference time for the coordinate system, in [s].
    reference.total_energy
        The reference total energy for the coordinate system, in [eV].
    intensity
        The beam intensity (number of real particles). Default is 0,
        meaning no collective effects.

    Raises
    ------
    ValueError
        If neither `dt` nor `dE` is provided.
    """

    def __init__(
        self,
        particle_type: ParticleType,
        dt: NumpyArray | None = None,
        dE: NumpyArray | None = None,
        reference_time: float | None = None,
        reference_total_energy: float | None = None,
        intensity: int = 0,
    ) -> None:
        super().__init__(
            intensity=intensity,
            particle_type=particle_type,
        )
        if dt is not None and dE is not None:
            pass
        elif (dE is None) and (dt is None):
            raise ValueError("dE or dt must be given!")
        elif dt is not None:
            dE = backend.zeros_like(dt)
        elif dE is not None:
            dt = backend.zeros_like(dE)
        else:
            raise RuntimeError(
                f"{dE=} {dt=}"
            )  # pragma: no cover Not Reachable

        self.setup_beam(
            dt=dt,
            dE=dE,
            reference_time=reference_time,
            reference_total_energy=reference_total_energy,
        )
