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
from blond.core.backends.mpi_distributed.callables import rms_emittance
from blond.core.beam.base import BeamBaseClass, BeamFlags
from blond.generals.cupy.no_cupy_import import is_cupy_array
from blond.generals.distributed.distributed_array import DistributedArray
from blond.generals.distributed.helpers import distributed_arange

if TYPE_CHECKING:  # pragma: no cover
    from typing import Literal

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from matplotlib.collections import QuadMesh
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
        mpi_mode: Literal["root-distributes", "all-ranks"] = "all-ranks",
        **kwargs,
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
        reference_total_energy
            The reference total energy for the coordinate system, in [eV].
            Particle energies `dE` are relative to this reference.
        mpi_mode
            - "root-distributes": The array is distributed from the root node to all ranks.
            - "all-ranks":  All ranks setup the beam independently.
        **kwargs
            Unused - Keyword arguments to make the non-abstract implementation
            extendable.
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

        self._dE: DistributedArray = DistributedArray(
            backend.array(dE, dtype=backend.float)
        )
        self._dt: DistributedArray = DistributedArray(
            backend.array(dt, dtype=backend.float)
        )

        # intentionally 32 bit, this should be enough for all thinkable flags
        self._flags: DistributedArray = DistributedArray(
            backend.array(flags, dtype=np.int32)
        )

        if reference_time:
            self.reference.time = reference_time
        if reference_total_energy:
            self.reference.total_energy = reference_total_energy

        if mpi_mode == "root-distributes":
            self._dE.mpi_scatter()
            self._dt.mpi_scatter()
            self._flags.mpi_scatter()
            # IDs need special treatment
            self._ids: DistributedArray = DistributedArray(
                backend.arange(len(dt), dtype=np.int32)
            )
            self._ids.mpi_scatter()
        elif mpi_mode == "all-ranks":
            # IDs need special treatment
            self._ids: DistributedArray = distributed_arange(
                len(dt), dtype=np.int32
            )
        else:
            raise NameError(f"Unknown {mpi_mode=}")

        self.invalidate_cache()

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
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
        **kwargs
            Additional keyword arguments for simulation setup.
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
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
    def rms_emittance(self):
        """
        Calculate the Root-Mean-Square emittance of the beam.

        Returns
        -------
        rms_emittance
            The Root-Mean-Square emittance in [s eV] of the beam.
        """
        return rms_emittance(dt=self._dt, dE=self._dE)

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
        return self._dt.global_size

    def plot_hist2d(self, **kwargs) -> QuadMesh:
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

        Returns
        -------
        image
            `matplotlib.collections.QuadMesh` object.

        Notes
        -----
        The x-axis represents time `dt` and the y-axis represents energy `dE`.
        """
        from blond.generals.distributed.distributed_array import (
            mpi_is_distributed,
        )

        if self._dt is None or self._dE is None:
            raise ValueError(
                "Beam `dt` and `dE` coordinates are not initialized!"
            )
        if "cmap" not in kwargs:
            kwargs["cmap"] = "viridis"
        if "bins" not in kwargs:
            kwargs["bins"] = 256
        if mpi_is_distributed():
            warnings.warn(
                "Plotting MPI single node distribution only.",
                UserWarning,
                stacklevel=2,
            )
        if is_cupy_array(self._dt.array_local):
            # variables below are just for the type hints to function correctly
            dE: CupyArray = self._dE.array_local
            dt: CupyArray = self._dt.array_local
            counts, xedges, yedges, image = plt.hist2d(
                dt.get(), dE.get(), **kwargs
            )
        else:
            counts, xedges, yedges, image = plt.hist2d(
                self._dt.array_local, self._dE.array_local, **kwargs
            )
        return image

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
        if is_cupy_array(self._dt.array_local):
            # variables below are just for the type hints to function correctly
            dE: CupyArray = self._dE.array_local
            dt: CupyArray = self._dt.array_local
            plt.scatter(dt.get(), dE.get(), **kwargs)
        else:
            plt.scatter(self._dt.array_local, self._dE.array_local, **kwargs)

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

        dE = self._dE.array_local
        dt = self._dt.array_local

        if is_cupy_array(dE):  # assume dt is the same like `dt`
            if axis == 0:
                dt = dt.get()
            elif axis == 1:
                dE = dE.get()
            else:
                raise ValueError(f"{axis=}")

        if axis == 0:
            xs = dt
        elif axis == 1:
            xs = dE
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
    reference_total_energy
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
        intensity: float = 0,
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


class EmptyBeam(Beam):
    """
    Create a beam without `dt`, `dE` coordinates for probing simulation dynamics.

    A EmptyBeam is a special beam type, designed for testing and
    analysis purposes.

    Parameters
    ----------
    particle_type
        The type of particle in the beam (e.g., protons, electrons).
        This determines properties like mass and charge.
    reference_time
        The reference time for the coordinate system, in [s].
    reference_total_energy
        The reference total energy for the coordinate system, in [eV].
    intensity
        The beam intensity (number of real particles). Default is 0,
        meaning no collective effects.
    """

    def __init__(
        self,
        particle_type: ParticleType,
        reference_time: float | None = None,
        reference_total_energy: float | None = None,
        intensity: float = 0,
    ) -> None:
        super().__init__(
            intensity=intensity,
            particle_type=particle_type,
        )
        self.setup_beam(
            dt=backend.zeros(0),
            dE=backend.zeros(0),
            reference_time=reference_time,
            reference_total_energy=reference_total_energy,
        )
