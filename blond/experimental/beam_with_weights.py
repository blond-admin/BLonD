# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Beam with per-macro-particle weights."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond import Beam
from blond.core.backends.backend import backend
from blond.core.beam.flags import BeamFlags
from blond.core.beam.particle_types import ParticleType
from blond.generals.distributed.distributed_array import DistributedArray
from blond.generals.distributed.helpers import mpi_is_distributed

if TYPE_CHECKING:  # pragma: no cover
    from typing import Literal

    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray


class WeightenedBeam(Beam):
    """
    A beam where each macro-particle carries an individual weight.

    Instead of every macro-particle representing ``intensity / n_macroparticles``
    real particles, each macro-particle *i* represents
    ``intensity * w_i / sum(w)`` real particles.  This allows non-uniform
    sampling distributions (e.g. importance sampling, merging beams of
    different sizes, or coarse-graining a distribution).

    Parameters
    ----------
    intensity
        Total number of real particles (beam intensity).
    particle_type
        Type of particle, e.g. ``proton``.
    is_counter_rotating
        Whether this beam counter-rotates. Default ``False``.
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
        )
        self._weights: DistributedArray | None = None

    def is_set_up(self) -> bool:
        """
        ``True`` if all required arrays (including weights) are initialized.

        Returns
        -------
        is_set_up
            ``True`` if all required arrays are initialized.
        """
        return super().is_set_up() and self._weights is not None

    def setup_beam(
        self,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: NumpyArray | CupyArray | None = None,
        reference_time: float | None = None,
        reference_total_energy: float | None = None,
        mpi_mode: Literal["root-distributes", "all-ranks"] = "all-ranks",
        weights: NumpyArray | CupyArray | None = None,
        **kwargs,
    ) -> None:
        """
        Configure the beam with particle coordinates and per-particle weights.

        Parameters
        ----------
        dt
            Time coordinates of each macro-particle relative to the reference
            time, in [s].
        dE
            Energy coordinates of each macro-particle relative to the reference
            energy, in [eV]. Must have the same length as ``dt``.
        flags
            Status flags for each macro-particle. If ``None``, all particles
            are marked ``ACTIVE``.
        reference_time
            Absolute reference time, in [s].
        reference_total_energy
            Reference total energy, in [eV].
        mpi_mode
            How particle data is distributed across MPI ranks:
            ``"root-distributes"`` or ``"all-ranks"``.
        weights
            Per-macro-particle weight. Must have the same length as ``dt``.
            Defaults to uniform weights (all ones) if ``None``.
        **kwargs
            Forwarded to ``Beam.setup_beam``.
        """
        if weights is None:
            weights = np.ones(len(dt), dtype=np.float64)
        assert len(dt) == len(weights), f"{len(dt)} != {len(weights)}"

        super().setup_beam(
            dt=dt,
            dE=dE,
            flags=flags,
            reference_time=reference_time,
            reference_total_energy=reference_total_energy,
            mpi_mode=mpi_mode,
            **kwargs,
        )

        self._weights: DistributedArray = DistributedArray(
            backend.array(weights, dtype=backend.float)
        )

        if mpi_mode == "root-distributes":
            self._weights.mpi_scatter()
        elif mpi_mode == "all-ranks":
            pass
        else:
            raise NameError(f"Unknown {mpi_mode=}")

    @property
    def weights(self) -> DistributedArray:
        """
        Per-macro-particle weights as a distributed array.

        Returns the full :class:`~blond.generals.distributed.distributed_array.DistributedArray`
        so that callers can access both ``.array_local`` for kernel calls and
        ``.sum()`` for MPI-wide reductions.

        Returns
        -------
        weights
            Distributed weight array for this beam.
        """
        return self._weights

    @property
    def ratio(self) -> float:
        """
        Number of real particles represented per unit weight.

        ``intensity / sum(weights)`` — multiply a macro-particle's weight by
        this value to get the number of real particles it represents.
        The sum is computed MPI-wide via :meth:`DistributedArray.sum`.

        Returns
        -------
        ratio
            Particles per unit weight.
        """
        return self.intensity / self._weights.sum()

    @property
    def dt_min(self) -> float:
        """
        Minimum time coordinate among active (weight > 0) macro-particles, in [s].

        Returns
        -------
        dt_min
            Earliest active time position in [s], relative to the reference time.
        """
        return self._dt.min(weights=self._weights)

    @property
    def dt_max(self) -> float:
        """
        Maximum time coordinate among active (weight > 0) macro-particles, in [s].

        Returns
        -------
        dt_max
            Latest active time position in [s], relative to the reference time.
        """
        return self._dt.max(weights=self._weights)

    @property
    def dt_mean(self) -> float:
        """
        Weighted mean time coordinate, in [s].

        Returns
        -------
        dt_mean
            Weight-averaged time position in [s].
        """
        return self._dt.mean(weights=self._weights)

    @property
    def dt_std(self) -> float:
        """
        Weighted standard deviation of time coordinates, in [s].

        Returns
        -------
        dt_std
            Weighted standard deviation of time positions in [s].
        """
        return self._dt.std(weights=self._weights)

    @property
    def dE_min(self) -> float:
        """
        Minimum energy coordinate among active (weight > 0) macro-particles, in [eV].

        Returns
        -------
        dE_min
            Lowest active energy in [eV], relative to the reference energy.
        """
        return self._dE.min(weights=self._weights)

    @property
    def dE_max(self) -> float:
        """
        Maximum energy coordinate among active (weight > 0) macro-particles, in [eV].

        Returns
        -------
        dE_max
            Highest active energy in [eV], relative to the reference energy.
        """
        return self._dE.max(weights=self._weights)

    @property
    def dE_mean(self) -> float:
        """
        Weighted mean energy coordinate, in [eV].

        Returns
        -------
        dE_mean
            Weight-averaged energy in [eV].
        """
        return self._dE.mean(weights=self._weights)

    @property
    def dE_std(self) -> float:
        """
        Weighted standard deviation of energy coordinates, in [eV].

        Returns
        -------
        dE_std
            Weighted standard deviation of energy positions in [eV].
        """
        return self._dE.std(weights=self._weights)

    @property
    def rms_emittance(self) -> float:
        """
        Weighted RMS emittance of the beam in [s eV].

        Uses the same raw-second-moment formula as the unweighted version,
        but replaces uniform averaging with weight-averaged moments:

        ``sqrt(E_w[dt²] · E_w[dE²] − E_w[dt·dE]²)``

        where ``E_w[x] = sum(w·x) / sum(w)``.

        Returns
        -------
        rms_emittance
            Weighted root-mean-square emittance in [s eV].
        """
        w = self._weights.array_local
        dt = self._dt.array_local
        dE = self._dE.array_local

        local_w_sum = float(w.sum())
        local_wdt2_sum = float(backend.dot(w * dt, dt))
        local_wdE2_sum = float(backend.dot(w * dE, dE))
        local_wdtdE_sum = float(backend.dot(w * dt, dE))

        if mpi_is_distributed():
            from mpi4py import MPI  # type: ignore

            comm = MPI.COMM_WORLD
            w_sum = comm.allreduce(local_w_sum, op=MPI.SUM)
            wdt2_sum = comm.allreduce(local_wdt2_sum, op=MPI.SUM)
            wdE2_sum = comm.allreduce(local_wdE2_sum, op=MPI.SUM)
            wdtdE_sum = comm.allreduce(local_wdtdE_sum, op=MPI.SUM)
        else:
            w_sum = local_w_sum
            wdt2_sum = local_wdt2_sum
            wdE2_sum = local_wdE2_sum
            wdtdE_sum = local_wdtdE_sum

        over_w = 1.0 / w_sum
        rms = np.sqrt(
            max(
                (wdt2_sum * over_w) * (wdE2_sum * over_w)
                - (wdtdE_sum * over_w) ** 2,
                0.0,
            )
        )
        return float(rms)

    def purge_flagged_entries(self, flag: int = BeamFlags.LOST.value) -> None:
        """
        Delete flagged macro-particles from all arrays, including weights.

        Parameters
        ----------
        flag
            The flag value used to select particles for removal.
            Default removes ``LOST`` particles.
        """
        # Capture the survival mask *before* the parent reorders and truncates
        # the flags array.  move_flagged_elements_to_end is a stable partition,
        # so the survivors keep their relative order, and mask-indexing weights
        # gives the correct subset in the correct order.
        mask = self._flags.array_local != flag
        super().purge_flagged_entries(flag=flag)
        self._weights.array_local = self._weights.array_local[mask]

    def plot_hist2d(self, **kwargs) -> None:
        """
        Plot a 2D histogram of the beam distribution, weighted by particle weights.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to ``matplotlib.pyplot.hist2d``.
        """
        import matplotlib.pyplot as plt

        from blond.generals.cupy.no_cupy_import import is_cupy_array

        if "cmap" not in kwargs:
            kwargs["cmap"] = "viridis"
        if "bins" not in kwargs:
            kwargs["bins"] = 256

        dt = self._dt.array_local
        dE = self._dE.array_local
        w = self._weights.array_local

        if is_cupy_array(dt):
            dt = dt.get()
            dE = dE.get()
            w = w.get()

        return plt.hist2d(dt, dE, weights=w, **kwargs)[3]

    def plot_scatter(self, ax=None, **kwargs) -> None:
        """
        Scatter-plot of beam coordinates with marker size proportional to weight.

        Parameters
        ----------
        ax
            Pyplot axis object, for example ``ax = plt.gca()``.
        **kwargs
            Keyword arguments for ``matplotlib.pyplot.scatter``.
            If ``s`` is not provided it is set to the local weight array so
            that marker area encodes particle weight.
        """
        import matplotlib.pyplot as plt

        from blond.generals.cupy.no_cupy_import import is_cupy_array

        if ax is None:
            ax = plt

        dt = self._dt.array_local
        dE = self._dE.array_local
        w = self._weights.array_local

        if is_cupy_array(dt):
            dt = dt.get()
            dE = dE.get()
            w = w.get()

        if "s" not in kwargs:
            kwargs["s"] = w

        return ax.scatter(dt, dE, **kwargs)

    def plot_hist(self, axis: int = 0, **kwargs) -> None:
        """
        Plot a weighted 1D histogram of beam coordinates along a single axis.

        Parameters
        ----------
        axis
            Which coordinate to plot: 0 for ``dt``, 1 for ``dE``.
        **kwargs
            Additional keyword arguments passed to ``matplotlib.pyplot.hist``.
        """
        import matplotlib.pyplot as plt

        from blond.generals.cupy.no_cupy_import import is_cupy_array

        if "bins" not in kwargs:
            kwargs["bins"] = 256

        dt = self._dt.array_local
        dE = self._dE.array_local
        w = self._weights.array_local

        if is_cupy_array(dt):
            dt = dt.get()
            dE = dE.get()
            w = w.get()

        if axis == 0:
            xs = dt
        elif axis == 1:
            xs = dE
        else:
            raise ValueError(f"{axis=}")

        plt.hist(xs, weights=w, **kwargs)

    @staticmethod
    def from_beam(beam: Beam) -> WeightenedBeam:
        """
        Convert a uniform-weight :class:`~blond.core.beam.beams.Beam` into a
        :class:`WeightenedBeam` with uniform weights (all ones).

        Parameters
        ----------
        beam
            An already set-up ``Beam`` instance.

        Returns
        -------
        weighted_beam
            A ``WeightenedBeam`` with the same coordinates as *beam* and
            uniform weights.
        """
        wb = WeightenedBeam(
            intensity=beam.intensity,
            particle_type=beam.particle_type,
            is_counter_rotating=beam.is_counter_rotating,
        )
        n = beam._dt.local_size
        wb.setup_beam(
            dt=beam._dt.array_local.copy(),
            dE=beam._dE.array_local.copy(),
            flags=beam._flags.array_local.copy(),
            weights=np.ones(n, dtype=np.float64),
            reference_time=float(beam.reference.time),
            reference_total_energy=beam.reference._total_energy,
            mpi_mode="all-ranks",
        )
        return wb
