# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Generalized functions to deal with Beam objects."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import backend
from blond.core.base import Preparable
from blond.core.beam.flags import BeamFlags
from blond.core.helpers import int_from_float_with_warning
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.core.ring.helpers import requires
from blond.generals.cupy_ import no_cupy_import
from blond.generals.distributed import distributed_array
from blond.generals.distributed import helpers as dist_help

if TYPE_CHECKING:  # pragma: no cover
    from os import PathLike
    from typing import Any, Literal, Self

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.particle_types import ParticleType
    from blond.core.simulation.simulation import Simulation
    from blond.generals.distributed.distributed_array import DistributedArray


class BeamBaseClass(Preparable, ABC):
    """
    Base class to make beam classes.

    Parameters
    ----------
    intensity
        Actual/real number of particles.
        a.k.a. beam intensity.
    particle_type
        Type of particles, e.g. protons.
    is_counter_rotating
        If this is a normal or counter-rotating beam.
    is_distributed
        Developer option to allow distributed computing.
    """

    def __init__(
        self,
        intensity: int | float,
        particle_type: ParticleType,
        is_counter_rotating: bool = False,
        is_distributed: bool = False,
    ) -> None:
        super().__init__()

        self.intensity = int_from_float_with_warning(
            intensity, warning_stacklevel=2
        )
        self._is_distributed = is_distributed
        self._is_counter_rotating = is_counter_rotating

        # should be initialized later using `setup_beam`
        self._dE: DistributedArray | None = None
        self._dt: DistributedArray | None = None
        self._flags: DistributedArray | None = None
        self._ids: DistributedArray | None = None

        self.reference = ReferenceCoordinates(
            time=0, total_energy=None, particle_type=particle_type
        )

    def __iadd__(self, other: Self) -> Self:
        """
        In place addition of another beam to this one.

        See `add_beam` for full documentation.

        Parameters
        ----------
        other
            The beam object to be added to this one.

        Returns
        -------
        self
            Self with the contents of other beam added.

        Examples
        --------
        >>> beam_1 = Beam([..])
        >>> beam_2 = Beam([..])
        >>> beam_1 += beam_2
        """
        self.add_beam(other)
        return self

    def add_beam(self, other: Self):
        """
        Add another beam to this one, mutates this beam.

        The particles from the other beam will be concatenated with the
        particles of this one.  The MPI distribution status, intensity
        ratio and particle types must match.

        The ``ids`` of the added beam will be incremented by the maximum id
        of the current beam plus one.  E.g.:
            ``self.ids = [0, 2, 4]``
            ``other.ids = [0, 1, 2, 3, 4]``
        After addition:
            ``self.ids = [0, 2, 4, 5, 6, 7, 8, 9]``

        Parameters
        ----------
        other
            The beam object to be added to this one.

        Raises
        ------
        RuntimeError
            Raised if one beam is distributed and one is not.
        ValueError
            Raised if the ratio values are not exactly equal.
            Raised if the particle types are not equal.
        """
        if self.is_distributed != other.is_distributed:
            raise RuntimeError(
                "A non-distributed beam cannot be added to a distributed beam."
                f"{self.is_distributed=}, {other.is_distributed=}"
            )

        if self.ratio != other.ratio:
            raise ValueError(
                "Beams can only be added if they have the same ratio."
                f"{self.ratio=}, {other.ratio=}"
            )

        if self.particle_type != other.particle_type:
            raise ValueError(
                "Cannot add beams with mismatched particle types."
                f"{self.particle_type=}, {other.particle_type=}"
            )

        new_ids = other._ids.array_local + int(self._ids.max()) + 1

        self._add_coordinates(
            other._dt,
            other._dE,
            other._flags,
            distributed_array.DistributedArray(new_ids),
        )

    def add_particles(self, dt: DistributedArray, dE: DistributedArray):
        """
        Add a new set of particle coordinates to the beam.

        The particle coordinates given by input `dt` and `dE` will be
        added to the beam object.  The intensity per macroparticle of
        the added particles will be set to match the existing beam and
        all particles will be flagged as active.

        Parameters
        ----------
        dt
            The time coordinates of the new particles.
        dE
            The energy coordinates of the new particles.

        Raises
        ------
        ValueError
            Raised if the local or global sizes of the `dt` and `dE`
            arrays do not match.
        """
        if (dt.local_size != dE.local_size) or (
            dt.global_size != dE.global_size
        ):
            raise ValueError(
                "The dt and dE array sizes are mismatched"
                f"{dt.local_size=}, {dE.local_size=}"
                f"{dt.global_size=}, {dE.global_size=}"
            )

        id_max = np.int32(self._ids.max())
        local_size = self._dt.local_size

        new_ids = dist_help.distributed_arange(local_size, np.int32)
        new_ids.array_local += id_max + 1

        new_flags = dist_help.distributed_zeros(local_size, np.int32)
        new_flags.array_local[:] = np.int32(BeamFlags.ACTIVE.value)

        self._add_coordinates(dt, dE, new_flags, new_ids)

    def _add_coordinates(
        self,
        new_dt: DistributedArray,
        new_dE: DistributedArray,
        new_flags: DistributedArray,
        new_ids: DistributedArray,
    ):
        """
        Protected function to add new coordinates to the beam.

        Parameters
        ----------
        new_dt
            The new dt coordinates.
        new_dE
            The new dE coordinates.
        new_flags
            The new particle flags.
        new_ids
            The new particle ids.
        """
        ratio = self.ratio

        self._dt = distributed_array.concatenate(self._dt, new_dt)
        self._dE = distributed_array.concatenate(self._dE, new_dE)
        self._flags = distributed_array.concatenate(self._flags, new_flags)
        self._ids = distributed_array.concatenate(self._ids, new_ids)

        self.intensity = ratio * self.common_array_size

    def to_dict(self, copy_to_cpu: bool = True) -> dict[str, Any]:
        """
        Convert the beam into a plain dictionary of its state.

        This is the single definition of what a beam consists of. It is what
        `save` writes to file, what the migrations of
        `blond.core.beam.migrations` operate on, and the natural input for
        converters to other codes. A subclass that adds state extends the
        dictionary of ``super().to_dict()`` and reads it back in `from_dict`.

        Parameters
        ----------
        copy_to_cpu
            Whether to copy the particle arrays to host memory. Keep the
            default unless the dictionary stays on the same machine and
            backend, e.g. `save` relies on it to write files that are
            readable without a GPU.

        Returns
        -------
        state
            The beam state, identifying the beam class in ``__class__`` and
            the layout in ``schema_version``.

        Raises
        ------
        ValueError
            If the beam has no particle arrays yet, i.e. ``setup_beam`` was
            never called.
        NotImplementedError
            If the beam is distributed over several MPI ranks.

        See Also
        --------
        from_dict : Rebuild a beam from such a dictionary.
        """
        # Imported here to avoid a cyclic import at module load time.
        from blond.core.beam.serialization import BEAM_SCHEMA_VERSION

        if self.is_distributed:
            raise NotImplementedError(
                "Converting a distributed beam to a dictionary is not "
                "supported; gather the beam on a single rank first."
            )
        if not self.is_set_up():
            raise ValueError(
                "The beam is not set up, there is no state to convert. Call "
                "`setup_beam(...)` first."
            )

        arrays = {
            "dt": self.read_partial_dt(),
            "dE": self.read_partial_dE(),
            "flags": self.read_partial_flags(),
            "ids": self.read_partial_ids(),
        }
        if copy_to_cpu:
            arrays = {
                name: no_cupy_import.copy_to_cpu(array)
                for name, array in arrays.items()
            }

        total_energy = self.reference._total_energy
        return {
            "__class__": type(self).__name__,
            "schema_version": BEAM_SCHEMA_VERSION,
            "intensity": int(self.intensity),
            "is_counter_rotating": bool(self._is_counter_rotating),
            "particle_type": self.particle_type.to_dict(),
            "reference": {
                "time": float(self.reference.time),
                "total_energy": (
                    None if total_energy is None else float(total_energy)
                ),
            },
            "particles": arrays,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> BeamBaseClass:
        """
        Rebuild a beam from the dictionary of `to_dict`.

        The beam class named in ``__class__`` decides what is built, so a
        `ProbeBeam` is restored as a `ProbeBeam`. The particle coordinates
        are placed on the active backend.

        Parameters
        ----------
        state
            Beam state at the current schema version. Older states are
            brought there by `blond.core.beam.migrations.migrate_raw_beam`.

        Returns
        -------
        beam
            The restored beam.

        Raises
        ------
        ValueError
            If ``__class__`` names a beam class unknown to this BLonD
            version.

        See Also
        --------
        to_dict : Convert a beam into such a dictionary.
        """
        # Imported here to avoid a cyclic import at module load time.
        from blond.core.beam import beams
        from blond.core.beam.particle_types import ParticleType

        beam_classes = {
            beam_class.__name__: beam_class
            for beam_class in (beams.Beam, beams.ProbeBeam, beams.EmptyBeam)
        }
        class_name = str(state["__class__"])
        if class_name not in beam_classes:
            raise ValueError(
                f"The beam state is of class {class_name!r}, which this "
                f"BLonD version does not know. Known classes: "
                f"{sorted(beam_classes)}."
            )
        beam_class = beam_classes[class_name]

        particle_type = ParticleType.from_dict(state["particle_type"])

        # The subclasses of `Beam` only differ in how their `__init__` builds
        # the particle coordinates; here those are given, so only the common
        # base state is initialized.
        beam = beam_class.__new__(beam_class)
        BeamBaseClass.__init__(
            beam,
            intensity=int(state["intensity"]),
            particle_type=particle_type,
            is_counter_rotating=bool(state["is_counter_rotating"]),
            is_distributed=False,
        )

        particles = state["particles"]
        beam.setup_beam(
            dt=backend.array(particles["dt"], dtype=backend.float),
            dE=backend.array(particles["dE"], dtype=backend.float),
            flags=backend.array(particles["flags"], dtype=np.int32),
            ids=backend.array(particles["ids"], dtype=np.int32),
            reference_time=float(state["reference"]["time"]),
            reference_total_energy=(
                None
                if state["reference"]["total_energy"] is None
                else float(state["reference"]["total_energy"])
            ),
        )
        return beam

    def save(self, path: str | PathLike) -> None:
        """
        Write the beam to an HDF5 file.

        The file format is versioned; beams written by older BLonD versions
        are migrated when they are loaded again.

        Parameters
        ----------
        path
            Destination file path. An existing file is overwritten.

        See Also
        --------
        load : Read a beam back from an HDF5 file.
        blond.core.beam.serialization.save_beam : Implementation.

        Examples
        --------
        >>> beam.save("beam.h5")
        """
        # Imported here to avoid a cyclic import at module load time.
        from blond.core.beam.serialization import save_beam

        save_beam(self, path)

    @staticmethod
    def load(path: str | PathLike) -> BeamBaseClass:
        """
        Read a beam from an HDF5 file written by `save`.

        Particle coordinates are placed on the active backend, so a beam
        saved on a CPU can be loaded on a GPU and vice versa.

        Parameters
        ----------
        path
            Source file path.

        Returns
        -------
        beam
            The restored beam.

        See Also
        --------
        save : Write a beam to an HDF5 file.
        blond.core.beam.serialization.load_beam : Implementation.

        Examples
        --------
        >>> beam = Beam.load("beam.h5")
        """
        # Imported here to avoid a cyclic import at module load time.
        from blond.core.beam.serialization import load_beam

        return load_beam(path)

    def signed_charge_with_direction(self):
        """
        Return the charge, corrected with the direction of the beam.

        If a particle traverses an electrical field, the directionality is taken into account through the vector of the
        electrical field. For particles traveling in the opposite direction, this has to be inverted, which is handled
        through an opposite sign of the charge.

        Field     -->
        Velocity  -->
        Acceleration

        Field     -->
        Velocity  <--
        Same field, but deceleration for counter-rotating beam.

        Its the most convenient way to include the change via the charge.

        Returns
        -------
        signed_charge_with_direction
            Charge, corrected with the direction of the beam.
        """
        return (
            self.particle_type.charge * -1
            if self.is_counter_rotating
            else self.particle_type.charge
        )

    @property
    def dE(self) -> DistributedArray:
        """
        Beam macro-particle energy coordinates, in [eV].

        Returns
        -------
        dE
            Beam macro-particle energy coordinates, in [eV].
        """
        if self._dE is None:
            raise AttributeError(
                "Beam is not properly initialized. "
                "You can use `setup_beam` or the beam preparation methods.."
            )
        return self._dE

    @property
    def dt(self) -> DistributedArray:
        """
        Beam macro-particle time coordinates, in [s].

        Returns
        -------
        dt
            Beam macro-particle time coordinates, in [s].
        """
        if self._dt is None:
            raise AttributeError(
                "Beam is not properly initialized. "
                "You can use `setup_beam` or the beam preparation methods.."
            )
        return self._dt

    @property
    def flags(self) -> DistributedArray:
        """
        Beam macro-particle flags.

        Returns
        -------
        flags
            Beam macro-particle flags.

        See Also
        --------
        blond.core.beam.flags.BeamFlags: The available flags.
        """
        if self._flags is None:
            raise AttributeError(
                "Beam is not properly initialized. "
                "You can use `setup_beam` or the beam preparation methods.."
            )
        return self._flags

    @property
    def ids(self) -> DistributedArray:
        """
        The macro-particle ids. Each particle keeps its id, even after losses.

        Returns
        -------
        ids
            The macro-particle ids.
        """
        if self._ids is None:
            raise AttributeError(
                "Beam is not properly initialized. "
                "You can use `setup_beam` or the beam preparation methods.."
            )
        return self._ids

    @requires(["MagneticCycleBase"])
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
            Configure-run parameters collected by the MRO chain.
        """
        super().on_run_simulation(
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
            total_energy_init=simulation.magnetic_cycle.get_total_energy_init(
                particle_type=self.particle_type,
            ),
            **kwargs,
        )

    def configure_run(
        self,
        *,
        beam: BeamBaseClass,
        n_turns: int,
        total_energy_init: float,
        **kwargs,
    ) -> None:
        """
        Validate beam arrays and set the reference total energy.

        Parameters
        ----------
        beam
            Simulation `Beam` object.
        n_turns
            Number of turns to simulate.
        total_energy_init
            Initial total energy in [eV] from the magnetic cycle.
        **kwargs
            Passed to the next level in the MRO chain.
        """
        super().configure_run(beam=beam, n_turns=n_turns, **kwargs)
        msg = (
            "Beam was not initialized. This is possible using"
            " `simulation.prepare_beam(...)` or"
            " `beam.setup_beam(...)`."
        )
        assert self._dt is not None, msg
        assert self._dE is not None, msg
        assert self._flags is not None, msg
        assert self._ids is not None, msg

        # Display a warning when the reference energy is overwritten,
        # but not when None is overwritten.
        if (
            self.reference._total_energy != total_energy_init
            and self.reference._total_energy is not None
        ):
            msg = (
                f"`Bunch` was prepared for"
                f" total_energy = {self.reference._total_energy} eV,"
                f" but "
                f" {total_energy_init=} eV."
                f" The energy is overwritten according to simulation."
            )
            warnings.warn(msg, stacklevel=1)
        self.reference.total_energy = total_energy_init

    @property
    @abstractmethod  # pragma: no cover
    def ratio(self) -> float:
        """Ratio of the intensity vs. the sum of weights."""
        pass

    @property
    def particle_type(self) -> ParticleType:
        """
        Type of particles, e.g. protons.

        Returns
        -------
        particle_type
            Type of particles, e.g. protons.
        """
        return self.reference._particle_type

    @abstractmethod  # pragma: no cover
    def setup_beam(
        self,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: NumpyArray | CupyArray = None,
        reference_time: float | None = None,
        reference_total_energy: float | None = None,
        mpi_mode: Literal["root-distributes", "all-ranks"] = "all-ranks",
        **kwargs,
    ) -> None:
        """
        Set beam array attributes for simulation.

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s].
        dE
            Macro-particle energy coordinates, in [eV].
        flags
            Macro-particle flags.
        reference_time
            Time of the reference frame (global time), in [s].
        reference_total_energy
            Time of the reference frame (global total energy), in [eV].
        mpi_mode
            Specifies how the particle data is distributed across multiple ranks (processing
            units) in a parallel environment:

            - "root-distributes": The root node (rank 0) holds the full array and splits it
              into smaller chunks, which are then distributed to all ranks, including rank 0.
              Each rank stores its own chunk of the data. This mode is useful when loading
              large datasets (e.g., with `np.loadtxt(...)`) and distributing parts of the data
              across ranks.

            - "all-ranks": Each rank independently generates and stores a full copy of the data.
              While this mode uses more memory, it can be simpler to implement in scenarios where
              each rank needs to work with its own independent data (e.g., generating separate
              random distributions with `np.random.randn()`).
        **kwargs
            Keyword arguments to make the non-abstract implementation
            extendable.
        """
        pass

    @property  # as readonly attributes
    def is_distributed(self) -> bool:
        """
        Developer option to allow distributed computing.

        Returns
        -------
        is_distributed
            Developer option to allow distributed computing.
        """
        return self._is_distributed

    @property  # as readonly attributes
    def is_counter_rotating(self) -> bool:
        """
        If this is a normal or counter-rotating beam.

        Returns
        -------
        is_counter_rotating
            If this is a normal or counter-rotating beam.
        """
        return self._is_counter_rotating

    @abstractmethod  # pragma: no cover
    def plot_hist2d(self) -> None:
        """Plot 2D histogram of beam coordinates."""
        pass

    @property
    @abstractmethod  # pragma: no cover  # as readonly attributes
    def dt_min(self) -> float:
        """Minimum dt coordinate, in [s]."""
        pass

    @property
    @abstractmethod  # pragma: no cover  # as readonly attributes
    def dt_max(self) -> float:
        """Maximum dt coordinate, in [s]."""
        pass

    @property
    @abstractmethod  # pragma: no cover  # as readonly attributes
    def dE_min(self) -> float:
        """Minimum dE coordinate, in [eV]."""
        pass

    @property
    @abstractmethod  # pragma: no cover  # as readonly attributes
    def dE_max(self) -> float:
        """Maximum dE coordinate, in [eV]."""
        pass

    @property
    @abstractmethod  # pragma: no cover  # as readonly attributes
    def common_array_size(self) -> int:
        """Size of the beam, considering distributed beams."""
        pass

    @property
    @abstractmethod  # pragma: no cover  # as readonly attributes
    def rms_emittance(self):
        """
        Calculate the Root-Mean-Square emittance of the beam.

        Returns
        -------
        rms_emittance
            The Root-Mean-Square emittance in [s eV] of the beam.
        """
        pass

    def n_macroparticles_partial(self) -> int:
        """
        Return size of the beam, ignoring that beam might be distributed.

        Returns
        -------
        n_macroparticles_partial
            Size of the beam, ignoring that beam might be distributed.

        Notes
        -----
        Depends on `is_distributed`
        If not distributed, returns all particles.
        Using `_dt` and `_dE` will result in the same behaviour.

        If distributed, returns only the particles
        visible to the current node.
        """
        if self._dE is not None:
            return self._dE.local_size
        else:
            raise AttributeError(
                f"{self._dE=}. You can use `setup_beam("
                f"...)` for initialisation."
            )

    def read_partial_ids(self) -> NumpyArray | CupyArray:
        """
        Return id-array on current node (distributed computing ready).

        Returns
        -------
        ids
            Id-array on current node (distributed computing ready).

        Notes
        -----
        Depends on `is_distributed`
        If not distributed, returns all particles.
        Using `_dt` and `_dE` will result in the same behaviour

        If distributed, returns only the particles
        visible to the current node.
        """
        return self._ids.array_local

    def read_partial_dt(self) -> NumpyArray | CupyArray:
        """
        Return dt-array on current node (distributed computing ready), in [s].

        Returns
        -------
        dt
            Dt-array on current node (distributed computing ready), in [s].

        Notes
        -----
        Depends on `is_distributed`
        If not distributed, returns all particles.
        Using `_dt` and `_dE` will result in the same behaviour

        If distributed, returns only the particles
        visible to the current node.
        """
        return self._dt.array_local

    def write_partial_dt(self) -> NumpyArray | CupyArray:
        """
        Return dt-array on current node (distributed computing ready), in [s].

        Returns
        -------
        dt
            Dt-array on current node (distributed computing ready), in [s].

        Notes
        -----
        Depends on `is_distributed`
        If not distributed, returns all particles.
        Using `_dt` and `_dE` will result in the same behaviour.

        If distributed, returns only the particles
        visible to the current node.
        """
        return self._dt.array_local

    def read_partial_dE(self) -> NumpyArray | CupyArray:
        """
        Return dE-array on current node (distributed computing ready), in [eV].

        Returns
        -------
        dE
            DE-array on current node (distributed computing ready), in [eV].

        Notes
        -----
        Depends on `is_distributed`
        If not distributed, returns all particles.
        Using `_dt` and `_dE` will result in the same behaviour.

        If distributed, returns only the particles
        visible to the current node.
        """
        return self._dE.array_local

    def write_partial_dE(self) -> NumpyArray | CupyArray:
        """
        Return dE-array on current node (distributed computing ready), in [eV].

        Returns
        -------
        dE
            DE-array on current node (distributed computing ready), in [eV].

        Notes
        -----
        Depends on `is_distributed`
        If not distributed, returns all particles.
        Using `_dt` and `_dE` will result in the same behaviour.

        If distributed, returns only the particles
        visible to the current node.
        """
        return self._dE.array_local

    def write_partial_flags(self) -> NumpyArray | CupyArray:
        """
        Return flags-array on current node (distributed computing ready).

        Returns
        -------
        flags
            Flags-array on current node (distributed computing ready).

        Notes
        -----
        Depends on `is_distributed`
        If not distributed, returns all particles.
        Using `_dt` and `_dE` will result in the same behaviour.

        If distributed, returns only the particles
        visible to the current node.
        """
        return self._flags.array_local

    def read_partial_flags(self) -> NumpyArray | CupyArray:
        """
        Return flags-array on current node (distributed computing ready).

        Returns
        -------
        flags
            Flags-array on current node (distributed computing ready).

        Notes
        -----
        Depends on `is_distributed`
        If not distributed, returns all particles.
        Using `_dt` and `_dE` will result in the same behaviour.

        If distributed, returns only the particles
        visible to the current node.
        """
        return self._flags.array_local

    def sort_by_dt(self) -> None:
        """
        Sort the macro-particles in place by ascending ``dt``.

        All per-particle arrays (``dt``, ``dE``, ``ids``, ``flags``) are
        permuted by the same order, so particle identity is preserved.

        Raises
        ------
        NotImplementedError
            If the beam is distributed across MPI ranks: a per-node sort
            cannot order the global beam, so sorting is unsupported there.
        """
        if self.is_distributed:
            raise NotImplementedError(
                "`sort_by_dt` cannot sort an MPI-distributed beam: a "
                "per-node sort does not order the global beam."
            )

        order = (
            self._dt.array_local.argsort()
        )  # ndarray method works for NumPy and CuPy

        self._dt.array_local[:] = self._dt.array_local[order]
        self._dE.array_local[:] = self._dE.array_local[order]
        self._ids.array_local[:] = self._ids.array_local[order]
        self._flags.array_local[:] = self._flags.array_local[order]

    def purge_flagged_entries(self, flag: int = BeamFlags.LOST.value) -> None:
        """
        Delete flagged array entries from the array.

        Parameters
        ----------
        flag
            The flag to be used as a selector what to place at the end.
            Default is to remove lost particles ``flag=0``.
        """
        from blond.core.backends.backend import (
            backend,  # prevent cyclic import
        )
        from blond.generals.distributed.helpers import mpi_barrier

        n_before_truncation_global = self._dt.global_size

        n_after_truncation_local = (
            backend.specials.move_flagged_elements_to_end(
                flag=flag,
                flags=self._flags.array_local,
                dt=self._dt.array_local,
                dE=self._dE.array_local,
                ids=self._ids.array_local,
            )
        )
        self._flags.array_local = self._flags.array_local[
            :n_after_truncation_local
        ]
        self._dt.array_local = self._dt.array_local[:n_after_truncation_local]
        self._dE.array_local = self._dE.array_local[:n_after_truncation_local]
        self._ids.array_local = self._ids.array_local[
            :n_after_truncation_local
        ]

        mpi_barrier()
        n_after_truncation_global = self._dt.global_size

        self.intensity *= (
            n_after_truncation_global / n_before_truncation_global
        )
