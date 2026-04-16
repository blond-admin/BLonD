# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Generalized functions to deal with Beam objects."""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from blond.core.base import Preparable
from blond.core.beam.flags import BeamFlags
from blond.core.helpers import int_from_float_with_warning
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.core.ring.helpers import requires
from blond.generals.cupy.no_cupy_import import copy_to_cpu

# Schema version for the on-disk ``.npz`` beam format.
# Bump on any breaking change to ``save`` / ``load``.
_BEAM_SCHEMA_VERSION = 1

if TYPE_CHECKING:  # pragma: no cover
    from os import PathLike
    from typing import Any, Literal

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

    def save(self, path: str | PathLike) -> None:
        """
        Serialize the beam to disk in a portable, version-checked format.

        The file is a NumPy ``.npz`` archive (a zip of ``.npy`` files) containing:

        - ``dt``, ``dE``, ``flags``, ``ids``: the particle coordinate arrays
        - ``metadata``: a 0-D array holding a JSON string with the schema
          version, intensity, particle-type physical constants, reference
          coordinates and counter-rotation flag.

        Unlike pickle, this format is explicit and robust to Python / BLonD
        version changes: ``load`` checks the schema version and fails loudly
        when the on-disk format is incompatible, rather than silently
        reconstructing a broken object.

        Parameters
        ----------
        path
            Destination file path. A ``.npz`` suffix is appended automatically
            by ``numpy.savez`` if not already present.

        See Also
        --------
        load : Deserialize a beam from disk.

        Notes
        -----
        - Only single-rank (non-distributed) beams are supported; distributed
          beams raise an ``AssertionError``.
        - CuPy (GPU) arrays are moved to CPU memory before writing so the file
          is portable to non-GPU hosts. They are re-promoted to the active
          backend on load.
        """
        assert self._dt is not None, "Beam is not initialized."
        assert self._dE is not None, "Beam is not initialized."
        assert self._flags is not None, "Beam is not initialized."
        assert self._ids is not None, "Beam is not initialized."
        assert not self._is_distributed, (
            "Saving a distributed beam is not supported."
        )

        metadata = {
            "schema_version": _BEAM_SCHEMA_VERSION,
            "class": type(self).__name__,
            "intensity": float(self.intensity),
            "is_counter_rotating": bool(self._is_counter_rotating),
            "particle_type": {
                "mass": float(self.particle_type.mass),
                "charge": float(self.particle_type.charge),
                "user_decay_rate": float(self.particle_type.user_decay_rate),
            },
            "reference": {
                "time": float(self.reference.time),
                "total_energy": (
                    None
                    if self.reference._total_energy is None
                    else float(self.reference._total_energy)
                ),
            },
        }

        np.savez(
            path,
            metadata=np.array(json.dumps(metadata)),
            dt=copy_to_cpu(self._dt.array_local),
            dE=copy_to_cpu(self._dE.array_local),
            flags=copy_to_cpu(self._flags.array_local),
            ids=copy_to_cpu(self._ids.array_local),
        )

    @staticmethod
    def load(path: str | PathLike) -> BeamBaseClass:
        """
        Deserialize a beam that was written with ``save``.

        The schema version embedded in the file is checked against the
        in-memory ``_BEAM_SCHEMA_VERSION`` constant; a ``ValueError`` is
        raised if they do not match.

        Parameters
        ----------
        path
            Source file path.

        Returns
        -------
        beam
            The restored beam. Particle coordinates are promoted to the
            currently active backend.

        Raises
        ------
        ValueError
            If the file's schema version is not supported.

        See Also
        --------
        save : Serialize the beam state to disk.
        """
        # Local imports to avoid cyclic imports at module load time.
        from blond.core.beam.beams import Beam
        from blond.core.beam.particle_types import ParticleType

        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            dt = archive["dt"]
            dE = archive["dE"]
            flags = archive["flags"]
            ids = archive["ids"]

        schema_version = metadata.get("schema_version")
        if schema_version != _BEAM_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported beam schema version {schema_version!r}; "
                f"this BLonD installation expects version "
                f"{_BEAM_SCHEMA_VERSION}."
            )

        particle_type = ParticleType(
            mass=metadata["particle_type"]["mass"],
            charge=metadata["particle_type"]["charge"],
            user_decay_rate=metadata["particle_type"]["user_decay_rate"],
        )

        beam = Beam(
            intensity=metadata["intensity"],
            particle_type=particle_type,
            is_counter_rotating=metadata["is_counter_rotating"],
        )
        beam.setup_beam(
            dt=dt,
            dE=dE,
            flags=flags,
            reference_time=metadata["reference"]["time"],
            reference_total_energy=metadata["reference"]["total_energy"],
        )
        # ``setup_beam`` assigns fresh ids via ``arange``; overwrite with the
        # stored ids so particle identification is preserved across saves.
        from blond.core.backends.backend import backend

        beam._ids.array_local = backend.array(ids, dtype=np.int32)
        return beam

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
                "You can sse `setup_beam` or the beam preparation methods.."
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
                "You can sse `setup_beam` or the beam preparation methods.."
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
        blond.core.beam.base.BeamFlags: The available flags.
        """
        if self._flags is None:
            raise AttributeError(
                "Beam is not properly initialized. "
                "You can sse `setup_beam` or the beam preparation methods.."
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
                "You can sse `setup_beam` or the beam preparation methods.."
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
            Additional keyword arguments.
        """
        super().on_run_simulation(
            beam=beam,
            simulation=simulation,
            n_turns=n_turns,
        )
        msg = (
            "Beam was not initialized. This is possible using"
            " `simulation.prepare_beam(...)` or"
            " `beam.setup_beam(...)`."
        )
        assert self._dt is not None, msg
        assert self._dE is not None, msg
        assert self._flags is not None, msg
        assert self._ids is not None, msg
        total_energy_init = simulation.magnetic_cycle.get_total_energy_init(
            particle_type=self.particle_type,
        )
        if (
            self.reference._total_energy != total_energy_init
            and self.reference._total_energy is not None
        ):
            # Display a warning when the reference energy is overwritten,
            # but not when None is overwritten.
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

    @requires(["MagneticCycleBase"])
    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        """
        pass  # this gets never called

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
