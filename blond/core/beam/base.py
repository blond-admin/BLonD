# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Generalized functions to deal with Beam objects."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.constants import speed_of_light as c0  # type: ignore

from blond.core.backends.backend import backend
from blond.core.base import HasPropertyCache, Preparable
from blond.core.beam.flags import BeamFlags
from blond.core.helpers import int_from_float_with_warning
from blond.core.ring.helpers import requires

if TYPE_CHECKING:  # pragma: no cover
    from typing import Self

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import ArrayLike
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.particle_types import ParticleType
    from blond.core.simulation.simulation import Simulation


class BeamBaseClass(Preparable, HasPropertyCache, ABC):
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
        self._particle_type = particle_type
        self._is_counter_rotating = is_counter_rotating

        # should be initialized later using `setup_beam`
        self._dE: NumpyArray | CupyArray | None = None
        self._dt: NumpyArray | CupyArray | None = None
        self._flags: NumpyArray | CupyArray | None = None
        self._ids: NumpyArray | CupyArray | None = None

        self.reference_time: float = 0.0
        # todo cached properties
        self._reference_total_energy: float | None = (
            None  # todo cached  properties
        )

    def __iadd__(self, other: Self | ArrayLike) -> Self:
        """
        Inplace addition function.

        Inplace addition to the beam object.  Either adds two instances
        of BeamBaseClass together or adds an 2D array of particle
        coordinates (format: `[dt, dE]`) to the beam.

        Parameters
        ----------
            other
                Either an instance of BeamBaseClass or an ArrayLike of
                `[dt, dE]`.

        Examples
        --------
            >>> main_beam = Beam(...)
            >>> injection_2 = Beam(...)
            >>> # Track to 2nd injection
            >>> meain_beam += injection_2
            >>> # Continue tracking


            >>> main_beam = Beam(...)
            >>> for i in range(1, n_injections):
            >>>     # Track one turn
            >>>     main_beam += np.load(f"injected_turn_{i}.npy")
        """
        if isinstance(other, type(self)):
            self.add_beam(other)
        else:
            self.add_particles(other)

        return self

    def add_beam(self, other: Self):
        """
        Add the contents of another beam to this one.

        Function to add the particles from another beam to this beam.
        Requires that both beams specify the same ParticleType and have
        the same intensity per macroparticle.  The coordinates and
        flags of the new beam will be concatenated with those of the
        existing beam.  The ids of the new beam will be replaced, with
        the values incremeted from the number of macroparticles in the
        original beam.

        Parameters
        ----------
            other
                The BeamBaseClass object to add to this Beam
            purge (optional):
                Specify if flagged particles should be purged when
                adding the beams.  If True, all particles with flag
                different to the Active value will be purged.
                Defaults to False.

        Raises
        ------
            ValueError: If self.ratio != other.ratio, a ValueError is
                        raised.
            TypeError: If self.particle_type != other.particle_type
        """
        if self.ratio != other.ratio:
            raise ValueError("Cannot add beams with a different ratio")

        if self.particle_type != other.particle_type:
            raise TypeError("Cannot add beams with different particle_type")

        self._append_to_self(
            other._dt, other._dE, other._flags, other.intensity
        )

    def add_particles(self, new_particles: ArrayLike):
        """
        Add new particles to the beam.

        Function to add new particles to the beam from an array.  The
        input array must be 2D with format `[dt, dE]`.  The new
        particles will be assumed to have the same intensity per
        macroparticle as the original beam.

        Parameters
        ----------
        new_particles
            2D array of `[dt, dE]` defining the coordinates
                           of the new particles.

        Raises
        ------
            ValueError: If the length of the new particle arrays do not
                        match, a ValueError is raised.
        """
        dt = backend.array(new_particles[0])
        dE = backend.array(new_particles[1])

        if len(dt) != len(dE):
            raise ValueError(
                "The input particles must have equal numbers of "
                "`dt` and `dE` coordinates"
            )

        self._append_to_self(dt, dE)

    def _append_to_self(
        self,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: NumpyArray | CupyArray | None = None,
        intensity: float | None = None,
    ):
        if intensity is None:
            intensity = self.ratio * len(dt)

        if flags is None:
            flags = (
                np.ones(len(dt), dtype=self._flags.dtype)
                * BeamFlags.ACTIVE.value
            )

        id_max = backend.max(self._ids)
        ids = backend.arange(
            id_max + 1, id_max + len(dt) + 1, dtype=self._ids.dtype
        )

        if self._dt.dtype != dt.dtype:
            raise TypeError(
                "New time coordinates do not have the correct "
                "`dtype`.  Should be {self._dt.dtype} but is "
                "{dt.dtype}"
            )

        if self._dE.dtype != dE.dtype:
            raise TypeError(
                "New energy coordinates do not have the correct "
                "`dtype`.  Should be {self._dE.dtype} but is "
                "{dE.dtype}"
            )

        if self._flags.dtype != flags.dtype:
            raise TypeError(
                "New flags do not have the correct "
                "`dtype`.  Should be {self._flags.dtype} but is "
                "{flags.dtype}"
            )

        self._dt = backend.concatenate((self._dt, dt))
        self._dE = backend.concatenate((self._dE, dE))
        self._ids = backend.concatenate((self._ids, ids))
        self._flags = backend.concatenate((self._flags, flags))
        self.intensity += intensity

    @requires(["EnergyCycleBase"])
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

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation `Beam` object.
        n_turns
            Number of turns to simulate.
        turn_i_init
            Initial turn to execute simulation.
        **kwargs
            Additional keyword arguments.
        """
        super().on_run_simulation(
            beam=beam,
            simulation=simulation,
            n_turns=n_turns,
            turn_i_init=turn_i_init,
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
        new_reference_total_energy = (
            simulation.magnetic_cycle.get_total_energy_init(
                turn_i_init=turn_i_init,
                t_init=self.reference_time,
                particle_type=self.particle_type,
            )
        )
        if self._reference_total_energy != new_reference_total_energy:
            msg = (
                f"`Bunch` was prepared for"
                f" total_energy = {self._reference_total_energy} eV,"
                f" but simulation at {turn_i_init=} is"
                f" {new_reference_total_energy} eV."
                f" The energy is overwritten according to simulation."
            )
            warnings.warn(msg, stacklevel=1)
        self.reference_total_energy = new_reference_total_energy

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
        return self._particle_type

    @property
    def reference_total_energy(self) -> float:
        """
        Total beam energy [eV].

        Returns
        -------
        reference_total_energy
            Total beam energy [eV].
        """
        if self._reference_total_energy is None:
            raise ValueError(
                "Beam is not properly set up, please set "
                "`reference_total_energy` first!"
            )
        return self._reference_total_energy

    @reference_total_energy.setter
    def reference_total_energy(self, reference_total_energy: float) -> None:
        """
        Total beam energy [eV].

        Parameters
        ----------
        reference_total_energy
            Total beam energy [eV].
        """
        self.invalidate_cache_reference()
        self._reference_total_energy = reference_total_energy

    @cached_property
    def reference_gamma(self) -> float:
        """
        Beam reference gamma a.k.a. Lorentz factor [].

        Returns
        -------
        reference_gamma
            Beam reference gamma a.k.a. Lorentz factor [].
        """
        # reference_total_energy in eV and mass_inv in [c²/eV]
        if self._reference_total_energy is None:
            raise ValueError(
                "Beam is not properly set up, please set "
                "`reference_total_energy` first!"
            )
        val = self._reference_total_energy * self._particle_type.mass_inv
        return val

    @cached_property
    def reference_beta(self) -> float:
        """
        Beam reference fraction of speed of light (v/c0) [].

        Returns
        -------
        reference_beta
            Beam reference fraction of speed of light (v/c0) [].
        """
        gamma = self.reference_gamma
        val = np.sqrt(1.0 - 1.0 / (gamma * gamma))
        return val

    @cached_property
    def reference_velocity(self) -> float:
        """
        Beam reference speed [m/s].

        Returns
        -------
        reference_velocity
            Beam reference speed [m/s].
        """
        return self.reference_beta * c0

    @abstractmethod  # pragma: no cover
    def setup_beam(
        self,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: NumpyArray | CupyArray = None,
        reference_time: float | None = None,
        reference_total_energy: float | None = None,
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

    @requires(["EnergyCycleBase"])
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

    @cached_property
    @abstractmethod  # pragma: no cover  # as readonly attributes
    def dt_min(self) -> float:
        """Minimum dt coordinate, in [s]."""
        pass

    @cached_property
    @abstractmethod  # pragma: no cover  # as readonly attributes
    def dt_max(self) -> float:
        """Maximum dt coordinate, in [s]."""
        pass

    @cached_property
    @abstractmethod  # pragma: no cover  # as readonly attributes
    def dE_min(self) -> float:
        """Minimum dE coordinate, in [eV]."""
        pass

    @cached_property
    @abstractmethod  # pragma: no cover  # as readonly attributes
    def dE_max(self) -> float:
        """Maximum dE coordinate, in [eV]."""
        pass

    @cached_property
    @abstractmethod  # pragma: no cover  # as readonly attributes
    def common_array_size(self) -> int:
        """Size of the beam, considering distributed beams."""
        pass

    cached_props = (
        "dE_min",
        "dE_max",
        "dt_min",
        "dt_max",
        "common_array_size",
        "ratio",
        "reference_gamma",
        "reference_beta",
        "reference_velocity",
    )

    def invalidate_cache_reference(self) -> None:
        """Reset cache of `cached_property` attributes."""
        super()._invalidate_cache(
            (
                "reference_gamma",
                "reference_beta",
                "reference_velocity",
            )
        )

    def invalidate_cache_dE(self) -> None:
        """Reset cache of `cached_property` attributes."""
        super()._invalidate_cache(
            (
                "dE_min",
                "dE_max",
            )
        )

    def invalidate_cache_dt(self) -> None:
        """Reset cache of `cached_property` attributes."""
        super()._invalidate_cache(
            (
                "dt_min",
                "dt_max",
            )
        )

    def invalidate_cache(self) -> None:
        """Delete the stored values of functions with @cached_property."""
        self._invalidate_cache(BeamBaseClass.cached_props)

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
            return len(self._dE)
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
        return self._ids

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
        return self._dt

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
        self.invalidate_cache_dt()
        return self._dt

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
        return self._dE

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
        self.invalidate_cache_dE()
        return self._dE

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
        self.invalidate_cache_dt()
        self.invalidate_cache_dE()
        return self._flags

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
        return self._flags

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

        n_before_truncation = len(self._flags)
        n_after_truncation = backend.specials.move_flagged_elements_to_end(
            flag=flag,
            flags=self._flags,
            dt=self._dt,
            dE=self._dE,
            ids=self._ids,
        )
        self._flags = self._flags[:n_after_truncation]
        self._dt = self._dt[:n_after_truncation]
        self._dE = self._dE[:n_after_truncation]
        self._ids = self._ids[:n_after_truncation]

        self.intensity *= n_after_truncation / n_before_truncation
