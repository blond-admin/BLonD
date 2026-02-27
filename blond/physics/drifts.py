# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Collection of implementations to handle movement in bent synchrotron sections."""

from __future__ import annotations

import abc
from abc import ABC
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np

from blond.core.backends.backend import backend
from blond.core.base import (
    AltersReference,
    BeamPhysicsRelevant,
    HasPropertyCache,
    Schedulable,
)
from blond.core.reference_clock.reference_clock import ReferenceCoordinates

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


def _assert_purely_real_or_imaginary(val: complex | NumpyArray):
    """
    Assert that a complex number is purely real or purely imaginary.

    A complex number is considered *purely real* if its imaginary part is zero,
    and *purely imaginary* if its real part is zero. This function raises an
    `AssertionError` if the number has both nonzero real and imaginary parts.

    Parameters
    ----------
    val : complex
        Complex number to be validated.

    Raises
    ------
    AssertionError
        If `val` has both real and imaginary parts nonzero.

    Examples
    --------
    >>> _assert_purely_real_or_imaginary(5 + 0j)   # purely real
    >>> _assert_purely_real_or_imaginary(0 + 3j)   # purely imaginary
    >>> _assert_purely_real_or_imaginary(0j)       # zero (both parts 0) is fine
    >>> _assert_purely_real_or_imaginary(2 + 4j)
    Traceback (most recent call last):
        ...
    AssertionError: Expected number with only real or only imaginary part, not (2+4j)
    """
    if np.any((val.real != 0) & (val.imag != 0)):
        raise ValueError(
            f"Expected purely real or purely imaginary number, not {val}."
        )


class DriftBaseClass(BeamPhysicsRelevant, AltersReference, ABC):
    """
    Base class of a drift.

    Parameters
    ----------
    orbit_length
        Length of drift, in [m].
        Length / Velocity => Time to pass the element.
    section_index
        Section index to group elements into sections.
    **kwargs
        Additional keyword arguments for MRO of fused elements.
    """

    def __init__(
        self,
        orbit_length: float,
        section_index: int = 0,
        **kwargs: dict[str, Any],  # for MRO of fused elements
    ) -> None:
        super().__init__(
            section_index=section_index,
            **kwargs,  # for MRO of fused elements
        )

        self.orbit_length = orbit_length

    @abc.abstractmethod  # pragma: no cover
    def eta_0(self, gamma: float) -> backend.float:
        """
        Drift in arc parameter eta for one turn in synchrotron.

        Parameters
        ----------
        gamma
            Lorentz gamma factor.

        Returns
        -------
        eta_0
            Drift in arc parameter eta for one turn in synchrotron.
        """
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        """
        super().on_init_simulation(simulation=simulation)

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
        pass


class DriftSimple(DriftBaseClass, Schedulable, HasPropertyCache):
    """
    Base class to implement beam drifts in synchrotrons.

    Parameters
    ----------
    orbit_length
        Length of drift, in [m].
    section_index
        Section index to group elements into sections.
    momentum_compaction_factor
        Momentum compaction factor.
    **kwargs
        Additional keyword arguments for MRO of fused elements.
    """

    def __init__(
        self,
        orbit_length: float,
        section_index: int = 0,
        momentum_compaction_factor: float | None = None,
        **kwargs: dict[str, Any],  # for MRO of fused elements
    ) -> None:
        """
        Base class to implement beam drifts in synchrotrons.

        Parameters
        ----------
        orbit_length
            Length of drift, in [m].
            Length / Velocity => Time to pass the element.
        section_index
            Section index to group elements into sections.
        momentum_compaction_factor
            Momentum compaction factor.
        **kwargs
            Additional keyword arguments for MRO of fused elements.

        Examples
        --------
        Parameters can be scheduled along the simulation execution

        >>> from blond import DriftSimple
        >>> drift = DriftSimple(...)
        >>> drift.schedule(attribute='momentum_compaction_factor', value=np.array(...), mode="per-turn")
        """
        super().__init__(
            orbit_length=orbit_length,
            section_index=section_index,
            intended_for_scheduling=[
                "momentum_compaction_factor",
            ],
            **kwargs,  # for MRO of fused elements
        )

        self._simulation: Simulation | None = None

        self._last_eta_0: float | None = None

        self.momentum_compaction_factor: float | None = (
            momentum_compaction_factor
        )

    @staticmethod
    def headless(
        momentum_compaction_factor: NumpyArray | tuple[NumpyArray, NumpyArray],
        orbit_length: float,
        section_index: int = 0,
    ) -> DriftSimple:
        """
        Initialize object without simulation context.

        Parameters
        ----------
        momentum_compaction_factor
            Momentum compaction factor.
        orbit_length
            Length of drift, in [m].
            Length / Velocity => Time to pass the element.
        section_index
            Section index to group elements into sections.

        Returns
        -------
        drift_simple
            DriftSimple object without simulation context.
        """
        from blond.core.base import DynamicParameter

        d = DriftSimple(
            orbit_length=orbit_length,
            section_index=section_index,
        )
        if isinstance(momentum_compaction_factor, int | float):
            d.momentum_compaction_factor = float(momentum_compaction_factor)
        else:
            d.schedule(
                "momentum_compaction_factor", momentum_compaction_factor
            )
        from blond.core.beam.base import BeamBaseClass
        from blond.core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        simulation.turn_i = Mock(DynamicParameter)
        simulation.turn_i.value = 0
        d.on_init_simulation(simulation=simulation)
        d.on_run_simulation(
            simulation=simulation,
            n_turns=1,
            beam=Mock(BeamBaseClass),
        )
        return d

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        """
        super().on_init_simulation(simulation=simulation)
        self._simulation = simulation
        if (
            self.momentum_compaction_factor is None
        ) and "momentum_compaction_factor" not in self.schedules:
            raise ValueError(
                "You need to define `momentum_compaction_factor` via `.momentum_compaction_factor=...` "
                "or `.schedule(attribute='momentum_compaction_factor', value=...)`"
            )

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        super()._track(beam=beam)

        if self.schedule_active:
            self.apply_schedules(
                turn_i=self._simulation.turn_i.value,
                reference_time=beam.reference.time,
            )

        dt = self.track_reference(beam.reference)
        gamma = beam.reference.gamma
        self._last_eta_0 = self.eta_0(gamma)

        if beam.common_array_size > 0:
            backend.specials.drift_simple(
                dt=beam.write_partial_dt(),
                dE=beam.read_partial_dE(),
                T=dt,
                eta_0=self._last_eta_0,
                beta=beam.reference.beta,
                energy=beam.reference.total_energy,
            )

    def track_reference(
        self, reference: ReferenceCoordinates, **kwargs
    ) -> float:
        """
        Update the coordinates of the reference coordinate system.

        Parameters
        ----------
        reference
            The object that holds the reference time [s] and total energy [eV].
        **kwargs
            Allows more arguments in the method definition outside the
            abstract class.

        Returns
        -------
        reference_time_change
            Change of reference time.
        """
        reference_time_change = self.orbit_length / reference.velocity
        reference.time += reference_time_change
        return reference_time_change

    def eta_0(self, gamma: float) -> float:
        """
        Drift in arc parameter eta for one turn in synchrotron.

        Parameters
        ----------
        gamma
            Lorentz gamma factor.

        Returns
        -------
        eta_0
            Drift in arc parameter eta for one turn in synchrotron.
        """
        return self.alpha_0 - (1 / (gamma * gamma))

    # alias of momentum_compaction_factor
    @property  # as readonly attributes
    def alpha_0(self) -> float:
        """
        Momentum compaction factor.

        Returns
        -------
        alpha_0
            Momentum compaction factor.
        """
        return self.momentum_compaction_factor

    def invalidate_cache(self):
        """Delete the stored values of functions with @cached_property."""
        # super()._invalidate_cache(DriftSimple.cached_props)
        pass
