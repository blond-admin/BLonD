# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Collection of implementations to handle movement in bent synchrotron sections.

Authors
-------
Simon Lauber
"""

from __future__ import annotations

import abc
import cmath
from abc import ABC
from typing import TYPE_CHECKING
from unittest.mock import Mock

from blond.core.backends.backend import backend
from blond.core.base import BeamPhysicsRelevant, HasPropertyCache, Schedulable

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


def _assert_purely_real_or_imaginary(val: complex):
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
    if val.real != 0 and val.imag != 0:
        raise ValueError(
            f"Expected purely real or purely imaginary number, not {val}."
        )


class DriftBaseClass(BeamPhysicsRelevant, Schedulable, ABC):
    """
    Base class of a drift.

    Parameters
    ----------
    orbit_length
        Length of drift, in [m].
        Length / Velocity => Time to pass the element
    section_index
        Section index to group elements into sections

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
        """Drift in arc parameter eta for one turn in synchrotron."""
        pass

    def track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        super().track(beam=beam)

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        simulation
            `Simulation` context manager
        """
        super().on_init_simulation(simulation=simulation)

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

        simulation
            `Simulation` context manager
        beam
            Simulation `Beam` object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        pass


class DriftSimple(DriftBaseClass, HasPropertyCache):
    """
    Base class to implement beam drifts in synchrotrons.

    Parameters
    ----------
    orbit_length
        Length of drift, in [m]
    section_index
        Section index to group elements into sections
    transition_gamma
        Gamma of transition crossing
    """

    def __init__(
        self,
        orbit_length: float,
        section_index: int = 0,
        transition_gamma: complex | float | None = None,
        momentum_compaction_factor: float | None = None,
        **kwargs: dict[str, Any],  # for MRO of fused elements
    ) -> None:
        """
        Base class to implement beam drifts in synchrotrons.

        Parameters
        ----------
        orbit_length
            Length of drift, in [m].
            Length / Velocity => Time to pass the element
        section_index
            Section index to group elements into sections
        transition_gamma
            Gamma of transition crossing


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
            **kwargs,  # for MRO of fused elements
        )

        self._transition_gamma: complex | None = None
        self._momentum_compaction_factor: float | None = None

        self._simulation: Simulation | None = None

        match (momentum_compaction_factor, transition_gamma):
            case (None, None):
                pass
            case (None, _):
                self.transition_gamma = transition_gamma
            case (_, None):
                self.momentum_compaction_factor = momentum_compaction_factor
            case (_, _):
                raise ValueError(
                    "Got `momentum_compaction_factor` and "
                    "`transition_gamma` as argument. "
                    "Please provide only one of them."
                )

    @property  # read only, set by `transition_gamma`
    def momentum_compaction_factor(self) -> float | None:
        """Momentum compaction factor."""
        return self._momentum_compaction_factor

    @momentum_compaction_factor.setter  # read only, set by `transition_gamma`
    def momentum_compaction_factor(
        self, momentum_compaction_factor: float
    ) -> None:
        """Momentum compaction factor."""
        self._momentum_compaction_factor = momentum_compaction_factor
        self._transition_gamma = 1 / cmath.sqrt(momentum_compaction_factor)

    @property
    def transition_gamma(self) -> complex | None:
        """Gamma of transition crossing."""
        return self._transition_gamma

    @transition_gamma.setter
    def transition_gamma(self, transition_gamma: complex) -> None:
        """Gamma of transition crossing."""
        _assert_purely_real_or_imaginary(transition_gamma)

        _momentum_compaction_factor = 1.0 / (
            transition_gamma * transition_gamma
        )

        # .real is only possible, because we asserted that the momentum
        # compaction factor is entirely real or complex.
        self._momentum_compaction_factor = _momentum_compaction_factor.real

        self._transition_gamma = complex(transition_gamma)

    @staticmethod
    def headless(
        transition_gamma: float
        | int
        | NumpyArray
        | tuple[NumpyArray, NumpyArray],
        orbit_length: float,
        section_index: int = 0,
    ) -> DriftSimple:
        """
        Initialize object without simulation context.

        Parameters
        ----------
        transition_gamma
            Gamma of transition crossing
        orbit_length
            Length of drift, in [m].
            Length / Velocity => Time to pass the element
        section_index
            Section index to group elements into sections

        Returns
        -------
        drift_simple
        """
        from blond.core.base import DynamicParameter

        d = DriftSimple(
            orbit_length=orbit_length,
            section_index=section_index,
        )
        if isinstance(transition_gamma, float):
            d.transition_gamma = transition_gamma
        else:
            d.schedule("transition_gamma", transition_gamma, mode="per-turn")
        from blond.core.beam.base import BeamBaseClass
        from blond.core.simulation.simulation import Simulation

        simulation = Mock(Simulation)
        simulation.turn_i = Mock(DynamicParameter)
        simulation.turn_i.value = 0
        d.on_init_simulation(simulation=simulation)
        d.on_run_simulation(
            simulation=simulation,
            turn_i_init=0,
            n_turns=1,
            beam=Mock(BeamBaseClass),
        )
        return d

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        simulation
            `Simulation` context manager
        """
        super().on_init_simulation(simulation=simulation)
        self._simulation = simulation
        if (
            self.transition_gamma is None
        ) and "transition_gamma" not in self.schedules:
            raise ValueError(
                "You need to define `transition_gamma` via `.transition_gamma=...` "
                "or `.schedule(attribute='transition_gamma', value=...)`"
            )

    def track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        super().track(beam=beam)
        if self.schedule_active:
            self.apply_schedules(
                turn_i=self._simulation.turn_i.value,
                reference_time=beam.reference_time,
            )
        dt = self.orbit_length / beam.reference_velocity
        gamma = beam.reference_gamma
        eta_0 = self.alpha_0 - (1 / (gamma * gamma))
        backend.specials.drift_simple(
            dt=beam.write_partial_dt(),
            dE=beam.read_partial_dE(),
            T=dt,
            eta_0=eta_0,
            beta=beam.reference_beta,
            energy=beam.reference_total_energy,
        )
        beam.reference_time += dt

    def eta_0(self, gamma: float) -> float:
        """Drift in arc parameter eta for one turn in synchrotron."""
        return self.alpha_0 - (1 / (gamma * gamma))

    # alias of momentum_compaction_factor
    @property  # as readonly attributes
    def alpha_0(self) -> float:
        """Momentum compaction factor."""
        return self.momentum_compaction_factor

    def invalidate_cache(self):
        """Delete the stored values of functions with @cached_property."""
        # super()._invalidate_cache(DriftSimple.cached_props)
        pass
