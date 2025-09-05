from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np

from .._core.backends.backend import backend
from .._core.base import BeamPhysicsRelevant, HasPropertyCache, Schedulable

if TYPE_CHECKING:  # pragma: no cover
    from typing import Iterable
    from typing import Optional
    from typing import Optional as LateInit
    from typing import Tuple, Union

    from numpy.typing import NDArray as NumpyArray

    from .._core.beam.base import BeamBaseClass
    from .._core.simulation.simulation import Simulation


class DriftBaseClass(BeamPhysicsRelevant, Schedulable, ABC):
    def __init__(
        self,
        orbit_length: float,
        section_index: int = 0,
        **kwargs,  # for MRO of fused elements
    ):
        """
        Base class of a drift

        Parameters
        ----------
        orbit_length
            Length of drift, in [m].
            Length / Velocity => Time to pass the element
        section_index
            Section index to group elements into sections

        """
        super().__init__(
            section_index=section_index,
            **kwargs,  # for MRO of fused elements
        )

        self.orbit_length = orbit_length

    def __str__(self) -> str:
        return f""

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        super().track(beam=beam)

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        super().on_init_simulation(simulation=simulation)

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        turn_i_init: int,
        **kwargs,
    ) -> None:
        """
        Lateinit method when `simulation.run_simulation` is called

        simulation
            Simulation context manager
        beam
            Simulation beam object
        n_turns
            Number of turns to simulate
        turn_i_init
            Initial turn to execute simulation
        """
        pass


class DriftSimple(DriftBaseClass, HasPropertyCache):
    """
    Base class to implement beam drifts in synchrotrons

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
        transition_gamma: Optional[float] = None,
        **kwargs,  # for MRO of fused elements
    ):
        """
        Base class to implement beam drifts in synchrotrons

        Parameters
        ----------
        orbit_length
            Length of drift, in [m].
            Length / Velocity => Time to pass the element
        section_index
            Section index to group elements into sections
        transition_gamma
            Gamma of transition crossing

        """

        super().__init__(
            orbit_length=orbit_length,
            section_index=section_index,
            **kwargs,  # for MRO of fused elements
        )

        self._transition_gamma: float | None = None
        self._momentum_compaction_factor: float | None = None

        self._simulation: LateInit[Simulation] = None

        if transition_gamma is not None:
            self.transition_gamma = transition_gamma  # use setter method

    @property  # read only, set by `transition_gamma`
    def momentum_compaction_factor(self) -> np.float64 | np.float32:
        """Momentum compaction factor"""
        return self._momentum_compaction_factor

    @property
    def transition_gamma(self):
        """Gamma of transition crossing"""
        return self._transition_gamma

    @transition_gamma.setter
    def transition_gamma(self, transition_gamma):
        """Gamma of transition crossing"""
        transition_gamma = backend.float(transition_gamma)
        self._momentum_compaction_factor = 1 / (
            transition_gamma * transition_gamma
        )
        self._transition_gamma = transition_gamma

    @staticmethod
    def headless(
        transition_gamma: float | Iterable | Tuple[NumpyArray, NumpyArray],
        orbit_length: float,
        section_index: int = 0,
    ) -> DriftSimple:
        """
        Initialize object without simulation context


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
        from .._core.base import DynamicParameter

        d = DriftSimple(
            orbit_length=orbit_length,
            section_index=section_index,
        )
        if isinstance(transition_gamma, float):
            d.transition_gamma = backend.float(transition_gamma)
        else:
            d.schedule("transition_gamma", transition_gamma, mode="per-turn")
        from .._core.beam.base import BeamBaseClass
        from .._core.simulation.simulation import Simulation

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
        Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        super().on_init_simulation(simulation=simulation)
        self._simulation = simulation
        if (
            self.transition_gamma is None
        ) and "transition_gamma" not in self.schedules.keys():
            raise ValueError(
                "You need to define `transition_gamma` via `.transition_gamma=...` "
                "or `.schedule(attribute='transition_gamma', value=...)`"
            )

    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

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
        dt = backend.float(self.orbit_length / beam.reference_velocity)
        gamma = beam.reference_gamma
        eta_0 = self.alpha_0 - (1 / (gamma * gamma))
        backend.specials.drift_simple(
            dt=beam.write_partial_dt(),
            dE=beam.read_partial_dE(),
            T=backend.float(dt),
            eta_0=backend.float(eta_0),
            beta=backend.float(beam.reference_beta),
            energy=backend.float(beam.reference_total_energy),
        )
        beam.reference_time += dt

    def eta_0(self, gamma: float) -> backend.float:
        return backend.float(self.alpha_0 - (1 / (gamma * gamma)))

    # alias of momentum_compaction_factor
    @property  # as readonly attributes
    def alpha_0(self) -> backend.float:
        """Momentum compaction factor"""
        return self.momentum_compaction_factor

    def invalidate_cache(self):
        """Delete the stored values of functions with @cached_property"""
        # super()._invalidate_cache(DriftSimple.cached_props)
        pass


class DriftSpecial(DriftBaseClass):
    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        super().on_init_simulation(simulation=simulation)

    pass


class DriftXSuite(DriftBaseClass):
    def track(self, beam: BeamBaseClass) -> None:
        """Main simulation routine to be called in the mainloop

        Parameters
        ----------
        beam
            Beam class to interact with this element
        """
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called

        simulation
            Simulation context manager
        """
        super().on_init_simulation(simulation=simulation)

    pass
