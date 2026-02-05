# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Holds the `PooledInterpolationKick` to delay the execution of kicks.

Notes
-----
Authors:
Simon Lauber
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from blond import backend
from blond.core.base import BeamPhysicsRelevant, Preparable

if TYPE_CHECKING:  # pragma: no cover
    from typing import TypeVar

    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation

    T = TypeVar("T")

logger = logging.getLogger(__name__)


class PooledInterpolationKick(BeamPhysicsRelevant):
    """
    Aggregate interpolated kicks from multiple elements for improved performance.

    In many simulations, several components (e.g. wakefields and RF cavities)
    are placed sequentially along the beamline. Instead of computing and
    applying each kick immediately, this class collects the individual
    interpolated contributions, sums them, and applies the combined kick
    at once, significantly reducing computational overhead.

    Parameters
    ----------
    section_index
        Identifier grouping elements that belong to the same section of the ring.
        Defaults to 0.
    name
        Human-readable name for the element. If not provided, a unique name is
        automatically generated.
    **kwargs
        Additional keyword arguments passed to the parent.
    """

    def __init__(
        self, section_index: int = 0, name: str | None = None, **kwargs
    ) -> None:
        super().__init__(section_index, name)
        self._buffer_voltage = {}
        self._buffer_time_axis = {}

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        """
        self.wipe_buffer()

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs,
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
        self.wipe_buffer()

    def wipe_buffer(self) -> None:
        """
        Reset the buffer and forget about all previous `register` calls.
        """
        self._buffer_voltage = {}
        self._buffer_time_axis = {}

    def register(self, time_axis: NumpyArray, voltage: NumpyArray) -> None:
        """
        Register a voltage to be applied with the next `track` invocation.

        Parameters
        ----------
        time_axis
            Time axis of the voltage, in [s].
        voltage
            Voltage along the time axis, in [V].
        """
        from blond.physics.impedances.sources import get_hash

        key = get_hash(time_axis)
        if key in self._buffer_voltage:
            self._buffer_voltage[key] += voltage
        else:
            self._buffer_voltage[key] = voltage.copy()
            self._buffer_time_axis[key] = time_axis.copy()

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Apply the element's physics effect to the beam.

        Parameters
        ----------
        beam
            The beam object whose state will be updated by this element.
        """
        for key in self._buffer_voltage.keys():
            if beam.common_array_size > 0:
                voltage = self._buffer_voltage[key]
                time = self._buffer_time_axis[key]
                backend.specials.change_dE_interpolated(
                    dt=beam.read_partial_dt(),
                    dE=beam.write_partial_dE(),
                    bin_centers=time,
                    voltage=voltage,
                    charge=beam.particle_type.charge,
                    acceleration_kick=0.0,
                )
            self._buffer_voltage[key][:] = 0  # consume buffer


class SupportsPooledInterpolationKickMixIn(Preparable):
    """
    Make a class hold the delayed kicks.

    Parameters
    ----------
    delayed_kick
        The common interface to apply the kick later.
        `PooledInterpolationKick.track(...)` must be executed elsewhere.
    **kwargs
        Additional keyword arguments.

    Notes
    -----
    If you are adding `SupportsPooledInterpolationKickMixIn` to a class,
    make sure to implement in the  `_track` method
    the intended behaviour using `self._delayed_kick.register(...)`.
    """

    def __init__(
        self,
        delayed_kick: PooledInterpolationKick | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self._delayed_kick = delayed_kick

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
            simulation=simulation,
            beam=beam,
            n_turns=n_turns,
        )

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        """
        super().on_init_simulation(simulation=simulation)
        if self._delayed_kick is not None:
            self._check_executor_in_pipeline(simulation=simulation)

    def _check_executor_in_pipeline(self, simulation: Simulation):
        """
        Check if the `PooledInterpolationKick` is in the element pipeline.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        """
        delayed_kicks = simulation.ring.elements.get_elements(
            PooledInterpolationKick, recursive=True
        )
        assert self._delayed_kick in delayed_kicks, (
            "You intend to delay the kicks, but no executor was found in the element pipeline."
        )
