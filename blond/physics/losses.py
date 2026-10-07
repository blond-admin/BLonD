# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Collection of implementations to handle beam losses in synchrotrons."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import backend
from blond.core.base import BeamPhysicsRelevant, Schedulable

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.base import DynamicParameter
    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


class LossesBaseClass(BeamPhysicsRelevant, ABC):
    """
    Base class for labeling/removing lost particles.

    Parameters
    ----------
    purge_flagged_macroparticles
        If true, particles will be immediately removed
        from the ``Beam`` array when ``track(...)`` is executed.

        If false, the ``Beam.flags`` will be set, but particles will still
        be considered for beam physics.

    Attributes
    ----------
    purge_flagged_macroparticles
        If true, particles will be immediately removed
        from the ``Beam`` array when ``track(...)`` is executed.

        If false, the ``Beam.flags`` will be set, but particles will still
        be considered for beam physics.
    """

    def __init__(self, purge_flagged_macroparticles: bool) -> None:
        super().__init__()
        self.purge_flagged_macroparticles = purge_flagged_macroparticles

    def _track(self, beam: BeamBaseClass) -> None:  # pragma: no cover
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        pass

    def _purge_particles(
        self, beam: BeamBaseClass, force: bool = False
    ) -> None:
        """
        Potentially remove flagged particles.

        Parameters
        ----------
        beam
            Beam to remove the particles from.
        force
            If true, will definitely purge particles.
            Otherwise, it depends on `self.purge_flagged_macroparticles`.
        """
        if self.purge_flagged_macroparticles or force:
            beam.purge_flagged_entries()


class BoxLosses(LossesBaseClass, Schedulable):
    """
    Particles outside a rectangle will be flagged lost.

    Parameters
    ----------
    purge_flagged_macroparticles
        If true, particles will be immediately removed
        from the ``Beam`` array when ``track(...)`` is executed.

        If false, the ``Beam.flags`` will be set, but particles will still
        be considered for beam physics.
    t_min
        Macro-particles with ``dt < t_min`` will be labeled/removed, in [s].
    t_max
        Macro-particles with ``dt > t_max`` will be labeled/removed, in [s].
    e_min
        Macro-particles with ``dE < e_min`` will be labeled/removed, in [eV].
    e_max
        Macro-particles with ``dE > e_max`` will be labeled/removed, in [eV].

    Attributes
    ----------
    t_min
        Macro-particles with ``dt < t_min`` will be labeled/removed, in [s].
    t_max
        Macro-particles with ``dt > t_max`` will be labeled/removed, in [s].
    e_min
        Macro-particles with ``dE < e_min`` will be labeled/removed, in [eV].
    e_max
        Macro-particles with ``dE > e_max`` will be labeled/removed, in [eV].
    """

    def __init__(
        self,
        purge_flagged_macroparticles: bool,
        t_min: float | None = None,
        t_max: float | None = None,
        e_min: float | None = None,
        e_max: float | None = None,
    ) -> None:
        super().__init__(
            purge_flagged_macroparticles=purge_flagged_macroparticles,
        )
        if t_min is None:
            # USe float instead of None
            # for easier implementation of kernels.
            t_min = np.finfo(backend.float).min
        if t_max is None:
            # USe float instead of None
            # for easier implementation of kernels.
            t_max = np.finfo(backend.float).max
        if e_min is None:
            # USe float instead of None
            # for easier implementation of kernels.
            e_min = np.finfo(backend.float).min
        if e_max is None:
            # USe float instead of None
            # for easier implementation of kernels.
            e_max = np.finfo(backend.float).max

        assert t_min < t_max, (
            f"`t_min` must be smaller than `t_max`, but got {t_min=} and {t_max=}."
        )
        assert e_min < e_max, (
            f"`e_min` must be smaller than `e_max`, but got {e_min=} and {e_max=}."
        )

        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.e_min = float(e_min)
        self.e_max = float(e_max)

        self._turn_counter: DynamicParameter | None = None
        self._register_schedulable_variables(
            "t_min", "t_max", "e_min", "e_max"
        )

    def on_init_simulation(self, simulation: Simulation, **kwargs) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        **kwargs
            Configure parameters collected by the MRO chain.
        """
        super().on_init_simulation(
            simulation, turn_counter=simulation.turn_counter, **kwargs
        )

    def configure(
        self, *, turn_counter: DynamicParameter | None = None, **kwargs
    ) -> None:
        """
        Store the turn counter needed for schedule application during tracking.

        Parameters
        ----------
        turn_counter
            Live turn counter; accessed as ``turn_counter.value`` each track call.
        **kwargs
            Passed to the next level in the MRO chain.
        """
        self._turn_counter = turn_counter
        super().configure(**kwargs)

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        if self.schedule_active:
            assert self._turn_counter is not None, (
                "Turn counter must be set with active scheduling."
            )
            self.apply_schedules(
                turn_i=self._turn_counter.value,
                reference_time=beam.reference.time,
            )
        if beam.common_array_size > 0:
            backend.specials.loss_box(
                e_max=backend.float(self.e_max),
                e_min=backend.float(self.e_min),
                t_min=backend.float(self.t_min),
                t_max=backend.float(self.t_max),
                dt=beam.read_partial_dt(),
                dE=beam.read_partial_dE(),
                flags=beam.write_partial_flags(),
            )
            self._purge_particles(beam=beam, force=False)
