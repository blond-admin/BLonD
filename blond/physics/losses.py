# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
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
from blond.core.base import BeamPhysicsRelevant

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from blond import Simulation
    from blond.core.beam.base import BeamBaseClass


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

    def track(self, beam: BeamBaseClass) -> None:  # pragma: no cover
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


class BoxLosses(LossesBaseClass):
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
        Macro-particles with ``dE < t_min`` will be labeled/removed, in [s].
    e_max
        Macro-particles with ``dE > t_min`` will be labeled/removed, in [s].

    Attributes
    ----------
    t_min : float
        Macro-particles with ``dt < t_min`` will be labeled/removed, in [s].
    t_max : float
        Macro-particles with ``dt > t_max`` will be labeled/removed, in [s].
    e_min : float
        Macro-particles with ``dE < t_min`` will be labeled/removed, in [s].
    e_max : float
        Macro-particles with ``dE > t_min`` will be labeled/removed, in [s].
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

    def on_init_simulation(self, simulation: Simulation) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            Simulation context manager.
        """
        pass

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
            Simulation context manager.
        beam
            Simulation beam object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        pass

    def track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        backend.specials.loss_box(
            e_max=self.e_max,
            e_min=self.e_min,
            t_min=self.t_min,
            t_max=self.t_max,
            dt=beam.read_partial_dt(),
            dE=beam.read_partial_dE(),
            flags=beam.write_partial_flags(),
        )
        self._purge_particles(beam=beam, force=False)
