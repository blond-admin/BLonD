# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Implementation of the PSB beam control.

Notes
-----
None.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from blond.physics.feedbacks.beam_feedback import (
    BeamFeedbackBase,
)

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


class PSBBeamControl(BeamFeedbackBase):
    r"""
    Class for the PSB beam control.

    This class implements the feedbacks present in the beam
    control of the rf system in the Proton Synchrotron Booster.

    Parameters
    ----------
    pl_gain
        The gain of the beam-phase loop.
    rl_gain
        The gain of the radial loop.
    *args
        Variable positional arguments.
    **kwargs
        Variable keyword arguments.
    """

    def __init__(
        self,
        pl_gain: float,
        rl_gain: list[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.pl_gain = pl_gain
        self.rl_gain = rl_gain

        self.lhc_y = 0

        self.domega_rf = 0.0
        self.dphi = 0.0
        self.reference = 0.0

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs,
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
            **kwargs,
        )

        # TODO: implement

    def get_beam_attribute(self, beam: BeamBaseClass):
        """
        Calculate the beam phase.

        This method implements the measurement of the beam, which is the
        input for the feedback. In the case of the PSB beam control, this
        is the rf component phase of the beam.

        Parameters
        ----------
        beam
            A beam object to extract the beam attribute from.
        """
        self.beam_phase()

    def compute_correction(self, beam: BeamBaseClass):
        """
        Calculate the frequency correction from the beam control.

        This method implements the feedback systems in the PSB beam control, i.e.
        the beam-phase loop and the synchronization loop.

        Parameters
        ----------
        beam
            A beam object to extract the beam attribute from.
        """
        pass
        # TODO: implement
