# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


"""
**Various beam phase loops with optional synchronisation/frequency/radial loops
for the CERN machines**

Notes
-----
Authors:
Helga Timko
Alexandre Lasheen
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from blond import Simulation
from blond.physics.feedbacks.base import (
    GlobalFeedback,
)

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass
    from blond.physics.profiles import ProfileBaseClass


class BeamFeedbackBase(GlobalFeedback):
    def __init__(
        self,
        profile: ProfileBaseClass,
        delay: int = 0,
        window_coefficient: float = 0.0,
        time_offset: float | None = None,
        beam_current_threshold=None,
    ):
        super().__init__(profile=profile)
        self.delay = delay
        self.window_coefficient = window_coefficient
        self.time_offset = time_offset
        self.beam_current_threshold = beam_current_threshold

        self.domega_rf = 0

        self.dphi: float = 0.0

        self.phi_beam: float = 0.0

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs,
    ) -> None:
        pass

    @abstractmethod
    def get_beam_attribute(self, beam: BeamBaseClass):
        # could be mean energy, mean phase or whatever
        pass

    @abstractmethod
    def apply_corrections(self, beam: BeamBaseClass):
        # shift the RF station phase or so
        pass

    def beam_phase(self):
        pass

    def phase_difference(
        self, beam: BeamBaseClass, RFnoise=None, noiseFB=None
    ):
        pass

    def _track(self, beam: BeamBaseClass):
        pass
