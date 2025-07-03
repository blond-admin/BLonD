# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Module containing the multi-turn injection class, responsible for
adding additional injections to the beam object.

:Authors: **Simon Albright**
"""

# Standard library imports
from __future__ import annotations
from typing import TYPE_CHECKING

# General imports
import numpy as np

# BLonD imports
from . import beam

if TYPE_CHECKING:
    from typing import Dict, List

    from .beam import Beam


class MultiTurnInjection:

    def __init__(self, beam: Beam):

        self.beam = beam
        self._injections: Dict[int, Beam] = {}

    def __next__(self):
        if self._counter[0] in self._injections.keys():
            self.beam += self._injections.pop(self._counter)
        elif len(self._injections):
            raise StopIteration("All defined injections have been used")

    def __iter__(self):
        return self

    def track(self):
        next(self)

    #TODO:  Not a good solution, but avoids passing in the RFStation object
    # could be solved with the proposed TurnCounter object
    def set_counter(self, counter: List[int]):
        self._counter = counter

    def add_injection(self, beam: Beam, injection_turn: int = None):

        if beam.ratio != self.beam.ratio:
            raise ValueError("The particles per macroparticle ratio must be "
                             + "the same for all injections.")

        if injection_turn is None:
            injection_turn = len(self._injections) + 1

        self._injections[injection_turn] = beam
    