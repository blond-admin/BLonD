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
    from typing import Dict, List, Self, Union, Iterable

    from .beam import Beam


class MultiTurnInjection:
    """
    Class to handle multi-turn injection.  Multiple Beam objects can be
    added along with the turn number on which they are injected.  On
    each call to track() the turn number will be checked and, if it
    matches the specified injection turn, the corresponding Beam object
    will be added to the simulation.

    Parameters
    ----------
    beam : Beam
        The instance of the Beam class to which injections are added
    """

    def __init__(self, beam: Beam):

        self.beam = beam
        self._injections: Dict[int, Union[Beam, Iterable[float]]] = {}

    def __next__(self):
        """
        If the current turn has an injection, the beam will be added to 
        the simulation.


        Raises
        ------
        StopIteration
            Raised when the specified injections have been exhausted
        """
        if self._counter[0] in self._injections.keys():
            self.beam += self._injections.pop(self._counter)
        elif len(self._injections):
            raise StopIteration("All defined injections have been used")

    def __iter__(self) -> Self:
        return self

    def track(self):
        """
        Track function, calls next(self), allows the MultiTurnInjection
        object to be used as a trackable object.
        """
        next(self)

    #TODO:  Not a good solution, but avoids passing in the RFStation object
    # could be solved with the proposed TurnCounter object
    def set_counter(self, counter: List[int]):
        """
        Add a 1D list of int to the object to be used as the turn 
        counter.  This should be derived from the RFStation object.
        """
        self._counter = counter

    def add_injection(self, beam: Union[Beam, Iterable[float]],
                      injection_turn: int = None):
        """
        Specify a Beam object to be used for an injection.  If no
        injection_turn is specified, it will be set to the the number
        of already defined injections + 1.

        Parameters
        ----------
        beam : Union[Beam, Iterable[float]]
            Either:
                An instance of the Beam class to be added for this
                injection
            Or:
                A [2, n] array of particle coordinates that define the
                (dt, dE) of the injected beam
        injection_turn : int, optional
            The turn number on which the beam will be injected.
            The default is None.
            If None, the value will be set to the number of already
            defined injections + 1
        
        Raises
        ------
        ValueError:
            If a Beam object is provided with a different ratio of
            particles per macroparticle to self.beam, a ValueError
            will be raised.

            If an array is provided, it must be of shape (2, n), if not
            a ValueError will be raised.
        """

        if isinstance(beam, beam.Beam):
            self._check_beam_injection(beam)
        else:
            self._check_array_injection(beam)

        if injection_turn is None:
            injection_turn = len(self._injections) + 1

        self._injections[injection_turn] = np.array(beam)
    

    def _check_beam_injection(self, beam: Beam):

        if beam.ratio != self.beam.ratio:
            raise ValueError("The particles per macroparticle ratio must be "
                             + "the same for all injections.")

    def _check_array_injection(self, beam: Iterable[float]):

        beam = np.array(beam)
        if beam.shape[0] != 2 or len(beam.shape) != 2:
            raise ValueError("Injection array must be of shape (2, n)")
