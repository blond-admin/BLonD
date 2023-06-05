# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to wrap tracking into an iterator with option to call
user specified functions every n turns**

:Authors: **Simon Albright**
'''


class TrackIteration:
    '''
    Class to provide an iterator for tracking with an option to run passed
    functions every n turns

    Parameters
    ----------

    trackMap : iterable of objects
        Each object will be called on every turn with object.track()
    initTurn : integer
        The turn number tracking will start from, only used to initialise
        a turn counter
    finalTurn : integer
        The last turn number to track next(TrackIteration) will raise
        StopIteration when turnNumber == finalTurn

    Attributes
    ----------

    functionList : List of functions to be called with specified interval
    '''

    def __init__(self, trackMap, initTurn=0, finalTurn=-1):

        if not all((callable(m) for m in trackMap)):
            raise AttributeError("All map objects must be callable")

        self._map = trackMap
        if isinstance(initTurn, int):
            self.turnNumber = initTurn
        else:
            raise TypeError("initTurn must be an integer")
        if isinstance(finalTurn, int):
            self._finalTurn = finalTurn
        else:
            raise TypeError("finalTurn must be an integer")

        self.functionList = []

    def _track_turns(self, n_turns):
        '''
        Function to track for specified number of turns
        calls next() function n_turns times
        '''

        for i in range(n_turns):
            next(self)

    def add_function(self, predicate, repetionRate, *args, **kwargs):
        '''
        Takes a user defined callable and calls it every repetionRate
        number of turns with predicate(trackMap, turnNumber, ``*args``, ``**kwargs``)
        '''

        self.functionList.append((self._partial(predicate, args, kwargs), repetionRate))

    def __next__(self):
        '''
        First raises StopIteration if turnNumber == finalTurn

        Next calls track() from each element in trackMap list and raises
        StopIteration if no more turns available

        Finally iterates over each function specified in add_function
        and calls them with predicate(trackMap, turnNumber) if
        turnNumber % repetitionRate == 0
        '''

        if self.turnNumber == self._finalTurn:
            raise StopIteration

        try:
            for m in self._map:
                m()
        except IndexError:
            raise StopIteration

        self.turnNumber += 1

        for func, rate in self.functionList:
            if self.turnNumber % rate == 0:
                func(self._map, self.turnNumber)

        return self.turnNumber

    def __iter__(self):
        '''
        returns self
        '''

        return self

    def __call__(self, n_turns=1):
        '''
        Makes object callable with option to specify number of tracked turns
        default tracks 1 turn
        '''

        self._track_turns(n_turns)
        return self.turnNumber

    def _partial(self, predicate, args, kwargs):
        '''
        reimplementation of functools.partial to prepend
        rather than append to *args
        '''

        def partFunc(_map, turn):
            return predicate(_map, turn, *args, **kwargs)

        return partFunc
