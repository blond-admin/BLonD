# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to generate coasting beam**

:Authors: **Simon Albright**
'''


class TrackIteration(object):

    def __init__(self, trackMap, initTurn = 0, finalTurn = -1):
        
        if not all((hasattr(m, 'track') for m in trackMap)):
            raise AttributeError("All map objects must be trackable")
            
        self._map = trackMap
        self._initTurn = initTurn
        self._finalTurn = finalTurn
        self.turnNumber = initTurn
        self.functionList = []

    def track_turns(self, n_turns):
        for i in range(n_turns):
            next(self)


    def add_function(self, predicate, repitionRate, *args, **kwargs):
        
        '''
        Takes a user defined callable and calls it every repritionRate
        number of turns with predicate(trackMap, turnNumber, *args, **kwargs)
        '''
        
        self.functionList.append((self._partial(predicate, args, kwargs), repitionRate))
        

    def __next__(self):

        if self.turnNumber == self._finalTurn:
            raise StopIteration

        try:
            for m in self._map:
                m.track()
            self.turnNumber += 1
            for func, rate in self.functionList:
                if self.turnNumber % rate == 0:
                    func(self._map, self.turnNumber)
            return self.turnNumber

        except IndexError:
            raise StopIteration


    def __iter__(self):
        return self


    def __call__(self, n_turns = 1):
        self.track_turns(n_turns)
        
    
    def _partial(self, predicate, args, kwargs):
        '''
        reimplementation of functools.partial to prepend rather than append to *args
        '''
        def partFunc(_map, turn):
            return predicate(_map, turn, *args, **kwargs)

        return partFunc
