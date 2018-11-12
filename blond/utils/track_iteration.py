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
    
    def __init__(self, map):
        
        if not all((hasattr(m, 'track') for m in map)):
            raise AttributeError("All map objects must be trackable")
            
        self.map = map
        


    def track_turns(self, n_turns):
        
        for i in range(n_turns):
            next(self)


    def __next__(self):
        for m in self.map:
            m.track()
    
    def __iter__(self):
        return self

