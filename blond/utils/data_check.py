# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public License version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.md.
# In applying this license, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Function(s) for pre-processing input data**

:Authors: **Simon Albright**
'''

import sys

def check_dimensions(input_data, *args):
    
    
    for a in args:
        if a == 0:
            if _check_number(input_data):
                return True

        elif isinstance(a, int):
            if _check_length(input_data, a):
                return True

        else:
            if _check_dimensions(input_data, a):
                return True
                
    else:
        return False



def _check_number(input_data):
    
    try:
        int(input_data)
        return True
    except TypeError:
        return False
    
    
def _check_length(input_data, length):
    
    return len(input_data) == length
    
    
def _check_dimensions(input_data, dim):
    
    return np.array(input_data).shape == tuple(dim)
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

    
