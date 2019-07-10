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

#General imports
import numpy as np

#BLonD imports
import blond.utils.exceptions as blExcept


#Check input and return InputDataError exception with user defined
#message if False
def check_input(variable, msg, *args):
    
    result = check_data_dimensions(variable, *args)
    
    if result[0]:
        return result
    else:
        raise blExcept.InputDataError(msg)


#General function to check if input_data is number, or nD array
#for each member of args the input_data is checked
#0 is taken as checking if input_data is a number
#integers are taken as the length of a list-like input
#tuples or lists are taken as dimensions
#[[1, 2], [1, 2]] will return true for length = 2 or dims = (2, 2)
def check_data_dimensions(input_data, *args):
    
    success = False
    
    for a in args:
        if a == 0:
            if _check_number(input_data):
                success = True

        elif isinstance(a, int):
            if _check_length(input_data, a):
                success = True

        else:
            if _check_dimensions(input_data, a):
                success = True
            
        if success:
            return True, type(input_data)
                
    else:
        return False, type(input_data)


#returns True if input_data can be cast to int
def _check_number(input_data):
    
    try:
        int(input_data)
        if isinstance(input_data, np.ndarray):
            raise TypeError
        return True
    except (TypeError, ValueError):
        return False
    

#Returns True if len(input_data) == length
#Should this return True if n-dim > 1?
def _check_length(input_data, length):
    
    if not _check_number(length):
        raise TypeError("Length must be numeric")
    
    try:
        return len(input_data) == length
    except TypeError:
        return False
    
    
#Casts input_data to numpy array and dimensions to tuple
#compares shape of array to tuple and returns True if equal.
#If dim[n] == -1 it is set to input_data.shape[n] to allow comprehension
#of arrays with arbitrary length in one or more dimensions.
def _check_dimensions(input_data, dim):
    
    try:
        iter(dim)
    except TypeError:
        dim = [dim]
    
    inputShape = np.array(input_data).shape
    
    try:
        if -1 in dim:
            try:
                dim = [inputShape[i] if dim[i] == -1 \
                                     else dim[i] for i in range(len(dim))]
            except IndexError:
                return False
    except TypeError:
        raise TypeError("dim must be number or iterable of numbers")
        
    return inputShape == tuple(dim)
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

    
