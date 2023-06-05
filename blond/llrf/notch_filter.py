
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Method to apply a notch filter to a specified impedance source**

:Authors: **Danilo Quartullo**
'''

from __future__ import division

import numpy as np


# FIRST METHOD
def impedance_notches(f_rev, frequencies, imp_source, list_harmonics, list_width_depth):

    halfmajorWidth = list_width_depth[0]
    halfminorWidth = list_width_depth[1]
    majorDepth = list_width_depth[2]
    minorDepth = list_width_depth[3]
    array_left = np.array([])
    indices_final = np.array([])
    frequencies_remainder_array = np.array([])
    Z_remainder_array = np.array([])

    for i in list_harmonics:
        exec("y_" + str(i) + " = np.interp(i*f_rev, frequencies, imp_source)")
        array_left = np.append(array_left, np.array([i * f_rev - halfmajorWidth, i * f_rev + halfmajorWidth]))
        exec("indices_" + str(i) + " = np.where((frequencies >= (i*f_rev-halfmajorWidth))&(frequencies <= (i*f_rev+halfmajorWidth)))[0]")
        indices_final = np.append(indices_final, locals()['indices_' + str(i)])
        frequencies_remainder_array = np.append(frequencies_remainder_array, np.array([i * f_rev - halfminorWidth, i * f_rev, i * f_rev + halfminorWidth]))
        Z_remainder_array = np.append(Z_remainder_array, np.array([locals()['y_' + str(i)] / minorDepth, locals()['y_' + str(i)] / majorDepth, locals()['y_' + str(i)] / minorDepth]))

    left_y = np.interp(array_left, frequencies, imp_source)
    frequencies_remainder = np.delete(frequencies, indices_final)
    Z_remainder = np.delete(imp_source, indices_final)

    frequencies_remainder_array = np.append(frequencies_remainder_array, array_left)
    Z_remainder_array = np.append(Z_remainder_array, left_y)
    frequencies_remainder = np.append(frequencies_remainder, frequencies_remainder_array)
    Z_remainder = np.append(Z_remainder, Z_remainder_array)

    ordered_freq = np.argsort(frequencies_remainder)
    frequencies_remainder = np.take(frequencies_remainder, ordered_freq)
    Z_closed = np.take(Z_remainder, ordered_freq)

    return [frequencies_remainder, Z_closed]
