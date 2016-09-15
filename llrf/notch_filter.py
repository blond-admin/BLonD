
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Methods to apply a notch filter to a specified impedance source**

:Authors: **Simon Albright, Danilo Quartullo**
'''

from __future__ import division
from builtins import range, object
import numpy as np


#FIRST METHOD
def impedance_notches(f_rev, frequencies, real_imp, imag_imp, list_harmonics, list_width_depth):
	
	halfmajorWidth = list_width_depth[0]
	halfminorWidth = list_width_depth[1]
	majorDepth = list_width_depth[2]
	minorDepth = list_width_depth[3]
	array_left = np.array([])
	indices_final = np.array([])
	frequencies_remainder_array = np.array([])
	Re_Z_remainder_array = np.array([])
	Im_Z_remainder_array = np.array([])
	
	for i in list_harmonics:
		exec("yre_"+str(i)+" = np.interp(i*f_rev, frequencies, real_imp)")
		exec("yim_"+str(i)+" = np.interp(i*f_rev, frequencies, imag_imp)")
		array_left = np.append(array_left, np.array([i*f_rev-halfmajorWidth, i*f_rev+halfmajorWidth]))
		exec("indices_"+str(i)+" = np.where((frequencies >= (i*f_rev-halfmajorWidth))&(frequencies <= (i*f_rev+halfmajorWidth)))[0]")
		indices_final = np.append(indices_final, locals()['indices_'+str(i)])
		frequencies_remainder_array = np.append(frequencies_remainder_array, np.array([i*f_rev-halfminorWidth, i*f_rev, i*f_rev+halfminorWidth]))
		Re_Z_remainder_array = np.append(Re_Z_remainder_array, np.array([locals()['yre_'+str(i)]/minorDepth, locals()['yre_'+str(i)]/majorDepth, locals()['yre_'+str(i)]/minorDepth]))
		Im_Z_remainder_array = np.append(Im_Z_remainder_array, np.array([locals()['yim_'+str(i)]/minorDepth, locals()['yim_'+str(i)]/majorDepth, locals()['yim_'+str(i)]/minorDepth]))	
		
	left_yre = np.interp(array_left, frequencies, real_imp)
	left_yim = 	np.interp(array_left, frequencies, imag_imp)
	frequencies_remainder = np.delete(frequencies, indices_final)
	Re_Z_remainder = np.delete(real_imp, indices_final)
	Im_Z_remainder = np.delete(imag_imp, indices_final)
	
	frequencies_remainder_array = np.append(frequencies_remainder_array, array_left)
	Re_Z_remainder_array = np.append(Re_Z_remainder_array, left_yre)
	Im_Z_remainder_array = np.append(Im_Z_remainder_array, left_yim)
	frequencies_remainder = np.append(frequencies_remainder, frequencies_remainder_array)
	Re_Z_remainder = np.append(Re_Z_remainder, Re_Z_remainder_array)
	Im_Z_remainder = np.append(Im_Z_remainder, Im_Z_remainder_array)
	
	ordered_freq = np.argsort(frequencies_remainder)
	frequencies_remainder = np.take(frequencies_remainder, ordered_freq)
	Re_Z_closed = np.take(Re_Z_remainder, ordered_freq)
	Im_Z_closed = np.take(Im_Z_remainder, ordered_freq)
	
	return [frequencies_remainder, Re_Z_closed, Im_Z_closed]


#SECOND METHOD, DO NOT USE IT!!! 
#SLOWER AND LESS PRECISE THAN FIRST METHOD, MAYBE TO BE REMOVED
class Filter(object):

	def __init__(self, majorWidth, minorWidth, majorDepth, minorDepth, harmonics, resolution = None):

		self.majorWidth = majorWidth
		self.minorWidth = minorWidth
		self.majorDepth = majorDepth
		self.minorDepth = minorDepth
		self.harmonics = harmonics

		self.notches = np.ascontiguousarray([1, self.minorDepth, self.majorDepth, self.minorDepth, 1]*len(self.harmonics))
		self.notchFreqs = np.zeros([5*len(self.harmonics)])

		if resolution is None:
			self.interpFreqs = np.linspace(min(harmonics), max(harmonics), 1000*len(harmonics) + 1)
		elif isinstance(resolution, np.ndarray):
			self.interpFreqs = resolution
		elif isinstance(resolution, list) or isinstance(resolution, tuple):
			self.interpFreqs = np.linspace(resolution[0], resolution[1], resolution[2])
		else:
			self.interpFreqs = np.linspace(0, max(harmonics), resolution)


	def make_filter(self, frev):

		self.frev = frev

		for i in range(len(self.harmonics)):

			self.notchFreqs[i*5] = self.harmonics[i] - 0.5*self.majorWidth/self.frev
			self.notchFreqs[i*5+1] = self.harmonics[i] - 0.5*self.minorWidth/self.frev
			self.notchFreqs[i*5+2] = self.harmonics[i]
			self.notchFreqs[i*5+3] = self.harmonics[i] + 0.5*self.minorWidth/self.frev
			self.notchFreqs[i*5+4] = self.harmonics[i] + 0.5*self.majorWidth/self.frev


	def filter_impedance(self, frequencies, real_imp = None, imag_imp = None):

		step = np.diff(self.interpFreqs*self.frev)[0]
		
		if np.min(frequencies) < np.min(self.interpFreqs*self.frev):
			nSteps = (np.min(self.interpFreqs*self.frev)-np.min(frequencies))/step
			lower = np.linspace(np.min(self.interpFreqs*self.frev)-(nSteps+1)*step, np.min(self.interpFreqs*self.frev), nSteps+2)
		else:
			lower = np.array([])
		
		
		if np.max(frequencies) > np.max(self.interpFreqs*self.frev):
			nSteps = (np.max(frequencies)-np.max(self.interpFreqs*self.frev))/step
			upper = np.linspace(np.max(self.interpFreqs*self.frev), np.max(self.interpFreqs*self.frev)+(nSteps+1)*step, nSteps+2)
		else:
			upper = np.array([])
		
		
		newFreqArray = np.concatenate((lower, self.interpFreqs*self.frev, upper))
		
		notchInterp = np.interp(newFreqArray, self.notchFreqs*self.frev, self.notches)
		
		if real_imp is not None:
			realInterp = np.interp(newFreqArray, frequencies, real_imp)*notchInterp
		else:
			realInterp = None
		
		if imag_imp is not None:
			imagInterp = np.interp(newFreqArray, frequencies, imag_imp)*notchInterp
		else:
			imagInterp = None
		
		return (newFreqArray, realInterp, imagInterp)


