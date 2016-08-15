
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Methods to apply a notch filter to a specified impedance source**

:Authors: **Simon Albright**
'''

from __future__ import division
from builtins import range, object
import numpy as np



#FIRST METHOD
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

#SECOND METHOD
#it needs to be generalised, it works now specifically for Finemet cavities
def finemet_one_gap_closed_loop(f_rev, frequencies, real_imp, imag_imp):

	yre_1 = np.interp(f_rev, frequencies, real_imp)
	yim_1 = np.interp(f_rev, frequencies, imag_imp)
	yre_2 = np.interp(2*f_rev, frequencies, real_imp)
	yim_2 = np.interp(2*f_rev, frequencies, imag_imp)
	yre_3 = np.interp(3*f_rev, frequencies, real_imp)
	yim_3 = np.interp(3*f_rev, frequencies, imag_imp)
	yre_4 = np.interp(4*f_rev, frequencies, real_imp)
	yim_4 = np.interp(4*f_rev, frequencies, imag_imp)
	yre_5 = np.interp(5*f_rev, frequencies, real_imp)
	yim_5 = np.interp(5*f_rev, frequencies, imag_imp)
	yre_6 = np.interp(6*f_rev, frequencies, real_imp)
	yim_6 = np.interp(6*f_rev, frequencies, imag_imp)
	yre_7 = np.interp(7*f_rev, frequencies, real_imp)
	yim_7 = np.interp(7*f_rev, frequencies, imag_imp)
	yre_8 = np.interp(8*f_rev, frequencies, real_imp)
	yim_8 = np.interp(8*f_rev, frequencies, imag_imp)
	 
	left_yre = np.interp(np.array([f_rev-16e3,f_rev+16e3, 2*f_rev-16e3, 2*f_rev+16e3,
	                               3*f_rev-16e3,3*f_rev+16e3, 4*f_rev-16e3,4*f_rev+16e3,
	                               5*f_rev-16e3,5*f_rev+16e3, 6*f_rev-16e3,6*f_rev+16e3,
	                               7*f_rev-16e3,7*f_rev+16e3, 8*f_rev-16e3,8*f_rev+16e3]), frequencies, real_imp)
	 
	left_yim = np.interp(np.array([1*f_rev-16e3,1*f_rev+16e3, 2*f_rev-16e3,2*f_rev+16e3,
	                               3*f_rev-16e3,3*f_rev+16e3, 4*f_rev-16e3,4*f_rev+16e3,
	                               5*f_rev-16e3,5*f_rev+16e3, 6*f_rev-16e3,6*f_rev+16e3,
	                               7*f_rev-16e3,7*f_rev+16e3, 8*f_rev-16e3,8*f_rev+16e3]), frequencies, imag_imp)
	 
	indices_1 = np.where((frequencies >= (1*f_rev-16e3))&(frequencies <= (1*f_rev+16e3)))[0]
	indices_2 = np.where((frequencies >= (2*f_rev-16e3))&(frequencies <= (2*f_rev+16e3)))[0]
	indices_3 = np.where((frequencies >= (3*f_rev-16e3))&(frequencies <= (3*f_rev+16e3)))[0]
	indices_4 = np.where((frequencies >= (4*f_rev-16e3))&(frequencies <= (4*f_rev+16e3)))[0]
	indices_5 = np.where((frequencies >= (5*f_rev-16e3))&(frequencies <= (5*f_rev+16e3)))[0]
	indices_6 = np.where((frequencies >= (6*f_rev-16e3))&(frequencies <= (6*f_rev+16e3)))[0]
	indices_7 = np.where((frequencies >= (7*f_rev-16e3))&(frequencies <= (7*f_rev+16e3)))[0]
	indices_8 = np.where((frequencies >= (8*f_rev-16e3))&(frequencies <= (8*f_rev+16e3)))[0]
	indices_final = np.concatenate((indices_1,indices_2,indices_3,indices_4,indices_5,indices_6,indices_7,indices_8))
	 
	frequencies_remainder = np.delete(frequencies, indices_final)
	 
	Re_Z_remainder = np.delete(real_imp, indices_final)
	Im_Z_remainder = np.delete(imag_imp, indices_final)
	 
	frequencies_remainder = np.append(frequencies_remainder, 
	                                       np.array([1*f_rev-8e3, 1*f_rev, 1*f_rev+8e3, 2*f_rev-8e3, 2*f_rev, 2*f_rev+8e3,
	                                                 3*f_rev-8e3, 3*f_rev, 3*f_rev+8e3, 4*f_rev-8e3, 4*f_rev, 4*f_rev+8e3,
	                                                 5*f_rev-8e3, 5*f_rev, 5*f_rev+8e3, 6*f_rev-8e3, 6*f_rev, 6*f_rev+8e3,
	                                                 7*f_rev-8e3, 7*f_rev, 7*f_rev+8e3, 8*f_rev-8e3, 8*f_rev, 8*f_rev+8e3,
	                                                 1*f_rev-16e3, 1*f_rev+16e3, 2*f_rev-16e3, 2*f_rev+16e3,
	                                                 3*f_rev-16e3, 3*f_rev+16e3, 4*f_rev-16e3, 4*f_rev+16e3,
	                                                 5*f_rev-16e3, 5*f_rev+16e3, 6*f_rev-16e3, 6*f_rev+16e3,
	                                                 7*f_rev-16e3, 7*f_rev+16e3, 8*f_rev-16e3, 8*f_rev+16e3]))
	 
	Re_Z_remainder = np.append(Re_Z_remainder, 
	                                       np.array([yre_1/1.4, yre_1/63, yre_1/1.4, yre_2/1.4, yre_2/63, yre_2/1.4,
	                                                 yre_3/1.4, yre_3/63, yre_3/1.4, yre_4/1.4, yre_4/63, yre_4/1.4,
	                                                 yre_5/1.4, yre_5/63, yre_5/1.4, yre_6/1.4, yre_6/63, yre_6/1.4,
	                                                 yre_7/1.4, yre_7/63, yre_7/1.4, yre_8/1.4, yre_8/63, yre_8/1.4,
	                                                 left_yre[0], left_yre[1], left_yre[2], left_yre[3],
	                                                 left_yre[4], left_yre[5], left_yre[6], left_yre[7],
	                                                 left_yre[8], left_yre[9], left_yre[10], left_yre[11],
	                                                 left_yre[12], left_yre[13], left_yre[14], left_yre[15]]))
	 
	Im_Z_remainder = np.append(Im_Z_remainder, 
	                                       np.array([yim_1/1.4, yim_1/63, yim_1/1.4, yim_2/1.4, yim_2/63, yim_2/1.4,
	                                                 yim_3/1.4, yim_3/63, yim_3/1.4, yim_4/1.4, yim_4/63, yim_4/1.4,
	                                                 yim_5/1.4, yim_5/63, yim_5/1.4, yim_6/1.4, yim_6/63, yim_6/1.4,
	                                                 yim_7/1.4, yim_7/63, yim_7/1.4, yim_8/1.4, yim_8/63, yim_8/1.4,
	                                                 left_yim[0], left_yim[1], left_yim[2], left_yim[3],
	                                                 left_yim[4], left_yim[5], left_yim[6], left_yim[7],
	                                                 left_yim[8], left_yim[9], left_yim[10], left_yim[11],
	                                                 left_yim[12], left_yim[13], left_yim[14], left_yim[15]]))
	 
	ordered_freq = np.argsort(frequencies_remainder)
	frequencies_remainder = np.take(frequencies_remainder, ordered_freq)
	Re_Z_closed = np.take(Re_Z_remainder, ordered_freq)
	Im_Z_closed = np.take(Im_Z_remainder, ordered_freq)
	
	return [frequencies_remainder, Re_Z_closed, Im_Z_closed]
