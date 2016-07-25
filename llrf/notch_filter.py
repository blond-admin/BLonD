
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
import numpy as np
import matplotlib.pyplot as plt
import time

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
