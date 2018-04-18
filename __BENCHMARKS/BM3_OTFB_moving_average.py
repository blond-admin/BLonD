# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Example for llrf.filters and llrf.cavity_feedback

:Authors: **Helga Timko**
"""

import numpy as np
import matplotlib.pyplot as plt 

from llrf.signal_processing import moving_average

n = 10000
n_ma = 100
iterations = 10

time = np.linspace(0, 10.e-6, n)
tau = 1.e-6
signal = 1 - np.exp(-time/tau) + 0.01*(np.random.randn(n) - 0.5)
time = np.linspace(0, 20.e-6, 2*n)
signal = np.concatenate((signal, signal))

plt.figure('mov avg')
plt.clf()
plt.plot(1e6*time, signal)
prev = np.zeros(n_ma-1)

for i in range(iterations):
    print("Average of end of previous signal", np.mean(prev))
    tmp = signal[-n_ma+1:]
    signal = moving_average(signal, n_ma, prev)
    prev = np.copy(tmp)
    print("Length of signal", len(signal))
    plt.plot(1e6*time, signal)

plt.show()

#def myMovAvg(x, N, x_prev=None, mode='same'):
#    
#    if x_prev is not None:
#        # Pad in front with x_prev signal
#        x = np.concatenate((x_prev, x))
#    
#    return np.convolve(x, np.ones(N)/N, mode=mode)
#time = np.linspace(0, 10.e-6, n)
#tau = 1.e-6
#signal = 1 - np.exp(-time/tau) + 0.01*(np.random.randn(n) - 0.5)
#time = np.linspace(0, 20.e-6, 2*n)
#signal = np.concatenate((signal, signal))
#
#plt.figure('mov avg2')
#plt.clf()
#plt.plot(1e6*time, signal)
#prev = np.zeros(n_ma-1)
#
#for i in range(iterations):
#    print("Average of end of previous signal", np.mean(prev))
#    tmp = signal[-n_ma+1:]
#    signal = myMovAvg(signal, n_ma, prev, mode='valid')
#    prev = np.copy(tmp)
#    print("Length of signal", len(signal))
#    plt.plot(1e6*time, signal)
#
#plt.show()
