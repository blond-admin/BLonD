# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public License version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.md.
# In applying this license, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
** CERN Tomoscope-related functions to generate particle distribution or 
probability density.**

:Authors: **Helga Timko**
'''


from __future__ import division, print_function

import linecache
import os
from builtins import str

import h5py as hp
import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt

from ..utils import bmath as bm


def distribution_from_tomoscope_data(dataDir, nPart, cutoff=1000, seed=1234,
                                     plotFig=True, saveDistr=False):
    '''
    'dataDir' is the directory of *directories* of offline-processed tomoscope 
    data containing 'plotinfo.data' and 'image001.data'
    Use 'cutoff' to eliminate measurement noise (use values around 50-100).
    Use 'seed' to change the random number seed.
    Use 'plotFig = True' to plot and save figures.
    Use 'saveDistr = True' to save particle coordinates.    
    '''

    # Directory in which plots will be stored
    distrDir = dataDir + '\\Distributions_' + \
        str(nPart) + 'particles_cutoff' + str(cutoff)
    # Try to create directory
    try:
        os.makedirs(distrDir)
    # Check whether already exists/creation failed
    except OSError:
        if os.path.exists(distrDir):
            pass
        else:
            raise

    # Loop over directories to be analysed
    for directory in os.listdir(dataDir):

        if 'Distributions' not in directory:

            print("Analysing data of directory %s" % directory)

            # Read tomoscope settings
            plotInfo = dataDir + '\\' + directory + '\\plotinfo.data'
            profLen = np.uint(linecache.getline(plotInfo, 4)[17:-1])
            dtBin = np.double(linecache.getline(plotInfo, 6)[9:-1])
            dEBin = np.double(linecache.getline(plotInfo, 8)[9:-1])
            x0 = np.double(linecache.getline(plotInfo, 12)[8:-1])
            y0 = np.double(linecache.getline(plotInfo, 13)[8:-1])

            # Read probability density from file
            probDistr = np.loadtxt(dataDir + '\\' + directory + '\\image001.data',
                                   dtype=np.double, unpack=True)
            probDistr = np.ascontiguousarray(probDistr)

            # Reconstruct particle distribution from probability distribution
            dt = np.empty(nPart)
            dE = np.empty(nPart)

            bm.distribution_from_tomoscope(dt, dE, probDistr, seed, profLen,
                                           cutoff, x0, y0, dtBin, dEBin)

            if plotFig:

                # Settings for plots
                plt.rc('axes', labelsize=14, labelweight='normal')
                plt.rc('lines', linewidth=1.5, markersize=6)
                plt.rc('font', family='sans-serif')
                plt.rc('legend', fontsize=12)

                # Plot distribution
                plt.hist2d(dt, dE, (profLen, profLen), cmap=cm.jet)
                plt.colorbar()
                plt.xlabel('Time offset [s]')
                plt.ylabel('Energy offset [eV]')
                plt.savefig(distrDir + '\\' + directory + '.png')
                plt.clf()

            if saveDistr:

                h5File = hp.File(distrDir + '\\' + directory + '.h5', 'w')

                # Create group
                h5File.require_group('Beam')
                h5Group = h5File['Beam']

                # Create & write datasets
                h5Group.create_dataset("dt", shape=(nPart,), dtype='f',
                                       compression="gzip", compression_opts=9)
                h5Group["dt"][:] = dt
                h5Group.create_dataset("dE", shape=(nPart,), dtype='f',
                                       compression="gzip", compression_opts=9)
                h5Group["dE"][:] = dE
                h5File.close()
