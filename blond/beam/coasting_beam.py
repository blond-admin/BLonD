
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

#General imports
from builtins import str
from builtins import range
import numpy as np
import warnings
import copy
import matplotlib.pyplot as plt
import numpy.random as rand

#BLonD imports
import blond.utils.exceptions as blExcept



#energy_offset represents the absolute energy difference between the center 
#of the coasting beam and the synchronous particle spread is used for the 
#absolute max/min in a parabolic or the standard deviation in a gaussian 
#distribution spread_type is used to convert the passed spread into dE
#according to dE/E = beta**2 * dP/P energy_offset gives an offset in dE for
#the two standard distributions if a user_distribution is used it is taken as
#being in dE
def generate_coasting_beam(Beam, t_start, t_stop, spread = 1E-3, 
                           spread_type = 'dp/p', energy_offset = 0, 
                           distribution = 'gaussian' , user_distribution = None,
                           user_probability = None):

    if spread_type == 'dp/p':
        energy_spread = Beam.energy * Beam.beta**2 * spread
    elif spread_type == 'dE/E':
        energy_spread = spread*Beam.energy
    elif spread_type == 'dp':
        energy_spread = Beam.energy * Beam.beta**2 * spread / Beam.momentum
    elif spread_type == 'dE':
        energy_spread = spread
    else:
        raise blExcept.DistributionError("spread_type " + str(spread_type) + \
                                   " not recognised")


    if distribution == 'gaussian':
        Beam.dE = rand.normal(loc = energy_offset, scale = energy_spread, \
                        size = Beam.n_macroparticles)


    elif distribution == 'parabolic':
        energyRange = np.linspace(-energy_spread, energy_spread, 10000)
        probabilityDistribution = 1 - (energyRange/energy_spread)**2
        probabilityDistribution /= np.cumsum(probabilityDistribution)[-1]
        Beam.dE = rand.choice(energyRange, size = Beam.n_macroparticles, \
                        p = probabilityDistribution) \
                            + (rand.rand(Beam.n_macroparticles) - 0.5) \
                            * (energyRange[1] - energyRange[0]) \
                            + energy_offset

    #If distribution == 'user' is selected the user must supply a uniformly
    #spaced distribution and the assosciated probability for each bin
    #momentum_spread and energy_offset are not used in this instance.
    elif distribution == 'user':
        if user_distribution is None or user_probability is None:
            raise blExcept.DistributionError("""Distribution 'user' requires
                                             'user_distribution' and 
                                             'user_probability' to be defined""")
            
        Beam.dE = rand.choice(user_distribution, size = Beam.n_macroparticles, \
                              p = user_probability) \
                              + (rand.rand(Beam.n_macroparticles) - 0.5) \
                              * (user_distribution[1] - user_distribution[0])

    else:
        raise blExcept.DistributionError("distribution type not recognised")

    Beam.dt = rand.rand(Beam.n_macroparticles)*(t_stop - t_start) + t_start
