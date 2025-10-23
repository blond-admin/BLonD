
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

from builtins import str

import numpy as np
import numpy.random as rand

import blond.utils.exceptions as blExcept



def generate_coasting_beam(beam, t_start, t_stop, spread=1E-3,
                           spread_type='dp/p', energy_offset=0,
                           distribution='gaussian', user_distribution=None,
                           user_probability=None):
    '''
    energy_offset represents the absolute energy difference between the center
    of the coasting beam and the synchronous particle spread is used for the
    absolute max/min in a parabolic or the standard deviation in a gaussian
    distribution spread_type is used to convert the passed spread into dE
    according to dE/E = beta**2 * dP/P energy_offset gives an offset in dE for
    the two standard distributions if a user_distribution is used it is taken as
    being in dE

    :param beam: _description_
    :type beam: _type_
    :param t_start: _description_
    :type t_start: _type_
    :param t_stop: _description_
    :type t_stop: _type_
    :param spread: _description_, defaults to 1E-3
    :type spread: _type_, optional
    :param spread_type: _description_, defaults to 'dp/p'
    :type spread_type: str, optional
    :param energy_offset: _description_, defaults to 0
    :type energy_offset: int, optional
    :param distribution: _description_, defaults to 'gaussian'
    :type distribution: str, optional
    :param user_distribution: _description_, defaults to None
    :type user_distribution: _type_, optional
    :param user_probability: _description_, defaults to None
    :type user_probability: _type_, optional
    :raises blExcept.DistributionError: _description_
    :raises blExcept.DistributionError: _description_
    :raises blExcept.DistributionError: _description_
    '''
    if spread_type == 'dp/p':
        energy_spread = beam.energy * beam.beta**2 * spread
    elif spread_type == 'dE/E':
        energy_spread = spread * beam.energy
    elif spread_type == 'dp':
        energy_spread = beam.energy * beam.beta**2 * spread / beam.momentum
    elif spread_type == 'dE':
        energy_spread = spread
    else:
        raise blExcept.DistributionError("spread_type " + str(spread_type) +
                                         " not recognised")

    if distribution == 'gaussian':
        beam.dE = rand.normal(loc=energy_offset, scale=energy_spread,
                              size=beam.n_macroparticles)

    elif distribution == 'parabolic':
        energy_range = np.linspace(-energy_spread, energy_spread, 10000)
        probability_distribution = 1 - (energy_range / energy_spread)**2
        probability_distribution /= np.cumsum(probability_distribution)[-1]
        beam.dE = rand.choice(energy_range, size=beam.n_macroparticles,
                              p=probability_distribution) \
            + (rand.rand(beam.n_macroparticles) - 0.5) \
            * (energy_range[1] - energy_range[0]) \
            + energy_offset

    # If distribution == 'user' is selected the user must supply a uniformly
    # spaced distribution and the assosciated probability for each bin
    # momentum_spread and energy_offset are not used in this instance.
    elif distribution == 'user':
        if user_distribution is None or user_probability is None:
            raise blExcept.DistributionError("""Distribution 'user' requires
                                             'user_distribution' and 
                                             'user_probability' to be defined""")

        beam.dE = rand.choice(user_distribution, size=beam.n_macroparticles,
                              p=user_probability) \
            + (rand.rand(beam.n_macroparticles) - 0.5) \
            * (user_distribution[1] - user_distribution[0])

    else:
        raise blExcept.DistributionError("distribution type not recognised")

    beam.dt = rand.rand(beam.n_macroparticles) * (t_stop - t_start) + t_start
