
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Phase-space variable conversions related to action. Single-RF case without
intensity effects is considered.**

:Authors: **Helga Timko**
'''

from __future__ import division

from builtins import range

import numpy as np
from scipy.special import ellipe, ellipk


def x(phimax):

    return np.sin(0.5 * phimax)


def x2(phimax):

    return np.sin(0.5 * phimax)**2


def action_from_phase_amplitude(x2):
    '''
    Returns the relative action for given oscillation amplitude in time.
    Action is normalised to the value at the separatrix, given in units of 1. 
    '''

    action = np.zeros(len(x2))

    indices = np.where(x2 != 1.)[0]
    indices0 = np.where(x2 == 1.)[0]
    action[indices] = (ellipe(x2[indices]) -
                       (1. - x2[indices]) * ellipk(x2[indices]))

    if indices0:

        action[indices0] = np.float(ellipe(x2[indices0]))

    return action


def tune_from_phase_amplitude(phimax):
    '''
    Find the tune w.r.t. the central synchrotron frequency corresponding to a
    given amplitude of synchrotron oscillations in phase 
    '''

    return 0.5 * np.pi / ellipk(x(phimax))


def phase_amplitude_from_tune(tune):
    '''
    Find the amplitude of synchrotron oscillations in phase corresponding to a
    given tune w.r.t. the central synchrotron frequency
    '''

    n = len(tune)
    phimax = np.zeros(n)

    for i in range(n):

        if tune[i] == 1.:

            phimax[i] = 0.

        elif tune[i] == 0.:

            phimax[i] = np.pi

        else:

            guess = 0.5 * np.pi
            difference = 0.25 * np.pi
            k = 0

            while np.fabs(tune[i] - tune_from_phase_amplitude(guess)) / tune[i] \
                    > 0.001 and np.fabs(difference / guess) > 1.e-10:

                guess += np.sign(tune_from_phase_amplitude(guess)
                                 - tune[i]) * difference
                difference *= 0.5
                k += 1
                if k > 100:
                    # PhaseSpaceError
                    raise RuntimeError("Exceeded maximum number of iterations in phase_amplitude_from_tune()!")

            phimax[i] = guess

    return phimax


def oscillation_amplitude_from_coordinates(Ring, RFStation, dt, dE,
                                           timestep=0, Np_histogram=None):
    '''
    Returns the oscillation amplitude in time for given particle coordinates,
    assuming single-harmonic RF system and no intensity effects. 
    Optional: RF parameters at a given timestep (default = 0) are used.
    Optional: Number of points for histogram output
    '''

    omega_rf = RFStation.omega_rf[0, timestep]
    phi_rf = RFStation.phi_rf[0, timestep]
    phi_s = RFStation.phi_s[timestep]
    eta = RFStation.eta_0[0]
    T0 = Ring.t_rev[0]
    V = RFStation.voltage[0, 0]
    beta_sq = RFStation.beta[0]**2
    E = RFStation.energy[0]
    const = eta * T0 * omega_rf / (2. * V * beta_sq * E)

    dtmax = np.fabs(np.arccos(np.cos(omega_rf * dt + phi_rf) + const * dE**2)
                    - phi_rf - phi_s) / omega_rf

    if Np_histogram is not None:

        histogram, bins = np.histogram(dtmax, Np_histogram, (0,
                                                             np.pi / omega_rf))
        histogram = np.double(histogram) / np.sum(histogram[:])
        bin_centres = 0.5 * (bins[0:-1] + bins[1:])

        return dtmax, bin_centres, histogram

    else:

        return dtmax


def action_from_oscillation_amplitude(RFStation, dtmax, timestep=0,
                                      Np_histogram=None):
    '''
    Returns the relative action for given oscillation amplitude in time,
    assuming single-harmonic RF system and no intensity effects.
    Action is normalised to the value at the separatrix, given in units of 1. 
    Optional: RF parameters at a given timestep (default = 0) are used.
    Optional: Number of points for histogram output
    '''

    omega_rf = RFStation.omega_RF[0, timestep]
    xx = x2(omega_rf * dtmax)
    action = np.zeros(len(xx))

    indices = np.where(xx != 1.)[0]
    indices0 = np.where(xx == 1.)[0]
    action[indices] = (ellipe(xx[indices]) -
                       (1. - xx[indices]) * ellipk(xx[indices]))
    if indices0:

        action[indices0] = np.float(ellipe(xx[indices0]))

    if Np_histogram is not None:

        histogram, bins = np.histogram(action, Np_histogram, (0, 1))
        histogram = np.double(histogram) / np.sum(histogram[:])
        bin_centres = 0.5 * (bins[0:-1] + bins[1:])

        return action, bin_centres, histogram

    else:

        return action
