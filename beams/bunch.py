'''
Created on 06.01.2014

@author: Kevin Li
'''

import numpy as np
import copy, h5py, sys
from scipy.constants import c, e, epsilon_0, m_e, m_p, pi
from beams.slices import *
from beams.matching import match_transverse, match_longitudinal, unmatched_inbucket


def bunch_matched_and_sliced(n_macroparticles, n_particles, charge, gamma, mass, 
                                    epsn_x, epsn_y, ltm, bunch_length,
                                    bucket, matching, n_slices, nsigmaz, slicemode='cspace'):

    bunch = Bunch(n_macroparticles, n_particles, charge, gamma, mass,
                 distribution='gauss')
    bunch.match_transverse(epsn_x, epsn_y, ltm)
    bunch.match_longitudinal(bunch_length, bucket, matching)
    slices = Slices(n_slices, nsigmaz, slicemode)
    slices.update_slices(bunch)

    return bunch, slices

def bunch_unmatched_inbucket_sliced(n_macroparticles, n_particles, charge, gamma, mass, 
                                    epsn_x, epsn_y, ltm, sigma_dz, sigma_dp, bucket,
                                    n_slices, nsigmaz, slicemode='cspace'):
    
    bunch = Bunch(n_macroparticles, n_particles, charge, gamma, mass,
                 distribution='gauss')
    bunch.match_transverse(epsn_x, epsn_y, ltm)
    bunch.unmatched_inbucket(sigma_dz, sigma_dp, bucket)
    slices = Slices(n_slices, nsigmaz, slicemode)
    slices.update_slices(bunch)

    return bunch, slices

class Bunch(object):

    def __init__(self, n_macroparticles, n_particles, charge, gamma, mass,
                 distribution='empty'):

        if distribution == 'empty':
            _create_empty(n_macroparticles)
        elif distribution == 'gauss':
            _creat_gauss(n_macroparticles)
        elif distribution == "uniform":
            _create_uniform(n_macroparticles)

        self.id = np.arange(1, n_particles + 1, dtype=int)

        _set_beam_physics(n_particles, charge, gamma, mass)
        
        self.x0 = self.x.copy()
        self.xp0 = self.xp.copy()
        self.y0 = self.y.copy()
        self.yp0 = self.yp.copy()
        self.z0 = self.z.copy()
        self.dp0 = self.dp.copy()

    def _create_empty(self, n_macroparticles):

        self.x = np.zeros(n_macroparticles)
        self.xp = np.zeros(n_macroparticles)
        self.y = np.zeros(n_macroparticles)
        self.yp = np.zeros(n_macroparticles)
        self.z = np.zeros(n_macroparticles)
        self.dp = np.zeros(n_macroparticles)

    def _create_gauss(self, n_macroparticles):

        self.x = np.random.randn(n_macroparticles)
        self.xp = np.random.randn(n_macroparticles)
        self.y = np.random.randn(n_macroparticles)
        self.yp = np.random.randn(n_macroparticles)
        self.z = np.random.randn(n_macroparticles)
        self.dp = np.random.randn(n_macroparticles)

    def _create_uniform(self, n_macroparticles):

        self.x = 2 * np.random.rand(n_macroparticles) - 1
        self.xp = 2 * np.random.rand(n_macroparticles) - 1
        self.y = 2 * np.random.rand(n_macroparticles) - 1
        self.yp = 2 * np.random.rand(n_macroparticles) - 1
        self.z = 2 * np.random.rand(n_macroparticles) - 1
        self.dp = 2 * np.random.rand(n_macroparticles) - 1

    def _set_beam_physics(self, n_particles, charge, gamma, mass):

        self.n_particles = n_particles
        self.charge = charge
        self.gamma = gamma
        self.mass = mass

    
    def n_macroparticles(self):

        return len(self.x)

 
    def beta(self):

        return np.sqrt(1 - 1 / self.gamma ** 2)

   
    def p0(self):

        return self.mass * self.gamma * self.beta * c

    def reinit(self):

        np.copyto(self.x, self.x0)
        np.copyto(self.xp, self.xp0)
        np.copyto(self.y, self.y0)
        np.copyto(self.yp, self.yp0)
        np.copyto(self.z, self.z0)
        np.copyto(self.dp, self.dp0)

  
    def sort_particles(self):
        # update the number of lost particles
        self.n_macroparticles_lost = (self.n_macroparticles - np.count_nonzero(self.id))

        # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)
        if self.n_macroparticles_lost:
            dz_argsorted = np.lexsort((self.dz, -np.sign(self.id))) # place lost particles at the end of the array
        else:
            dz_argsorted = np.argsort(self.dz)

        self.x = self.x.take(dz_argsorted)
        self.xp = self.xp.take(dz_argsorted)
        self.y = self.y.take(dz_argsorted)
        self.yp = self.yp.take(dz_argsorted)
        self.dz = self.dz.take(dz_argsorted)
        self.dp = self.dp.take(dz_argsorted)
        self.id = self.id.take(dz_argsorted)



